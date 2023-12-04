import os

import torch
import torch.nn as nn
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed import comm as dist
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.pipe.engine import PipelineEngine, TRAIN_BATCH_TIMER, PIPE_SEND_OUTPUT_TIMER, PIPE_SEND_GRAD_TIMER, PIPE_RECV_INPUT_TIMER, PIPE_RECV_GRAD_TIMER
from deepspeed.runtime.pipe import schedule
from deepspeed.runtime.utils import PartitionedTensor
from deepspeed.runtime.activation_checkpointing import checkpointing as ds_checkpointing
from deepspeed.runtime.pipe.module import PipelineModule
from deepspeed.runtime.pipe.topology import PipeDataParallelTopology, PipelineParallelGrid


def initialize(args=None,
               model=None,
               model_parameters=None,
               optimizer=None):
    assert model is not None, "deepspeed.initialize requires a model"

    dist_backend = get_accelerator().communication_backend_name()
    dist.init_distributed(dist_backend=dist_backend)

    config = args.deepspeed_config
    assert config is not None, "DeepSpeed requires --deepspeed_config to specify configuration file"

    mpu = model.mpu()
    config_class = DeepSpeedConfig(config, mpu)
    engine = CustomPipelineEngine(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model_parameters,
        mpu=mpu,
        config=config,
        config_class=config_class
    )

    return engine, engine.optimizer


class CustomPipelineEngine(PipelineEngine):


    def train_batch(self):
        if not torch._C.is_grad_enabled():
            raise RuntimeError(f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        # sequence length may change between macro batches (but not between gradient accumulation steps)
        self.reset_activation_shape()

        self.module.train()
        self._compute_loss = True

        # Do the work
        self.timers(TRAIN_BATCH_TIMER).start()
        sched = schedule.TrainSchedule(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self._exec_schedule(sched)
        agg_losses = self._aggregate_total_losses()
        # Actual training loss is always the first item.
        self.agg_train_loss = agg_losses[0].mean()

        self.timers(TRAIN_BATCH_TIMER).stop()

        if self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                elapsed = self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True) / 1000.0
                iter_time = elapsed / self.steps_per_print()
                tput = self.train_batch_size() / iter_time
                print(f'steps: {self.global_steps} '
                      f'loss: {self.agg_train_loss:0.4f} '
                      f'iter time (s): {iter_time:0.3f} '
                      f'samples/sec: {tput:0.3f}')
            else:
                self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True)

        # Monitoring
        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/train_loss', self.agg_train_loss.mean().item(),
                                    self.global_samples)]
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown() and self.global_steps % self.steps_per_print() == 0:
            self.timers.log([
                PIPE_SEND_OUTPUT_TIMER,
                PIPE_SEND_GRAD_TIMER,
                PIPE_RECV_INPUT_TIMER,
                PIPE_RECV_GRAD_TIMER,
            ])

        return agg_losses


    def eval_batch(self, data_iter):
        self.module.eval()
        self._compute_loss = True

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=self.micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        with torch.no_grad():
            self._exec_schedule(sched)

        # list of losses
        agg_eval_losses = self._aggregate_total_losses()

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/eval_loss', agg_eval_losses[0].mean().item(), self.global_samples)]
            self.monitor.write_events(self.summary_events)

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        return agg_eval_losses


    def _aggregate_total_losses(self):
        all_agg_outputs = []
        # gather each output for all the gradient accumulation steps
        grouped_outputs = [list(x) for x in zip(*self.fwd_outputs)]
        # if any are scalar, make them dim 1 so we can concat across DP ranks
        for outputs in grouped_outputs:
            for i, output in enumerate(outputs):
                if output.dim() == 0:
                    outputs[i] = torch.unsqueeze(output, 0)

        if self.is_last_stage():
            agg_sizes = []
            # loop to gather all the outputs across DP ranks
            for outputs in grouped_outputs:
                # concat all the grad_accum_steps
                concat_outputs = torch.cat(outputs)
                if self.is_data_parallel:
                    # might be different sizes across DP ranks, so, gather all the sizes
                    sizes = [None] * self.grid.get_data_parallel_world_size()
                    torch.distributed.all_gather_object(sizes, concat_outputs.size(), group=self.grid.get_data_parallel_group())
                    # once we know all the sizes we can gather the results across DP ranks
                    gather_result = [torch.zeros(size).to(self.device) for size in sizes]
                    dist.all_gather(gather_result, concat_outputs, group=self.grid.get_data_parallel_group())
                    # and finally, concat
                    agg_output = torch.cat(gather_result)
                else:
                    agg_output = concat_outputs
                agg_sizes.append(agg_output.size())
                all_agg_outputs.append(agg_output)

            # send the sizes, then broadcast to the PP ranks
            if self.is_pipe_parallel:
                torch.distributed.broadcast_object_list([agg_sizes], src=self.global_rank, group=self.grid.get_pipe_parallel_group())
                for agg_output in all_agg_outputs:
                    dist.broadcast(tensor=agg_output, src=self.global_rank, group=self.grid.get_pipe_parallel_group())
        else:
            # get the outputs from the last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            result = [None]
            torch.distributed.broadcast_object_list(result, src=src_rank, group=self.grid.get_pipe_parallel_group())
            agg_sizes = result[0]
            for agg_size in agg_sizes:
                agg_output = torch.zeros(agg_size).to(self.device)
                dist.broadcast(tensor=agg_output, src=src_rank, group=self.grid.get_pipe_parallel_group())
                all_agg_outputs.append(agg_output)

        return all_agg_outputs


    # We override this to handle the model returning a list of "losses", but only doing backprop on the first.
    def _exec_forward_pass(self, buffer_id):
        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)

        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()

        # collect the partitioned input from the previous stage
        if self.is_pipe_partitioned and not self.is_first_stage():
            part_input = PartitionedTensor.from_meta(meta=inputs[0],
                                                     local_part=inputs[1],
                                                     group=self.grid.get_slice_parallel_group())

            inputs = (part_input.full(), *inputs[2:])
            inputs[0].requires_grad = True
            # skip mask
            #inputs[1].requires_grad = True
            part_input = None
            inputs = inputs[0] if len(inputs) == 1 else inputs
            self.pipe_buffers['inputs'][buffer_id] = inputs

        # inputs has no gradient because it is from a cloned tensor
        outputs = super(PipelineEngine, self).forward(inputs)

        # Reset activation checkpointing buffers.
        # Need to call this between evaluation iterations
        if not self.module.training:
            ds_checkpointing.reset()

        # Partition the outputs if we are not the last stage
        if self.is_pipe_partitioned and not self.is_last_stage():
            if isinstance(outputs, tuple):
                first_output = outputs[0]
                # TODO: Improve pipe partitioning to pass multiple tensors that require grads
                assert all([torch.is_tensor(elt) and elt.requires_grad is False for elt in outputs[1:]])
                outputs_tail = outputs[1:]
            elif torch.is_tensor(outputs):
                first_output = outputs
                outputs_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            part = PartitionedTensor(tensor=first_output, group=self.grid.get_slice_parallel_group())
            # Clear the large output data, but save the computation graph
            first_output.data = torch.zeros(1)
            self.pipe_buffers['output_tensors'][buffer_id] = first_output
            # Inject the partitioned tensor into the output before sending
            outputs = (part.to_meta(), part.data(), *outputs_tail)
            part = None

        self.pipe_buffers['outputs'][buffer_id] = outputs

        # Optionally compute loss on the last device
        if self.is_last_stage():
            if self._compute_loss and self.module.loss_fn is not None:
                labels = self.pipe_buffers['labels'][buffer_id]
                losses = self.module.loss_fn(outputs, labels)
            else:
                # Some models just return loss from forward()
                losses = outputs
            if self.eval_return_logits:
                self.outputs = outputs
            if isinstance(losses, torch.Tensor):
                self.loss = losses
                self.fwd_outputs.append([self.loss.detach()])
            else:
                self.loss = losses[0]
                self.fwd_outputs.append([l.detach() for l in losses])

    # make our forward pass method apply
    PipelineEngine._INSTRUCTION_MAP[schedule.ForwardPass] = _exec_forward_pass


# This is just the Deepspeed PipelineModule, but with initialization split into two methods. We
# need this because part of the initialization (in __init__()) needs all processes to call it
# together, but actually setting up the layers we want do one process at a time to save RAM.
class CustomPipelineModule(PipelineModule):

    def __init__(self,
                 num_stages=None,
                 topology=None,
                 loss_fn=None,
                 seed_layers=False,
                 seed_fn=None,
                 base_seed=1234,
                 partition_method='parameters',
                 activation_checkpoint_interval=0,
                 activation_checkpoint_func=ds_checkpointing.checkpoint,
                 checkpointable_layers=None):

        super(PipelineModule, self).__init__()

        if num_stages is None and topology is None:
            raise RuntimeError('must provide num_stages or topology')

        self.micro_offset = 0

        self.loss_fn = loss_fn

        self.checkpointable_layers = checkpointable_layers
        if checkpointable_layers is not None:
            assert isinstance(checkpointable_layers, list), "param `checkpointable_layers` must be type of list."

        self.seed_layers = seed_layers
        self.seed_fn = seed_fn
        self.base_seed = base_seed
        if dist.get_rank() == 0:
            try:
                seed_str = self.seed_fn.__name__
            except AttributeError:
                seed_str = None
            print(f'SEED_LAYERS={self.seed_layers} BASE_SEED={self.base_seed} SEED_FN={seed_str}')

        # Setup world info
        self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
        self.global_rank = dist.get_rank(group=self.world_group)
        self.world_size = dist.get_world_size(group=self.world_group)
        self.local_rank = int(os.environ.get("LOCAL_RANK", None))
        assert self.local_rank is not None

        if topology:
            self._topo = topology
            self.num_stages = self._topo.get_dim('pipe')
        else:
            self.num_stages = num_stages
            if topology is None:
                if self.world_size % self.num_stages != 0:
                    raise RuntimeError(
                        f'num_stages ({self.num_stages}) must divide distributed world size ({self.world_size})')
                dp = self.world_size // num_stages
                topology = PipeDataParallelTopology(num_pp=num_stages, num_dp=dp)
                self._topo = topology

        # Construct communicators for pipeline topology
        self._grid = PipelineParallelGrid(process_group=self.world_group, topology=self._topo)

        self.partition_method = partition_method
        self.stage_id = self._topo.get_coord(self.global_rank).pipe
        self.activation_checkpoint_interval = activation_checkpoint_interval
        self.activation_checkpoint_func = activation_checkpoint_func
        # if configuration use_reentrant = False, self.activation_checkpoint_func will be set to ``checkpointing.non_reentrant_checkpoint``


    def init_layers(self, layers):
        # Initialize partition information
        self._layer_specs = list(layers)
        self._num_layers = len(self._layer_specs)
        self._local_start = 0
        self._local_stop = None
        self._partition_layers(method=self.partition_method)

        self.forward_funcs = []
        self.fwd_map = {}
        self.tied_modules = nn.ModuleDict()
        self.tied_weight_attrs = {}

        # Offset the random seed by the stage ID.
        #newseed = get_accelerator().initial_seed() + self._grid.get_stage_id()
        #ds_utils.set_random_seed(newseed)

        #with torch.random.fork_rng(devices=[get_accelerator().current_device_name()]):
        self._build()

        self.tied_comms = self._index_tied_modules()
        self._synchronize_tied_weights()

        # Free memory. We don't need all the layers anymore.
        del self._layer_specs


    def move_layers_to_device(self):
        self.to(get_accelerator().device_name(self.local_rank))
