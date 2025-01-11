from collections import deque

import deepspeed
import torch
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime import utils as ds_utils
from deepspeed.runtime.activation_checkpointing import checkpointing as ds_checkpointing
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.pipe import p2p, schedule
from deepspeed.runtime.pipe.engine import (
    BATCH_INPUT_TIMER,
    PIPE_RECV_GRAD_TIMER,
    PIPE_RECV_INPUT_TIMER,
    PIPE_SEND_GRAD_TIMER,
    PIPE_SEND_OUTPUT_TIMER,
    TRAIN_BATCH_TIMER,
    PipelineEngine,
)
from deepspeed.runtime.pipe.module import LayerSpec, PipelineModule
from deepspeed.runtime.pipe.schedule import (
    BackwardPass,
    BufferOpInstruction,
    ForwardPass,
    OptimizerStep,
    PipeInstruction,
    PipeSchedule,
    RecvActivation,
    RecvGrad,
    ReduceGrads,
    ReduceTiedGrads,
    SendActivation,
    SendGrad,
    _is_even,
    _is_odd,
)
from deepspeed.runtime.pipe.topology import ProcessTopology
from deepspeed.runtime.utils import PartitionedTensor
from torch import nn

from utils import eta_str, log


def initialize(args=None,
               model=None,
               model_parameters=None,
               optimizer=None,
               lora_model=None,
               config=None,
               tokenizer=None):
    assert model is not None, "deepspeed.initialize requires a model"

    dist_backend = get_accelerator().communication_backend_name()
    dist.init_distributed(dist_backend=dist_backend)

    if hasattr(args, "deepspeed_config") and args.deepspeed_config is not None:
        config = args.deepspeed_config

    mpu = model.mpu()
    config_class = DeepSpeedConfig(config, mpu)
    engine = CustomPipelineEngine(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model_parameters,
        mpu=mpu,
        config=config,
        config_class=config_class,
        lora_model=lora_model,
        tokenizer=tokenizer,
    )

    return engine, engine.optimizer


class LoadMicroBatchMultipleBuffers(PipeInstruction):
    def __init__(self, *buffer_ids, **kwargs):
        super().__init__(buffer_ids=buffer_ids, **kwargs)


class ReferenceLogitsForwardPass(BufferOpInstruction):
    pass


class CustomPipelineEngine(PipelineEngine):
    def __init__(self, *args, lora_model=None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_steps = None
        self.etas = deque()
        self.rl_config = {}
        # Assign list to avoid registering the nn.Module
        self.lora_model = [lora_model]
        self.tokenizer = tokenizer
        eos_token_ids = set()
        if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
            eos_token_ids.add(self.tokenizer.eos_token_id)
        model_config = self.module.model.config
        if model_config.eos_token_id:
            model_eos_token_ids = model_config.eos_token_id
            if isinstance(model_eos_token_ids, int):
                model_eos_token_ids = [model_eos_token_ids]
            eos_token_ids.update(model_eos_token_ids)
        self.eos_token_ids = eos_token_ids


    def configure_rl(self, rl_config):
        self.rl_config = rl_config


    def train_batch(self):
        if not torch._C.is_grad_enabled():
            raise RuntimeError('train_batch() requires gradients enabled. Use eval_batch() instead.')

        # sequence length may change between macro batches (but not between gradient accumulation steps)
        self.reset_activation_shape()

        self.module.train()
        self._compute_loss = True

        # Do the work
        self.timers(TRAIN_BATCH_TIMER).start()
        if self.rl_config.get('method', None) == 'dpo':
            sched = DPOTrainSchedule(micro_batches=self.micro_batches, stages=self.num_stages, stage_id=self.stage_id)
        else:
            sched = schedule.TrainSchedule(micro_batches=self.micro_batches, stages=self.num_stages, stage_id=self.stage_id)
        self._exec_schedule(sched)
        agg_losses = self._aggregate_total_losses()
        # Actual training loss is always the first item.
        self.agg_train_loss = agg_losses[0].mean()

        self.timers(TRAIN_BATCH_TIMER).stop()

        if self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                elapsed = self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True) / 1000.0
                iter_time = elapsed / self.steps_per_print()
                eta = iter_time * (self.total_steps - self.global_steps)
                self.etas.append(eta)
                while len(self.etas) > 10:
                    self.etas.popleft()
                rolling_eta = sum(self.etas) / len(self.etas)
                tput = self.train_batch_size() / iter_time
                log(f'step: {self.global_steps:>5} / {self.total_steps:>5} '
                    f'loss: {self.agg_train_loss:0.4f} '
                    f'iter time (s): {iter_time:0.3f} '
                    f'samples/sec: {tput:0.3f} '
                    f'eta: {eta_str(rolling_eta)} ')
            else:
                self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True)

        # Monitoring
        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [('Train/Samples/train_loss', self.agg_train_loss.mean().item(),
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
        # sequence length may change between macro batches (but not between gradient accumulation steps)
        self.reset_activation_shape()

        self.module.eval()
        self._compute_loss = True

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # Do the work
        if self.rl_config.get('method', None) == 'dpo':
            sched = DPOInferenceSchedule(micro_batches=self.micro_batches, stages=self.num_stages, stage_id=self.stage_id)
        else:
            sched = schedule.InferenceSchedule(micro_batches=self.micro_batches, stages=self.num_stages, stage_id=self.stage_id)

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        with torch.no_grad():
            self._exec_schedule(sched)

        # list of losses
        agg_eval_losses = self._aggregate_total_losses()

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [('Train/Samples/eval_loss', agg_eval_losses[0].mean().item(), self.global_samples)]
            self.monitor.write_events(self.summary_events)

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        return agg_eval_losses


    def sample_batch(self, prompts, max_new_tokens=1e9):
        assert isinstance(prompts, (list, tuple))
        self.reset_activation_shape()
        self.module.eval()
        self.module.set_sampling_mode(True)
        train_iterator = self.data_iterator
        original_micro_batches = self.micro_batches
        self.micro_batches = len(prompts)
        dist.barrier()

        if self.is_first_stage():
            # Tokenizer returns dict with 'input_ids', 'attention_mask' keys.
            # Tensors have batch dimension because we pass list of prompts.
            examples = []
            for prompt in prompts:
                if not isinstance(prompt, (list, tuple)):
                    prompt = [prompt]
                examples.append(self.tokenizer(prompt, return_tensors='pt', padding=True))
        else:
            examples = None
        with torch.no_grad():
            examples = self._exec_sampling_schedule(examples, max_new_tokens=max_new_tokens)
        text = [self.tokenizer.batch_decode(example['input_ids']) for example in examples]
        self.set_dataiterator(train_iterator)
        self.micro_batches = original_micro_batches
        self.module.set_sampling_mode(False)
        return text


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
                assert all(torch.is_tensor(elt) and elt.requires_grad is False for elt in outputs[1:])
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


    def _exec_load_micro_batch_multiple_buffers(self, buffer_ids):
        if self.wall_clock_breakdown():
            self.timers(BATCH_INPUT_TIMER).start()

        batch = self._next_batch()

        if self.is_first_stage():
            loaded = None
            if torch.is_tensor(batch[0]):
                loaded = batch[0].clone().to(self.device).detach()
                if self._config.pipeline['activation_checkpoint_interval'] > 0 and self._config.pipeline[
                        'use_reentrant']:
                    loaded.requires_grad = loaded.is_floating_point()
            else:
                assert isinstance(batch[0], (tuple, list))
                # Assume list or tuple
                loaded = []
                for x in batch[0]:
                    assert torch.is_tensor(x)
                    mine = x.clone().detach().to(self.device)
                    if self._config.pipeline['activation_checkpoint_interval'] > 0 and self._config.pipeline[
                            'use_reentrant']:
                        mine.requires_grad = mine.is_floating_point()
                    loaded.append(mine)
                loaded = tuple(loaded)

            for buffer_id in buffer_ids:
                self.pipe_buffers['inputs'][buffer_id] = loaded

        if self.is_last_stage():
            loaded = batch[1]
            if torch.is_tensor(batch[1]):
                loaded = batch[1].to(self.device)
            # XXX: torch 1.6.0 DataLoader will auto convert tuple to list
            elif isinstance(batch[1], (tuple, list)):
                loaded = []
                for x in batch[1]:
                    assert torch.is_tensor(x)
                    x = x.to(self.device).detach()
                    loaded.append(x)
                loaded = tuple(loaded)

            for buffer_id in buffer_ids:
                self.pipe_buffers['labels'][buffer_id] = loaded

        if self.wall_clock_breakdown():
            self.timers(BATCH_INPUT_TIMER).stop()


    @torch.no_grad()
    def _exec_reference_logits_forward_pass(self, buffer_id):
        self.lora_model[0].disable_adapter_layers()
        self.module.set_dpo_reference_mode(True)
        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()

        # collect the partitioned input from the previous stage
        if self.is_pipe_partitioned and not self.is_first_stage():
            if self.pipe_partition_input_meta_cache is None:
                self.pipe_partition_input_meta_cache = inputs[0].to('cpu')
            part_input = PartitionedTensor.from_meta(meta=self.pipe_partition_input_meta_cache,
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
                assert all(torch.is_tensor(elt) and elt.requires_grad is False for elt in outputs[1:])
                outputs_tail = outputs[1:]
            elif torch.is_tensor(outputs):
                first_output = outputs
                outputs_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            part = PartitionedTensor(tensor=first_output, group=self.grid.get_slice_parallel_group())
            # Clear the large output data, but save the computation graph
            first_output.data = torch.zeros(1, device=first_output.data.device)
            self.pipe_buffers['output_tensors'][buffer_id] = first_output
            # Inject the partitioned tensor into the output before sending
            outputs = (part.to_meta(), part.data(), *outputs_tail)
            part = None

        self.pipe_buffers['outputs'][buffer_id] = outputs
        self.lora_model[0].enable_adapter_layers()
        self.module.set_dpo_reference_mode(False)


    def _exec_send_micro_batch_id(self, send_micro_batch_id):
        assert isinstance(send_micro_batch_id, int)
        if self.num_stages == 1:
            return send_micro_batch_id
        send_micro_batch_id = torch.tensor(send_micro_batch_id)
        recv_micro_batch_id = torch.tensor(-1)
        if _is_even(self.stage_id):
            if not self.is_last_stage():
                p2p.send(send_micro_batch_id, self.next_stage)
            if not self.is_first_stage():
                p2p.recv(recv_micro_batch_id, self.prev_stage)
        else:
            if not self.is_first_stage():
                p2p.recv(recv_micro_batch_id, self.prev_stage)
            if not self.is_last_stage():
                p2p.send(send_micro_batch_id, self.next_stage)
        # last stage sends to first stage
        if self.is_first_stage():
            p2p.recv(recv_micro_batch_id, self.num_stages-1)
        if self.is_last_stage():
            p2p.send(send_micro_batch_id, 0)
        return recv_micro_batch_id.item()


    def _exec_load_micro_batch_for_sampling(self, buffer_id, inputs):
        loaded = (
            inputs['input_ids'],
            inputs['attention_mask'],
            torch.tensor([0]),  # labels must be provided, so use a dummy
            inputs['micro_batch_id'],
        )
        loaded = tuple(x.clone().detach().to(self.device) for x in loaded)
        self.pipe_buffers['inputs'][buffer_id] = loaded


    @torch.no_grad()
    def _exec_sampling_forward_pass(self, buffer_id):
        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()

        # collect the partitioned input from the previous stage
        if self.is_pipe_partitioned and not self.is_first_stage():
            if self.pipe_partition_input_meta_cache is None:
                self.pipe_partition_input_meta_cache = inputs[0].to('cpu')
            part_input = PartitionedTensor.from_meta(meta=self.pipe_partition_input_meta_cache,
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
                assert all(torch.is_tensor(elt) and elt.requires_grad is False for elt in outputs[1:])
                outputs_tail = outputs[1:]
            elif torch.is_tensor(outputs):
                first_output = outputs
                outputs_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            part = PartitionedTensor(tensor=first_output, group=self.grid.get_slice_parallel_group())
            # Clear the large output data, but save the computation graph
            first_output.data = torch.zeros(1, device=first_output.data.device)
            self.pipe_buffers['output_tensors'][buffer_id] = first_output
            # Inject the partitioned tensor into the output before sending
            outputs = (part.to_meta(), part.data(), *outputs_tail)
            part = None

        self.pipe_buffers['outputs'][buffer_id] = outputs


    def _sample_from_logits(self, buffer_id):
        logits = self.pipe_buffers['outputs'][buffer_id]
        input_ids = torch.argmax(logits, dim=-1)
        return input_ids


    def _valid_stage(self, stage_id):
        return 0 <= stage_id < self.num_stages


    def _valid_micro_batch(self, micro_batch_id):
        return 0 <= micro_batch_id < self.micro_batches


    def _exec_sampling_schedule(self, examples, max_new_tokens=1e9):
        assert isinstance(examples, list) and len(examples) > 0
        # Reserve and reset buffers.
        self._reserve_pipe_buffers(2)
        self.fwd_outputs = []
        eos_token_ids = torch.tensor(list(self.eos_token_ids))
        for example in examples:
            example['done'] = torch.tensor([False]*example['input_ids'].size(0))
            example['num_new_tokens'] = 0
        num_batches_done = 0
        num_batches = len(examples)

        if self.is_first_stage():
            queue = deque()
            for i, example in enumerate(examples):
                queue.append((i, {
                    'input_ids': example['input_ids'],
                    'attention_mask': example['attention_mask'],
                    'micro_batch_id': torch.tensor(i),
                }))

        step_id = 0
        prev_micro_batch_id = -1
        while True:
            micro_batch_id = -1
            # Alternate send/recv buffers
            if _is_even(self.stage_id):
                recv_buf = step_id % 2
                send_buf = (step_id + 1) % 2
            else:
                recv_buf = (step_id + 1) % 2
                send_buf = step_id % 2

            # Send prev_micro_batch_id to next stage. Last stage wraps around and sends to first stage.
            micro_batch_id = self._exec_send_micro_batch_id(prev_micro_batch_id)
            batch_size = examples[micro_batch_id]['input_ids'].size(0)

            # If last stage did a forward pass, send the sampled input_ids to the first stage.
            if self.is_first_stage() and self._valid_micro_batch(micro_batch_id):
                if self.num_stages > 1:
                    input_ids = torch.tensor([[-1]*batch_size])
                    p2p.recv(input_ids, self.num_stages-1)
                assert input_ids.size(-1) == 1
                input_ids = input_ids.to('cpu')
                prev_done = examples[micro_batch_id]['done']
                # Determine which items in the batch are done generating.
                done = prev_done | (input_ids == eos_token_ids).any(-1)
                examples[micro_batch_id]['done'] = done
                batch_done = done.all().item()
                # Output pad token and 0 attention mask for items in the batch that are already done.
                prev_done_reshaped = prev_done.unsqueeze(-1)
                input_ids = torch.where(prev_done_reshaped, self.tokenizer.pad_token_id, input_ids)
                attention_mask_extension = torch.where(prev_done_reshaped, 0, 1)
                input_ids = torch.cat([examples[micro_batch_id]['input_ids'], input_ids], dim=-1)
                examples[micro_batch_id]['input_ids'] = input_ids
                attention_mask = torch.cat([examples[micro_batch_id]['attention_mask'], attention_mask_extension], dim=-1)
                examples[micro_batch_id]['attention_mask'] = attention_mask
                examples[micro_batch_id]['num_new_tokens'] += 1
                if examples[micro_batch_id]['num_new_tokens'] >= max_new_tokens:
                    break
                if batch_done:
                    num_batches_done += 1
                else:
                    # Model needs full attention mask, but only most recent sampled input_id.
                    queue.append((micro_batch_id, {'input_ids': input_ids[..., -1:], 'attention_mask': attention_mask, 'micro_batch_id': torch.tensor(micro_batch_id)}))
            if self.is_last_stage() and self._valid_micro_batch(prev_micro_batch_id) and self.num_stages > 1:
                p2p.send(input_ids, 0)

            if self.is_first_stage():
                if len(queue) > 0:
                    micro_batch_id, inputs = queue.popleft()
                    self._exec_load_micro_batch_for_sampling(recv_buf, inputs)

            if _is_even(self.stage_id):
                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(prev_micro_batch_id):
                        self._exec_send_activations(send_buf)
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        self._exec_recv_activations(recv_buf)
            else:
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        self._exec_recv_activations(recv_buf)
                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(prev_micro_batch_id):
                        self._exec_send_activations(send_buf)

            if self._valid_micro_batch(micro_batch_id):
                self._exec_sampling_forward_pass(recv_buf)
                if self.is_last_stage():
                    input_ids = self._sample_from_logits(recv_buf)

            prev_micro_batch_id = micro_batch_id
            step_id += 1
            if num_batches_done == num_batches:
                break

        for example in examples:
            del example['done']
            del example['num_new_tokens']
        return examples


    # make our forward pass method apply
    PipelineEngine._INSTRUCTION_MAP[schedule.ForwardPass] = _exec_forward_pass
    PipelineEngine._INSTRUCTION_MAP[LoadMicroBatchMultipleBuffers] = _exec_load_micro_batch_multiple_buffers
    PipelineEngine._INSTRUCTION_MAP[ReferenceLogitsForwardPass] = _exec_reference_logits_forward_pass


class ColumnMajorParallelTopology(ProcessTopology):
    """
    A topology specialisation for hybrid data+pipeline parallelism optimized for LoRA training:
    - Sends high-volume "per token" hidden states over PCIe/NVLink.
    - Sends lower-volume "per step" LoRA gradient reductions over Ethernet/InfiniBand.
    """
    def __init__(self, num_pp, num_dp):
        # Swap the axes and dims to change the rank mapping
        super().__init__(axes=['data', 'pipe'], dims=[num_dp, num_pp])


class CustomPipelineModule(PipelineModule):
    def __init__(self, layers, use_column_major_topology, model=None, **kwargs):
        # Assign to list to avoid registering the nn.Module
        self._model = [model]
        # Hybrid LoRA data+pipeline parallelism may want to use "column-major" layout
        if use_column_major_topology:
            world_size = dist.get_world_size()
            num_stages = kwargs.get('num_stages')
            if num_stages > 1 and world_size > 1:
                assert world_size % num_stages == 0, f"world_size ({world_size}) must be divisible by num_stages ({num_stages})"
                num_dp = world_size // num_stages
                kwargs['topology'] = ColumnMajorParallelTopology(num_pp=num_stages, num_dp=num_dp)
        super().__init__(layers, **kwargs)

    @property
    def model(self):
        return self._model[0]

    def set_dpo_reference_mode(self, dpo_reference_mode):
        self.model.set_dpo_reference_mode(dpo_reference_mode)

    def set_sampling_mode(self, sampling_mode):
        self.model.set_sampling_mode(sampling_mode)

    def _partition_layers(self, method='uniform'):
        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe

        if self.global_rank == 0:
            print(f'Partitioning pipeline stages with method {method}')

        method = method.lower()

        estimated_sizes = None
        # Each stage gets a simple uniform number of layers.
        if method == 'uniform':
            num_layers = len(self._layer_specs)
            self.parts = ds_utils.partition_uniform(num_items=num_layers, num_parts=num_stages)
        elif method == 'parameters':
            param_counts = self._count_layer_params()
            self.parts = ds_utils.partition_balanced(weights=param_counts, num_parts=num_stages)
        elif method.startswith('type:'):
            layertype = method.split(':')[1]
            binary_weights = [0] * len(self._layer_specs)
            for idx in self._find_layer_type(layertype):
                binary_weights[idx] = 1
            self.parts = ds_utils.partition_balanced(weights=binary_weights, num_parts=num_stages)
        elif method == 'profile':
            raise NotImplementedError(f'Partitioning method {method} not implemented.')
        elif method == 'estimated_size':
            estimated_sizes = [getattr(l, 'estimated_size', 0) for l in self._layer_specs]
            self.parts = ds_utils.partition_balanced(weights=estimated_sizes, num_parts=num_stages)
        else:
            raise NotImplementedError(f'Partitioning method {method} not implemented.')

        # Print some information on the partitioning.
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f'stage={stage} layers={stop - start}')
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, LayerSpec):
                        name = layer.typename.__name__
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    logstr = f'    {idx+start:2d}: {name}'
                    if estimated_sizes:
                        es = estimated_sizes[idx+start]
                        logstr += f', estimated size: {es}'
                    print(logstr)
            if self.loss_fn:
                try:
                    print(f'  loss: {self.loss_fn.__name__}')
                except AttributeError:
                    print(f'  loss: {self.loss_fn.__class__.__name__}')
        deepspeed.comm.barrier()

        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])


class DPOTrainSchedule(PipeSchedule):
    """Train schedule for DPO. Does an extra forward pass for the reference logits."""

    def steps(self):
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        forward_step_id = 0
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            # Alternate send/recv buffers for reference logits forward.
            num_normal_pipe_buffers = self.num_pipe_buffers() - 2
            if _is_even(self.stage_id):
                recv_buf = step_id % 2 + num_normal_pipe_buffers
                send_buf = (step_id + 1) % 2 + num_normal_pipe_buffers
            else:
                recv_buf = (step_id + 1) % 2 + num_normal_pipe_buffers
                send_buf = step_id % 2 + num_normal_pipe_buffers

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.prev_stage):
                    cmds.append(SendGrad(prev_buffer))
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(self.prev_stage):
                    cmds.append(RecvActivation(curr_buffer))

                # Activations for reference logits.
                if _is_even(self.stage_id):
                    if self._valid_stage(self.next_stage):
                        if self._valid_micro_batch(micro_batch_id - 1):
                            cmds.append(SendActivation(send_buf))
                    if self._valid_stage(self.prev_stage):
                        if self._valid_micro_batch(micro_batch_id):
                            cmds.append(RecvActivation(recv_buf))
                else:
                    if self._valid_stage(self.prev_stage):
                        if self._valid_micro_batch(micro_batch_id):
                            cmds.append(RecvActivation(recv_buf))
                    if self._valid_stage(self.next_stage):
                        if self._valid_micro_batch(micro_batch_id - 1):
                            cmds.append(SendActivation(send_buf))
            else:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(self.next_stage):
                    cmds.append(RecvGrad(curr_buffer))
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.next_stage):
                    cmds.append(SendActivation(prev_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    # Load for normal forward and reference logits forward.
                    cmds.append(LoadMicroBatchMultipleBuffers(curr_buffer, recv_buf))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    # Reference logits forward.
                    cmds.append(ReferenceLogitsForwardPass(recv_buf))
                    cmds.append(ForwardPass(curr_buffer))
                    forward_step_id += 1
                else:
                    cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

    def num_pipe_buffers(self):
        buffers = min(self.stages - self.stage_id, self.micro_batches)
        # +2 buffers for reference logit forward pass.
        return max(2, buffers) + 2

    def _step_to_micro_batch(self, step_id):
        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False

        else:
            assert False

        return micro_batch_id, is_forward

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id

    # Override to account for the extra 2 buffers used for reference logit forward pass.
    def _buffer_idx(self, micro_batch_id):
        assert self._valid_micro_batch(micro_batch_id)
        return micro_batch_id % (self.num_pipe_buffers() - 2)


class DPOInferenceSchedule(PipeSchedule):
    def steps(self):
        total_steps = self.micro_batches + self.stages - 1
        for step_id in range(total_steps):
            cmds = []
            micro_batch_id = step_id - self.stage_id

            # Alternate send/recv buffers
            if _is_even(self.stage_id):
                recv_buf = step_id % 2
                send_buf = (step_id + 1) % 2
            else:
                recv_buf = (step_id + 1) % 2
                send_buf = step_id % 2

            ref_recv_buf = recv_buf + 2
            ref_send_buf = send_buf + 2

            if self.is_first_stage or self.is_last_stage:
                if self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatchMultipleBuffers(recv_buf, ref_recv_buf))

            if _is_even(self.stage_id):
                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id - 1):
                        cmds.append(SendActivation(ref_send_buf))
                        cmds.append(SendActivation(send_buf))
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        cmds.append(RecvActivation(ref_recv_buf))
                        cmds.append(RecvActivation(recv_buf))
            else:
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        cmds.append(RecvActivation(ref_recv_buf))
                        cmds.append(RecvActivation(recv_buf))

                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id - 1):
                        cmds.append(SendActivation(ref_send_buf))
                        cmds.append(SendActivation(send_buf))

            if self._valid_micro_batch(micro_batch_id):
                cmds.append(ReferenceLogitsForwardPass(ref_recv_buf))
                cmds.append(ForwardPass(recv_buf))

            yield cmds

    def num_pipe_buffers(self):
        return 4
