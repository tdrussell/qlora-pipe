import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed import comm as dist
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.pipe.engine import PipelineEngine, TRAIN_BATCH_TIMER, PIPE_SEND_OUTPUT_TIMER, PIPE_SEND_GRAD_TIMER, PIPE_RECV_INPUT_TIMER, PIPE_RECV_GRAD_TIMER
from deepspeed.runtime.pipe import schedule


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
        self.total_loss = None
        self._compute_loss = True

        # Do the work
        self.timers(TRAIN_BATCH_TIMER).start()
        sched = schedule.TrainSchedule(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self._exec_schedule(sched)
        # Actual training loss is always the first item.
        self.agg_train_loss = self._aggregate_total_losses()[0]

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

        # TODO: should return precisely what loss returned and allow others to be queried?
        return self.agg_train_loss


    def eval_batch(self, data_iter):
        self.module.eval()
        self.total_loss = None
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
        agg_eval_loss = self._aggregate_total_losses()

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/eval_loss', agg_eval_loss[0].mean().item(), self.global_samples)]
            self.monitor.write_events(self.summary_events)

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        return agg_eval_loss[0]


    def _aggregate_total_losses(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if isinstance(self.total_loss, torch.Tensor):
            all_losses = [self.total_loss]
        else:
            all_losses = self.total_loss

        all_agg_losses = []
        for loss in all_losses:
            if self.is_last_stage():
                loss = self._scale_loss_by_gas(loss)

                # Average loss across all data-parallel groups
                agg_loss = loss.clone().detach()
                if self.is_data_parallel:
                    dist.all_reduce(agg_loss, group=self.mpu.get_data_parallel_group())
                    agg_loss /= self.dp_world_size

                assert self.global_rank in self.grid.pp_group
                if self.is_pipe_parallel:
                    dist.broadcast(tensor=agg_loss, src=self.global_rank, group=self.mpu.get_pipe_parallel_group())
                all_agg_losses.append(agg_loss)
            else:
                # Get loss from last stage
                src_rank = self.grid.stage_to_global(self.num_stages - 1)
                assert src_rank in self.grid.pp_group
                agg_loss = torch.Tensor([0.]).to(self.device)
                dist.broadcast(tensor=agg_loss, src=src_rank, group=self.grid.get_pipe_parallel_group())
                all_agg_losses.append(agg_loss)

        return all_agg_losses