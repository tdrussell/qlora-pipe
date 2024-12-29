import glob
import json
import os
import shutil
import time
import sys
from pathlib import Path
import deepspeed
import torch
import transformers

from safetensors.torch import save_file
from utils import is_main_process, DTYPE_MAP


last_checkpoint_time = None
def need_to_checkpoint(config):
    global last_checkpoint_time
    checkpoint = False
    # rank 0 tracks if we need to checkpoint, broadcasts to everyone else
    if is_main_process():
        current_time = time.time()
        if last_checkpoint_time is None:
            last_checkpoint_time = current_time
        elif (current_time - last_checkpoint_time) / 60 > config['checkpoint_every_n_minutes']:
            checkpoint = True
            last_checkpoint_time = current_time
    result = [checkpoint]
    torch.distributed.broadcast_object_list(result, src=0)
    return result[0]


def convert_state_dict_dtype(state_dict, dtype):
    for key, v in state_dict.items():
        state_dict[key] = v.to(device='cpu', dtype=DTYPE_MAP[dtype])


class Saver:
    def __init__(self, model_engine, pipeline_model, train_dataloader, lora_config, save_root, args, config):
        self.model_engine = model_engine
        self.pipeline_model = pipeline_model
        self.train_dataloader = train_dataloader
        self.lora_config = lora_config
        self.save_root = save_root + '/' if save_root[-1] != '/' else save_root
        self.args = args
        self.config = config
        self.keep_states = config.get('keep_states', -1)
        self.chrono_states = {
            'step': [],
            'global_step': [],
        }

        # Load best loss from disk, if found, and if a best_loss model dir exists
        self.best_loss = None
        best_loss_path = os.path.join(self.save_root, 'best_loss.txt')
        if os.path.exists(best_loss_path) and os.path.isdir(os.path.join(self.save_root, 'best_loss')):
            with open(best_loss_path, 'r') as f:
                self.best_loss = float(f.read())
            print(f'Loaded best loss from disk: {self.best_loss}')


    # TODO: this is pretty hacky. Is there a way to get the state_dict from the lora model directly,
    # but still know which layers the given pipeline parallel stage actually trained?
    def save_lora(self, name):
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        stage_id = self.model_engine.grid.get_pipe_parallel_rank()
        save_dir = self.save_root + name
        tmp_dir = os.path.join(save_dir, 'tmp')
        if dp_id == 0 and stage_id == 0:
            os.makedirs(tmp_dir, exist_ok=False)
        deepspeed.comm.barrier()
        if dp_id == 0:
            partial_state_dict = {}
            for name, p in self.pipeline_model.named_parameters():
                if p.requires_grad:
                    if not hasattr(p, 'original_name'):
                        print(f'WARNING: parameter {name} requires_grad but does not have original_name. Not saving it.')
                        continue
                    partial_state_dict[p.original_name.replace('.default', '').replace('.modules_to_save', '')] = p.detach()
                    if 'save_dtype' in self.config:
                        convert_state_dict_dtype(partial_state_dict, self.config['save_dtype'])
            torch.save(partial_state_dict, os.path.join(tmp_dir, f'state_dict_{stage_id}.bin'))
        deepspeed.comm.barrier()
        if dp_id == 0 and stage_id == 0:
            state_dict = {}
            for path in glob.glob(os.path.join(tmp_dir, '*.bin')):
                state_dict.update(torch.load(path, map_location='cpu'))
            torch.save(state_dict, os.path.join(save_dir, 'adapter_model.bin'))
            self.lora_config.save_pretrained(save_dir)
            shutil.copy(self.args.config, save_dir)
            shutil.copy(self.args.deepspeed_config, save_dir)
            shutil.rmtree(tmp_dir)


    def save_full_model(self, name, max_shard_size='5GB'):
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        stage_id = self.model_engine.grid.get_pipe_parallel_rank()
        save_dir = self.save_root + name
        tmp_dir = os.path.join(save_dir, 'tmp')
        if dp_id == 0 and stage_id == 0:
            os.makedirs(tmp_dir, exist_ok=False)
        deepspeed.comm.barrier()
        if dp_id == 0:
            # With BF16_Optimizer, we get pickle errors unless we do p.detach(). I have no idea why.
            partial_state_dict = {p.original_name: p.detach() for p in self.pipeline_model.parameters()}
            if 'save_dtype' in self.config:
                convert_state_dict_dtype(partial_state_dict, self.config['save_dtype'])
            torch.save(partial_state_dict, os.path.join(tmp_dir, f'state_dict_{stage_id}.bin'))
        deepspeed.comm.barrier()
        if dp_id == 0 and stage_id == 0:
            state_dict = {}
            for path in glob.glob(os.path.join(tmp_dir, '*.bin')):
                state_dict.update(torch.load(path, map_location='cpu'))
            shards, index = transformers.modeling_utils.shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name='model.safetensors')
            for shard_file, shard in shards.items():
                save_file(shard, os.path.join(save_dir, shard_file), metadata={"format": "pt"})
            if index is not None:
                save_index_file = 'model.safetensors.index.json'
                save_index_file = os.path.join(save_dir, save_index_file)
                # Save the index as well
                with open(save_index_file, "w", encoding="utf-8") as f:
                    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                    f.write(content)
            shutil.copy(self.args.config, save_dir)
            shutil.copy(self.args.deepspeed_config, save_dir)
            additional_files_to_copy = [
                'added_tokens.json',
                'config.json',
                'generation_config.json',
                'special_tokens_map.json',
                'tokenizer.json',
                'tokenizer_config.json',
                'tokenizer.model',
            ]
            for path in glob.glob(os.path.join(self.config['model'], '*')):
                if os.path.basename(path) in additional_files_to_copy:
                    shutil.copy(path, save_dir)
            shutil.rmtree(tmp_dir)


    def will_save(self, type, name):
        if self.keep_states <= 0 or not is_main_process():
            return
        if type == 'step':
            self.chrono_states['step'].append(name)
            if len(self.chrono_states['step']) > self.keep_states:
                print(f"Deleting {self.chrono_states['step'][0]}")
                shutil.rmtree(os.path.join(self.save_root, self.chrono_states['step'].pop(0)))
        elif type == 'global_step':
            self.chrono_states['global_step'].append(name)
            if len(self.chrono_states['global_step']) > self.keep_states:
                print(f"Deleting {self.chrono_states['global_step'][0]}")
                shutil.rmtree(os.path.join(self.save_root, self.chrono_states['global_step'].pop(0)))
        else:
            raise ValueError(f'Unknown save type: {type}')


    def save_model(self, name):
        # ignore epoch saves for chrono_states
        if name.startswith("step"):
            self.will_save('step', name)
        self.save_full_model(name) if self.lora_config is None else self.save_lora(name)


    def save_checkpoint(self, step):
        self.will_save('global_step', f'global_step{step}')
        self.model_engine.save_checkpoint(
            self.save_root,
            client_state={
                'step': step,
                'custom_loader': self.train_dataloader.state_dict(),
            },
            save_latest=True,
            exclude_frozen_parameters=True
        )


    def process_epoch(self, epoch, step):
        if self.train_dataloader.epoch != epoch:
            self.save_checkpoint(step)
            self.save_model(f'epoch{epoch}')
            epoch = self.train_dataloader.epoch
            if epoch > self.config['epochs']:
                return None
            if is_main_process():
                print(f'Started new epoch: {epoch}')
        return epoch


    def process_step(self, step):
        # Look at some simple "signal files" the user can write to save and optionally quit manually
        should_manually_save = False
        should_manually_quit = False
        save_signal_file = Path(self.save_root) / 'save'
        save_quit_signal_file = Path(self.save_root) / 'save_quit'
        if save_signal_file.exists() and save_signal_file.is_file():
            should_manually_save = True
            deepspeed.comm.barrier()
            if is_main_process():
                os.remove(save_signal_file)
        elif save_quit_signal_file.exists() and save_quit_signal_file.is_file():
            should_manually_save = True
            should_manually_quit = True
            deepspeed.comm.barrier()
            if is_main_process():
                os.remove(save_quit_signal_file)

        if ('save_steps' in self.config and step % self.config['save_steps'] == 0) or should_manually_save:
            self.save_model(f'step{step}')

        pending_save_best_loss = os.path.exists(os.path.join(self.save_root, ".pending_save_best_loss"))
        if pending_save_best_loss:
            self.save_model('best_loss')
            if is_main_process():
                if self.old_best is not None:
                    print(f"New best evaluation loss: {self.best_loss:.4f} from {self.old_best:.4f} (Δ{self.old_best - self.best_loss:.5f} [{100 * (1 - self.best_loss / self.old_best):.2f}%])")
                else:
                    print(f"New best evaluation loss: {self.best_loss:.4f}")
                os.replace(os.path.join(self.save_root, '.pending_save_best_loss'), os.path.join(self.save_root, 'best_loss.txt'))

        if need_to_checkpoint(self.config) or should_manually_save:
            self.save_checkpoint(step)

        if should_manually_quit:
            print('Manually quitting')
            sys.exit()


    def append_eval_results(self, loss, save_best=True):
        if loss is not None:
            if self.best_loss is None:
                print(f"Evaluation loss: {loss:.4f}")
            elif loss >= self.best_loss:
                print(f"Evaluation loss: {loss:.4f} (best: {self.best_loss:.4f}, Δ: {self.best_loss - loss:.5f} [{100 * (1 - loss / self.best_loss):.2f}%])")
            if self.best_loss is None or loss < self.best_loss:
                self.old_best = self.best_loss
                self.best_loss = loss
                if save_best:
                    with open(os.path.join(self.save_root, ".pending_save_best_loss"), "w") as f:
                        f.write(str(self.best_loss))
        deepspeed.comm.barrier()
