import argparse
import os
from datetime import datetime, timezone
import shutil
import glob
import time

import torch
from torch.utils.tensorboard import SummaryWriter
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import deepspeed
import accelerate
import toml
import bitsandbytes

from dataset_utils import load_dataset
from llama_pipe import LlamaForCausalLMPipe
import dataloader
from utils import *
import engine

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--ignore_cache', action='store_true')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

def get_weight_norms(model):
    norms = []
    for w in (p.detach() for p in model.parameters() if p.requires_grad):
        norms.append(torch.linalg.matrix_norm(w, keepdim=True))
    return torch.cat(norms)

def print_model_info(model):
    if not is_main_process():
        return
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(pname)
            print(p.dtype)
            print(p.device)
            print(p.requires_grad)
            print()

def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]

def evaluate(model_engine, eval_dataloader, tb_writer, step):
    if is_main_process():
        print('Running eval')
    iterator = iter(eval_dataloader)
    total_loss = 0.
    count = 0.
    start = time.time()
    while True:
        loss = model_engine.eval_batch(iterator)
        if eval_dataloader.epoch == 2:
            break
        total_loss += loss.mean().item()
        count += 1
    duration = time.time() - start
    eval_dataloader.reset()
    eval_loss = total_loss / count
    if is_main_process():
        tb_writer.add_scalar('eval/loss', eval_loss, step)
        tb_writer.add_scalar('eval/eval_time_sec', duration, step)

# TODO: this is pretty hacky. Is there a way to get the state_dict from the lora model directly,
# but still know which layers the given pipeline parallel stage actually trained?
def save_lora(model_engine, pipeline_model, lora_config, save_dir):
    dp_id = model_engine.grid.get_data_parallel_rank()
    stage_id = model_engine.grid.get_pipe_parallel_rank()
    tmp_dir = os.path.join(save_dir, 'tmp')
    if dp_id == 0 and stage_id == 0:
        os.makedirs(tmp_dir, exist_ok=False)
    deepspeed.comm.barrier()
    if dp_id == 0:
        partial_state_dict = {p.original_name.replace('.default', ''): p for p in pipeline_model.parameters() if p.requires_grad}
        torch.save(partial_state_dict, os.path.join(tmp_dir, f'state_dict_{stage_id}.bin'))
    deepspeed.comm.barrier()
    if dp_id == 0 and stage_id == 0:
        state_dict = {}
        for path in glob.glob(os.path.join(tmp_dir, '*.bin')):
            state_dict.update(torch.load(path, map_location='cpu'))
        torch.save(state_dict, os.path.join(save_dir, 'adapter_model.bin'))
        lora_config.save_pretrained(save_dir)
        shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    # TODO: if resuming from checkpoint, probably should read all config files from checkpoint dir
    # rather than assume they are unchanged on the command line
    with open(args.config) as f:
        config = toml.load(f)

    if config['flash_attention']:
        from llama_attn_hijack_flash import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn(packed=False)
    elif config['xformers_attention']:
        from llama_attn_hijack_xformers import hijack_llama_attention
        hijack_llama_attention()
    elif config['sdp_attention']:
        from llama_attn_hijack_sdp import hijack_llama_sdp_attention
        hijack_llama_sdp_attention()

    deepspeed.init_distributed()

    # if this is a new run, create a new dir for it
    if not config['resume_from_checkpoint'] and is_main_process():
        run_dir = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.config, run_dir)
        shutil.copy(args.deepspeed_config, run_dir)
    # wait for all processes then get the most recent dir (may have just been created)
    deepspeed.comm.barrier()
    run_dir = get_most_recent_run_dir(config['output_dir'])

    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model'], local_files_only=True, use_fast=False, legacy=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'

    train_data, eval_data = load_dataset(config['dataset_path'], config['dataset_type'], tokenizer, config['sequence_len'], config['eval_size'], ignore_cache=args.ignore_cache and is_main_process())

    # for testing
    # train_data = train_data.select(list(range(20)))
    # eval_data = eval_data.select(list(range(20)))

    if config['torch_dtype'] == 'float16':
        torch_dtype = torch.float16
    elif config['torch_dtype'] == 'bfloat16':
        torch_dtype = torch.bfloat16
    else:
        raise NotImplementedError()
    
    # Ugly hack so we can move quantized models from GPU to CPU, and back to GPU again without triggering quantization a second time.
    bnb_cuda_old = bitsandbytes.nn.modules.Params4bit.cuda
    def bnb_cuda_hijack(self, device):
        if getattr(self, 'already_quantized', False):
            self.data = self.data.to(device)
            s = self.quant_state
            if s is not None:
                s[0] = s[0].to(device)
                if self.compress_statistics:
                    s[-3][0] = s[-3][0].to(device) # offset
                    s[-3][1][0] = s[-3][1][0].to(device) # nested quantization state statitics
                    s[-3][1][1] = s[-3][1][1].to(device) # nested quantization codebook
            return self
        self.already_quantized = True
        return bnb_cuda_old(self, device)
    bitsandbytes.nn.modules.Params4bit.cuda = bnb_cuda_hijack
    
    if os.path.exists(os.path.join(config['model'], 'quantize_config.json')):
        # TODO: GPTQ isn't going to work with all the new changes I made (e.g. offload_mlp_to_cpu)
        raise NotImplementedError()
        # if torch_dtype == torch.bfloat16:
        #     # TODO: fix bfloat16 with GPTQ
        #     if is_main_process() and torch_dtype != torch.float16:
        #         print('WARNING: forcing float16 because of GPTQ')
        #     torch_dtype = torch.float16
        # model_params = {
        #     'device_map': 'cpu',
        #     'quantization_config': transformers.GPTQConfig(bits=4, disable_exllama=True),
        #     'torch_dtype': torch_dtype,
        # }
        # model = LlamaForCausalLMPipe.from_pretrained(config['model'], local_files_only=True, **model_params)
    else:
        quantization_config_params = {
            'load_in_4bit': True,
            'bnb_4bit_compute_dtype': torch_dtype,
            'bnb_4bit_quant_type': 'nf4',
            'bnb_4bit_use_double_quant': config['use_double_quant'],
        }
        model_params = {
            'device_map': 'auto',
            'load_in_4bit': True,
            'quantization_config': transformers.BitsAndBytesConfig(**quantization_config_params),
            'torch_dtype': torch_dtype,
            'low_cpu_mem_usage': True,
        }
        # This makes the processes load the model one at a time. The GPUs are used to load and quantize,
        # so we would OOM without this.
        for i in range(int(os.environ['LOCAL_SIZE'])):
            if i == int(os.environ['LOCAL_RANK']):
                model = LlamaForCausalLMPipe.from_pretrained(config['model'], local_files_only=True, **model_params)
                for module in model.children():
                    module.to('cpu')
                torch.cuda.empty_cache()
            deepspeed.comm.barrier()

    #print_model_info(model)

    # Only need this if we're computing the loss outside the model.
    #vocab_size = model.config.vocab_size

    # There are hooks that try to automagically move intermediate tensors across devices.
    # Remove them so we can move the model (or individual layers) between devices correctly.
    accelerate.hooks.remove_hook_from_module(model, recurse=True)

    prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # LlamaRMSNorm layers are in fp32 after kbit_training, so we need to
    # convert them back to fp16/bf16 for flash-attn compatibility.
    for name, module in model.named_modules():
        if "norm" in name:
            module.to(torch_dtype)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                module.to(torch_dtype)

    lora_config = LoraConfig(
        r=config['lora_rank'],
        lora_alpha=config['lora_alpha'],
        target_modules=config['target_modules'],
        lora_dropout=config['lora_dropout'],
        bias='none',
        task_type='CAUSAL_LM',
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.config.use_cache = False
    for name, p in lora_model.named_parameters():
        p.original_name = name

    if config['activation_checkpointing']:
        pipeline_model = deepspeed.pipe.PipelineModule(
            layers=model.to_layers(),
            num_stages=config['pipeline_stages'],
            activation_checkpoint_interval=1,
            checkpointable_layers=['LlamaDecoderLayerPipe'],
            activation_checkpoint_func=deepspeed.checkpointing.checkpoint,
        )
    else:
        pipeline_model = deepspeed.pipe.PipelineModule(layers=model.to_layers(), num_stages=config['pipeline_stages'])

    parameters_to_train = [p for p in lora_model.parameters() if p.requires_grad]
    
    model_engine, optimizer = engine.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=parameters_to_train,
    )

    train_dataloader = dataloader.PipelineDataLoader(
        train_data,
        tokenizer,
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.grid.get_data_parallel_rank(),
        group_by_length=False if 'group_by_length' not in config else config['group_by_length']
    )
    model_engine.set_dataloader(train_dataloader)

    if 'lr_scheduler' not in config or config['lr_scheduler'] == 'none':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif config['lr_scheduler'] == 'cosine':
        steps_per_epoch = len(train_data) // model_engine.train_batch_size()
        total_steps = steps_per_epoch * config['epochs']
        total_steps -= config['warmup_steps'] if 'warmup_steps' in config else 0
        # Normally, you would pass the lr_scheduler to deepspeed.initialize(). But we need the
        # global batch_size in order to make the lr_scheduler.
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    else:
        raise NotImplementedError()
    
    if 'warmup_steps' in config:
        warmup_steps = config['warmup_steps']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_steps])

    model_engine.lr_scheduler = lr_scheduler

    # The MLP offloading only saves VRAM if we are also using activation checkpointing.
    if config['offload_mlp_to_cpu'] and config['activation_checkpointing']:
        model.offload_mlp_to_cpu()

    step = 1
    if config['resume_from_checkpoint']:
        load_path, client_state = model_engine.load_checkpoint(run_dir, load_module_strict=False, load_lr_scheduler_states=config['load_lr_scheduler_states'])
        assert load_path is not None
        train_dataloader.load_state_dict(client_state['custom_loader'])
        step = client_state['step'] + 1

    # Eval dataset doesn't need to repeat; we just use this to track "epoch" so we know when we're done iterating over it.
    eval_dataloader = dataloader.PipelineDataLoader(
        eval_data,
        tokenizer,
        model_engine.train_micro_batch_size_per_gpu(),
        1,
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.grid.get_data_parallel_rank(),
        shuffle=False,
        group_by_length=False if 'group_by_length' not in config else config['group_by_length']
    )

    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None

    epoch = train_dataloader.epoch
    if config['eval_before_first_step'] and not config['resume_from_checkpoint']:
        evaluate(model_engine, eval_dataloader, tb_writer, 0)
    while True:
        loss = model_engine.train_batch()
        model_engine.reset_activation_shape()  # sequence length may change between macro batches

        if train_dataloader.epoch != epoch:
            epoch = train_dataloader.epoch
            if epoch > config['epochs']:
                break
            if is_main_process():
                print(f'Started new epoch: {epoch}')
                tb_writer.add_scalar('train/epoch', epoch, step)

        if is_main_process() and step % config['logging_steps'] == 0:
            tb_writer.add_scalar('train/loss', loss.mean().item(), step)
            tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
            # TODO: gather the weight norms across all stages in the pipelined model, not just the first.
            weight_norms = get_weight_norms(pipeline_model)
            tb_writer.add_histogram('train/weight_norms_hist', weight_norms, step)
            tb_writer.add_scalar('train/weight_norms_avg', weight_norms.mean(), step)

        if step % config['save_steps'] == 0:
            save_lora(model_engine, pipeline_model, lora_config, f'{run_dir}/lora-{step}')

        if step % config['eval_steps'] == 0:
            evaluate(model_engine, eval_dataloader, tb_writer, step)

        if step % config['checkpoint_steps'] == 0:
            model_engine.save_checkpoint(
                run_dir,
                client_state={
                    'step': step,
                    'custom_loader': train_dataloader.state_dict(),
                },
                save_latest=True,
                exclude_frozen_parameters=True
            )

        step += 1

    deepspeed.comm.barrier()
    save_lora(model_engine, pipeline_model, lora_config, f'{run_dir}/lora-final')