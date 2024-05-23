import argparse
import os
from datetime import datetime, timezone
import shutil
import glob
import time
import itertools
from contextlib import contextmanager
import json
import gc

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import transformers
import peft
from peft import LoraConfig, get_peft_model
import deepspeed
from deepspeed.runtime.pipe.module import LayerSpec
import toml
import bitsandbytes
from fastchat.conversation import register_conv_template, Conversation, SeparatorStyle
from hqq.core import quantize as hqq_quantize

from dataset_utils import load_datasets
import dataloader
from saver import Saver
from utils import is_main_process, DTYPE_MAP
import engine
import llama_pipe
import mixtral_pipe
import unsloth_utils
import hqq_utils

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--debug_dataset', type=int, help='print out this many training examples and then quit')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


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


def set_config_defaults(config):
    config['full_fine_tune'] = config.get('full_fine_tune', False)
    config['load_in_4bit'] = config.get('load_in_4bit', False)


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


def write_metrics(tb_writer, prefix, metrics, step):
    tb_writer.add_scalar(f'{prefix}/optimized_loss', metrics[0].mean().item(), step)

    if len(metrics) >= 2:
        losses = metrics[1].view(-1)
        sorted_losses, sorted_losses_idx = torch.sort(losses)
        quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999], dtype=torch.float32).to(losses.device)
        quantiles_idx = [int(len(losses)*quantile) for quantile in quantiles]
        loss_quantiles = [sorted_losses[i] for i in quantiles_idx]
        for quantile, value in zip(quantiles, loss_quantiles):
            tb_writer.add_scalar(f'{prefix}/loss_quantile_{quantile:.3f}', value, step)
        tb_writer.add_scalar(f'{prefix}/loss', losses.mean().item(), step)
        tb_writer.add_histogram(f'{prefix}/log_loss_hist', torch.log(1e-10 + losses), step)

    if len(metrics) >= 3:
        entropy = metrics[2].view(-1)
        assert entropy.size() == losses.size()
        tb_writer.add_scalar(f'{prefix}/entropy', entropy.mean().item(), step)
        sorted_entropy = entropy[sorted_losses_idx]
        entropy_quantiles = []
        for i, j in itertools.zip_longest(quantiles_idx, quantiles_idx[1:]):
            entropy_quantiles.append(sorted_entropy[i:j].mean())
        for quantile, value in zip(quantiles, entropy_quantiles):
            tb_writer.add_scalar(f'{prefix}/entropy_quantile_{quantile:.3f}', value, step)

    if len(metrics) >= 4:
        tb_writer.add_scalar(f'{prefix}/top1_accuracy', metrics[3].mean().item(), step)
        tb_writer.add_scalar(f'{prefix}/top5_accuracy', metrics[4].mean().item(), step)
        tb_writer.add_scalar(f'{prefix}/top20_accuracy', metrics[5].mean().item(), step)

    if len(metrics) >= 7:
        tb_writer.add_scalar(f'{prefix}/load_balancing_loss', metrics[6].mean().item(), step)
    if len(metrics) >= 8:
        tb_writer.add_scalar(f'{prefix}/alternate_load_balancing_loss', metrics[7].mean().item(), step)


def evaluate_single(model_engine, name, eval_dataloader, tb_writer, step, eval_gradient_accumulation_steps):
    orig_micro_batches = model_engine.micro_batches
    model_engine.micro_batches = eval_gradient_accumulation_steps
    iterator = iter(eval_dataloader)
    all_metrics = None
    while True:
        metrics = model_engine.eval_batch(iterator)
        eval_dataloader.sync_epoch()
        if all_metrics is None:
            all_metrics = [[] for _ in range(len(metrics))]
        if eval_dataloader.epoch == 2:
            break
        for i, metric in enumerate(metrics):
            all_metrics[i].append(metric)

    eval_dataloader.reset()
    model_engine.micro_batches = orig_micro_batches
    eval_metrics = [torch.cat(metric_list) for metric_list in all_metrics]
    if is_main_process():
        write_metrics(tb_writer, f'eval/{name}', eval_metrics, step)


def evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    if is_main_process():
        print('Running eval')
    start = time.time()
    for name, eval_dataloader in eval_dataloaders.items():
        evaluate_single(model_engine, name, eval_dataloader, tb_writer, step, eval_gradient_accumulation_steps)
    duration = time.time() - start
    if is_main_process():
        tb_writer.add_scalar('eval/eval_time_sec', duration, step)


def apply_max_norm_regularization(model, config):
    # modifed from https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py
    A_keys = []
    B_keys = []
    norms = []
    keys_scaled = 0
    lora_scale = config['lora_alpha'] / config['lora_rank']

    state_dict = model.state_dict()
    for key in state_dict.keys():
        if 'lora_A' in key:
            A_keys.append(key)
            B_keys.append(key.replace('lora_A', 'lora_B'))

    for i in range(len(A_keys)):
        A = state_dict[A_keys[i]]
        B = state_dict[B_keys[i]]
        W = B @ A
        W *= lora_scale

        if 'scale_weight_norms' in config:
            max_norm = config['scale_weight_norms']
            norm = W.norm().clamp(min=max_norm / 2)
            desired = torch.clamp(norm, max=max_norm)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[A_keys[i]] *= sqrt_ratio
                state_dict[B_keys[i]] *= sqrt_ratio
        else:
            ratio = 1.0
        scalednorm = W.norm() * ratio
        norms.append(scalednorm.item())

    if len(norms) > 0:
        norms = torch.tensor(norms, dtype=torch.float32)
        avg_norm = sum(norms) / len(norms)
        max_norm = max(norms)
    else:
        avg_norm = 0
        max_norm = 0
    return keys_scaled, avg_norm, max_norm, norms


def parse_layers_to_transform(spec):
    parts = spec.split(',')
    result = []
    for part in parts:
        start, stop = part.split(':')
        result.extend(range(int(start), int(stop)+1))
    return result


@contextmanager
def one_at_a_time():
    for i in range(int(os.environ['LOCAL_SIZE'])):
        if i == int(os.environ['LOCAL_RANK']):
            yield
        deepspeed.comm.barrier()


def load_pipeline_model_with_lora(config, model_type):
    full_fine_tune = config['full_fine_tune']

    if config.get('quantization', None):
        assert not full_fine_tune
        no_quant_modules = ['lm_head']
        if model_type == 'mixtral':
            # the expert routing weights are tiny and probably important, don't quantize
            no_quant_modules.append('gate')
        if bnb_quant_config := config['quantization'].get('bnb', None):
            if bnb_compute_dtype := bnb_quant_config.get('bnb_4bit_compute_dtype', None):
                bnb_quant_config['bnb_4bit_compute_dtype'] = DTYPE_MAP[bnb_compute_dtype]
            if 'bnb_4bit_quant_type' not in bnb_quant_config:
                # Always want to default to nf4 if not specified.
                bnb_quant_config['bnb_4bit_quant_type'] = 'nf4'
            if llm_int8_skip_modules := bnb_quant_config.get('llm_int8_skip_modules', None):
                no_quant_modules.extend(llm_int8_skip_modules)
                no_quant_modules = list(set(no_quant_modules))
            bnb_quant_config['llm_int8_skip_modules'] = no_quant_modules
            quantization_config = transformers.BitsAndBytesConfig(**bnb_quant_config)
        elif hqq_quant_config := config['quantization'].get('hqq', None):
            quantization_config = hqq_utils.CustomHQQConfig(**hqq_quant_config)
            # Use ATEN backend if possible, else PYTORCH. PYTORCH_COMPILE was only a tiny bit faster, and requires triton nightly.
            hqq_quantize.HQQLinear.set_backend(hqq_quantize.HQQBackend.ATEN if quantization_config.use_aten() else hqq_quantize.HQQBackend.PYTORCH)
        else:
            raise NotImplementedError(f'Invalid quantization config')
        if is_main_process():
            print(f'Quantization config: {quantization_config}')
    else:
        quantization_config = None

    if model_type == 'llama' or model_type == 'mistral':
        model = llama_pipe.LlamaForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'mixtral':
        model = mixtral_pipe.MixtralForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'qwen2':
        model = llama_pipe.Qwen2ForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'cohere':
        model = llama_pipe.CohereForCausalLMPipe(config, quantization_config=quantization_config)
    elif model_type == 'phi3':
        model = llama_pipe.Phi3ForCausalLMPipe(config, quantization_config=quantization_config)
    else:
        raise NotImplementedError()

    # CAREFUL! The "primary" layers of the model have to have 'decoderlayer' in them for
    # activation checkpointing to automatically work correctly.
    layers = model.to_layer_specs()
    checkpointable_layers = set()
    for layer in layers:
        if isinstance(layer, LayerSpec) and 'decoderlayer' in layer.typename.__name__.lower():
            checkpointable_layers.add(layer.typename.__name__)
    checkpointable_layers = list(checkpointable_layers)

    partition_method = 'estimated_size'
    if config['activation_checkpointing']:
        # NOTE: must use a reentrant checkpointing function for MLP offloading to work.
        if config['activation_checkpointing'] == 'unsloth':
            checkpoint_func = unsloth_utils.unsloth_checkpoint
        else:
            checkpoint_func = deepspeed.checkpointing.checkpoint
        pipeline_model = engine.CustomPipelineModule(
            layers=layers,
            num_stages=config['pipeline_stages'],
            activation_checkpoint_interval=1,
            checkpointable_layers=checkpointable_layers,
            activation_checkpoint_func=checkpoint_func,
            partition_method=partition_method
        )
    else:
        pipeline_model = engine.CustomPipelineModule(
            layers=layers,
            num_stages=config['pipeline_stages'],
            partition_method=partition_method
        )

    target_modules = config['target_modules'] if 'target_modules' in config else 'all-linear'
    if full_fine_tune:
        lora_model = None
        lora_config = None
        for name, p in model.named_parameters():
            p.original_name = name
        if isinstance(target_modules, list):
            for name, p in pipeline_model.named_parameters():
                if not any(target in name for target in config['target_modules']):
                    p.requires_grad = False
                    print(f'not training {name} because it is not present in target_modules')
    else:
        layers_to_transform = parse_layers_to_transform(config['layers_to_transform']) if 'layers_to_transform' in config else None
        lora_config = LoraConfig(
            r=config['lora_rank'],
            lora_alpha=config['lora_alpha'],
            target_modules=target_modules,
            modules_to_save=config['modules_to_save'] if 'modules_to_save' in config else [],
            lora_dropout=config['lora_dropout'] if 'lora_dropout' in config else 0,
            layers_to_transform=layers_to_transform,
            bias='none',
            task_type='CAUSAL_LM',
            use_dora=config.get('use_dora', False)
        )


        lora_model = get_peft_model(model, lora_config)
        # If the underlying weights are floats, the lora weights have already been
        # cast to the same dtype, so we need to change the dtype here.
        for p in lora_model.parameters():
            if p.requires_grad:
                p.data = p.data.to(DTYPE_MAP[config.get('lora_weight_dtype', 'float32')])

        lora_model.model.config.use_cache = False
        for name, p in lora_model.named_parameters():
            p.original_name = name

    return pipeline_model, lora_model, lora_config


if __name__ == '__main__':
    register_conv_template(
        Conversation(
            name='llama3',
            system_template='<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>',
            roles=('<|start_header_id|>user<|end_header_id|>\n\n', '<|start_header_id|>assistant<|end_header_id|>\n\n'),
            sep_style=SeparatorStyle.NO_COLON_SINGLE,
            sep='<|eot_id|>',
            stop_token_ids=[128001, 128009],
        )
    )

    # TODO: if resuming from checkpoint, probably should read all config files from checkpoint dir
    # rather than assume they are unchanged on the command line
    with open(args.config) as f:
        config = toml.load(f)
    set_config_defaults(config)

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

    with open(os.path.join(config['model'], 'config.json')) as f:
        model_config = json.load(f)
        model_type = model_config.get('model_type', 'llama')

    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    # TODO: do we want to do this with cohere models? By default the EOS token is <|END_OF_TURN_TOKEN|>
    # if model_type == 'cohere':
    #     tokenizer.eos_token = '<EOS_TOKEN>'
    tokenizer.pad_token = tokenizer.eos_token

    train_data, eval_data_map = load_datasets(config, tokenizer)

    if args.debug_dataset:
        if is_main_process():
            for i, item in enumerate(iter(train_data)):
                print('input_ids:')
                print(item['input_ids'])
                print('decoded input_ids:')
                print(tokenizer.decode(item['input_ids']))
                print('attention_mask:')
                print(item['attention_mask'])
                print('labels:')
                print(item['labels'])
                print('-'*80)
                if i >= args.debug_dataset-1:
                    break
        quit()

    # for testing
    # train_data = train_data.select(list(range(100)))
    # eval_data = eval_data.select(list(range(50)))

    # Ugly hack so we can move quantized models from GPU to CPU, and back to GPU again without triggering quantization a second time.
    bnb_cuda_old = bitsandbytes.nn.modules.Params4bit.cuda
    def bnb_cuda_hijack(self, device):
        if getattr(self, 'already_quantized', False):
            self.data = self.data.to(device)
            self.quant_state.to(device)
            return self
        self.already_quantized = True
        return bnb_cuda_old(self, device)
    bitsandbytes.nn.modules.Params4bit.cuda = bnb_cuda_hijack

    pipeline_model, lora_model, lora_config = load_pipeline_model_with_lora(config, model_type)

    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    def get_optimizer(model_parameters):
        optim_config = config['optimizer']
        lr = optim_config['lr']
        optim_type = optim_config['type'].lower()
        if optim_type == 'adamw':
            return deepspeed.ops.adam.FusedAdam(
                model_parameters,
                lr=lr,
                betas=(optim_config.get('beta1', 0.9), optim_config.get('beta2', 0.999)),
                weight_decay=optim_config.get('weight_decay', 0.01)
            )
        elif optim_type == 'adamw8bit':
            return bitsandbytes.optim.AdamW8bit(
                model_parameters,
                lr=lr,
                betas=(optim_config.get('beta1', 0.9), optim_config.get('beta2', 0.999)),
                weight_decay=optim_config.get('weight_decay', 0.01)
            )
        else:
            raise NotImplementedError(optim_type)

    model_engine, optimizer = engine.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=parameters_to_train,
        optimizer=get_optimizer,
    )

    # TODO: I have recently realized that we are setting things to fp16/bf16, even though all the DS
    # config was not in fp16 / bf16 mode. DS being in fp16/bf16 changes things in many places, e.g.
    # it can give you a BF16_Optimizer wrapper that accumulates grads in fp32, the communication dtype
    # is different, etc. I need to really look through all the implications of this. This change is so
    # that we keep the normal optimizer, but the communication dtype is changed so that we don't
    # unnecessarily cast grads to fp32.
    weight_dtype = DTYPE_MAP[config.get('lora_weight_dtype')]
    model_engine.communication_data_type = weight_dtype

    # TODO: the main DeepSpeedEngine forces all parameters to the GPU, and also does things like
    # broadcast all parameters from data parallel rank 0 to all other ranks. Thus, MLP offloading
    # must come after engine.initialize(). If we want to avoid loading everything onto GPUs only
    # to offload the MLPs, we have to rewrite a lot of code to work around things.
    if config.get('offload_mlp_to_cpu', False):
        assert config['activation_checkpointing']  # MLP offloading only works with activation checkpointing
        for module in pipeline_model.modules():
            if hasattr(module, 'offload_mlp_to_cpu'):
                module.offload_mlp_to_cpu()
        torch.cuda.empty_cache()

    train_dataloader = dataloader.PipelineDataLoader(
        train_data,
        tokenizer,
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.grid.get_data_parallel_rank(),
        group_by_length=False if 'group_by_length' not in config else config['group_by_length'],
        batch_size_tokens=None if 'batch_size_tokens' not in config else config['batch_size_tokens'],
    )
    model_engine.set_dataloader(train_dataloader)
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
    model_engine.total_steps = steps_per_epoch * config['epochs']

    if is_main_process():
        # Warn if eval dataset is unusually large compared to the eval steps
        eval_data_length = sum([len(eval_data) for eval_data in eval_data_map.values()])
        train_data_length = len(train_data)
        evals_per_epoch = steps_per_epoch / config['eval_steps']
        relative_eval_time = evals_per_epoch * eval_data_length
        # train step very roughly 3 times slower due to backprop + usually activation checkpointing is enabled
        relative_train_time = train_data_length * 3
        # Expect <=15% of our time spent evaluating vs training
        fraction_evaling = relative_eval_time / (relative_eval_time + relative_train_time)
        print()
        print(f'eval_data_length: {eval_data_length}, eval_steps: {config["eval_steps"]}; evals per epoch: {evals_per_epoch}. '
              f'We will be spending approximately {fraction_evaling*100:.2f}% of our time evaluating.')
        if fraction_evaling > 0.15:
            print(f'WARNING: eval dataset is unusually large compared to eval_steps. We will spend a lot of time evaluating. Lowering eval_size and/or bumping eval_steps is recommended.')
        print()

    if 'lr_scheduler' not in config or config['lr_scheduler'] == 'constant' or config['lr_scheduler'] == 'none':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif config['lr_scheduler'] == 'cosine':
        total_steps = steps_per_epoch * config['epochs']
        total_steps -= config['warmup_steps'] if 'warmup_steps' in config else 0
        # Normally, you would pass the lr_scheduler to deepspeed.initialize(). But we need the
        # global batch_size in order to make the lr_scheduler.
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    else:
        raise NotImplementedError()

    load_optimizer_states = config.get('load_optimizer_states', True)
    # if resuming and not loading optimizer states, we can't use warmup or the LR never changes from the initial value (still don't know why)
    if 'warmup_steps' in config and config['warmup_steps'] > 0 and load_optimizer_states:
        warmup_steps = config['warmup_steps']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_steps])

    model_engine.lr_scheduler = lr_scheduler

    step = 1
    if config['resume_from_checkpoint']:
        load_path, client_state = model_engine.load_checkpoint(
            run_dir,
            load_module_strict=False,
            load_lr_scheduler_states='force_constant_lr' not in config,
            load_optimizer_states=load_optimizer_states
        )
        deepspeed.comm.barrier()  # just so the print below doesn't get swamped
        assert load_path is not None
        train_dataloader.load_state_dict(client_state['custom_loader'])
        step = client_state['step'] + 1
        del client_state
        # if we skip loading the optimizer states, we need to step the LR scheduler so we start at the right value
        if not load_optimizer_states:
            model_engine.lr_scheduler.step()

    if 'force_constant_lr' in config:
        model_engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = config['force_constant_lr']

    # this is a separate option, because if it's too high we might drop a significant fraction of the eval dataset
    eval_gradient_accumulation_steps = config['eval_gradient_accumulation_steps'] if 'eval_gradient_accumulation_steps' in config else 1
    # Eval dataset doesn't need to repeat; we just use this to track "epoch" so we know when we're done iterating over it.
    eval_dataloaders = {
        name: dataloader.PipelineDataLoader(
            eval_data,
            tokenizer,
            model_engine.train_micro_batch_size_per_gpu(),
            eval_gradient_accumulation_steps,
            model_engine.grid.get_data_parallel_world_size(),
            model_engine.grid.get_data_parallel_rank(),
            shuffle=False,
            group_by_length=False if 'group_by_length' not in config else config['group_by_length'],
            batch_size_tokens=None if 'batch_size_tokens' not in config else config['batch_size_tokens'],
        )
        for name, eval_data in eval_data_map.items()
    }

    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None

    epoch = train_dataloader.epoch
    if config['eval_before_first_step'] and not config['resume_from_checkpoint']:
        evaluate(model_engine, eval_dataloaders, tb_writer, 0, eval_gradient_accumulation_steps)

    saver = Saver(model_engine, pipeline_model, train_dataloader, lora_config, run_dir, args, config)

    while True:
        gc.collect()
        torch.cuda.empty_cache()
        metrics = model_engine.train_batch()
        train_dataloader.sync_epoch()
        if lora_config is not None:
            keys_scaled, avg_norm, max_norm, norms = apply_max_norm_regularization(pipeline_model, config)

        epoch = saver.process_epoch(epoch, step)
        if epoch is None:
            break

        if is_main_process() and step % config['logging_steps'] == 0:
            write_metrics(tb_writer, 'train', metrics, step)
            tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
            # TODO: gather the weight norms across all stages in the pipelined model, not just the first.
            if lora_config is not None and len(norms) > 0:
                tb_writer.add_scalar('train/weights_scaled', keys_scaled, step)
                tb_writer.add_scalar('train/avg_weight_norm', avg_norm, step)
                tb_writer.add_scalar('train/max_weight_norm', max_norm, step)
                tb_writer.add_histogram('train/weight_norm_hist', norms, step)
            tb_writer.add_scalar('train/epoch', step/steps_per_epoch, step)

        if step % config['eval_steps'] == 0:
            evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps)

        saver.process_step(step)

        step += 1

    if is_main_process():
        print('TRAINING COMPLETE!')
