import argparse
import os.path
import json

import toml
import deepspeed
import transformers
import bitsandbytes

from train import load_pipeline_model_with_lora
import engine
from utils import DTYPE_MAP


PROMPT = '''<|start_header_id|>user<|end_header_id|>

Where is Popeye Village located?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


if __name__ == '__main__':
    with open(args.config) as f:
        config = toml.load(f)
    config['full_fine_tune'] = True

    ds_config = {
        'train_micro_batch_size_per_gpu': config.get('micro_batch_size_per_gpu', 1),
        'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
        'steps_per_print': config.get('steps_per_print', 1),
    }

    deepspeed.init_distributed()

    with open(os.path.join(config['model'], 'config.json')) as f:
        model_config = json.load(f)
        model_type = model_config.get('model_type', 'llama')

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config['model'],
        local_files_only=True,
        model_max_length=int(1e30),
        padding_side='left',

    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # TODO: make it work with dynamic_shape=False for better performance.
    pipeline_model, lora_model, lora_config = load_pipeline_model_with_lora(config, model_type, dynamic_shape=True)

    model_engine, _ = engine.initialize(
        args=args,
        model=pipeline_model,
        lora_model=lora_model,
        config=ds_config,
        tokenizer=tokenizer,
    )
    weight_dtype = DTYPE_MAP[config.get('lora_weight_dtype', config.get('model_weight_dtype', 'float32'))]
    model_engine.communication_data_type = weight_dtype

    for text in model_engine.sample_batch([PROMPT]):
        print(text)
        print('-'*80)
