# Usage: python merge_lora.py input_path lora_path output_path
# Output path is created if it doesn't exist

import os
from pathlib import Path
import shutil

import argparse
import torch
import safetensors
import peft

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("input_path", type=str, help="The path to the input directory.")
parser.add_argument("lora_path", type=str, help="The path to the LoRA directory.")
parser.add_argument("output_path", type=str, help="The path to the output directory.")
parser.add_argument("--no-gpu", action="store_true", help="Use CPU for merging.")
args = parser.parse_args()

input_path, lora_path, output_path = Path(args.input_path), Path(args.lora_path), Path(args.output_path)
os.makedirs(output_path, exist_ok=True)

lora_config = peft.LoraConfig.from_json_file(lora_path / 'adapter_config.json')
scale = lora_config['lora_alpha'] / lora_config['r']

device = "cpu" if args.no_gpu else "cuda"

print('Loading LoRA model...')

# Check if we have adapter_model.bin or adapter_model.safetensors
if (lora_path / 'adapter_model.safetensors').exists():
    lora_state = safetensors.torch.load_file(lora_path / 'adapter_model.safetensors')
    if not args.no_gpu:
        # Move mapped entries to cuda
        for key, value in tqdm(lora_state.items()):
            lora_state[key] = value.to('cuda')
else:
    lora_state = torch.load(lora_path / 'adapter_model.bin', map_location=device)

def find_lora_weights(key):
    lora_A = None
    lora_B = None
    for lora_key, lora_weight in lora_state.items():
        if key.strip('.weight') in lora_key:
            if 'lora_A' in lora_key:
                lora_A = lora_weight
            elif 'lora_B' in lora_key:
                lora_B = lora_weight
            else:
                raise RuntimeError()
    assert not ((lora_A is None) ^ (lora_B is None))
    return lora_A, lora_B

shards = []
for shard in input_path.glob('model*.safetensors'):
    shards.append(shard)

print('Copying unmergable files to output')
for filepath in input_path.glob('*'):
    if filepath in shards:
        continue
    filepath = Path(filepath)
    if filepath.is_dir():
        continue
    print(f'copying {filepath.name} to output')
    shutil.copy(filepath, output_path)

print('Merging and copying state_dict to output')
for shard in (pbar := tqdm(shards)):
    tensors = {}
    with safetensors.safe_open(shard, framework='pt', device=device) as f:
        metadata = f.metadata()
        for key in f.keys():
            tensor = f.get_tensor(key)
            lora_A, lora_B = find_lora_weights(key)
            if lora_A is not None:
                pbar.set_description(f'found lora weights for {key}: {lora_A.size()}, {lora_B.size()}')
                old_type = tensor.dtype
                tensor = tensor.to(torch.float32)
                tensor += scale * lora_B.to(torch.float32) @ lora_A.to(torch.float32)
                tensor = tensor.to(old_type)
            tensors[key] = tensor
        safetensors.torch.save_file(tensors, output_path / shard.name, metadata=metadata)
