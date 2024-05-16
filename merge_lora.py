# Usage: python merge_lora.py input_path lora_path output_path
# Output path is created if it doesn't exist

import sys
import os
from pathlib import Path
import shutil

import torch
import safetensors
import peft

from tqdm import tqdm

input_path, lora_path, output_path = [Path(x) for x in sys.argv[1:]]
os.makedirs(output_path, exist_ok=True)

lora_config = peft.LoraConfig.from_json_file(lora_path / 'adapter_config.json')
scale = lora_config['lora_alpha'] / lora_config['r']

print('Loading LoRA model...')

# Check if we have adapter_model.bin or adapter_model.safetensors
if (lora_path / 'adapter_model.safetensors').exists():
    lora_state = safetensors.torch.load_file(lora_path / 'adapter_model.safetensors')
    # Move mapped entries to cuda
    for key, value in tqdm(lora_state.items()):
        lora_state[key] = value.to('cuda')
else:
    lora_state = torch.load(lora_path / 'adapter_model.bin', map_location='cuda')

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
    print(f'copying {Path(filepath).name} to output')
    shutil.copy(filepath, output_path)

print('Merging and copying state_dict to output')
for shard in (pbar := tqdm(shards)):
    tensors = {}
    with safetensors.safe_open(shard, framework='pt', device='cuda') as f:
        metadata = f.metadata()
        for key in f.keys():
            tensor = f.get_tensor(key)
            lora_A, lora_B = find_lora_weights(key)
            if lora_A is not None:
                pbar.set_description(f'found lora weights for {key}: {lora_A.size()}, {lora_B.size()}')
                delta = (lora_B @ lora_A) * scale
                delta = delta.to(tensor.dtype)
                tensor += delta
            tensors[key] = tensor
        safetensors.torch.save_file(tensors, output_path / shard.name, metadata=metadata)
