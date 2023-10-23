# Very hacky script to convert pipeline parallel Deepspeed checkpoints into a saved lora model.
# I originally wrote this because I screwed up the lora model saving initially, and needed a
# way to turn the training checkpoints into saved lora models to test them.

from glob import glob
import os.path
import re

import torch

def convert_ds_checkpoint_to_lora(ds_checkpoint_dir, lora_output_dir):
    layer_checkpoint_files = glob(os.path.join(ds_checkpoint_dir, 'layer_*-model_states.pt'))
    combined_state_dict = {}
    for path in layer_checkpoint_files:
        match = re.fullmatch('layer_(.+)-model_states.pt', os.path.basename(path))
        layer_idx = int(match.group(1)) - 2
        state_dict = torch.load(path)
        for name, weight in state_dict.items():
            converted_name = name.replace('orig', f'base_model.model.model.layers.{layer_idx}').replace('.default', '')
            combined_state_dict[converted_name] = weight
    os.makedirs(lora_output_dir, exist_ok=True)
    torch.save(combined_state_dict, os.path.join(lora_output_dir, 'adapter_model.bin'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()

    convert_ds_checkpoint_to_lora(args.input, args.output)
