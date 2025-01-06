# Convert a DPO dataset with prompt, chosen, rejected fields into chat format.
# Usage: python convert_dpo_dataset_to_chat_format.py hf_username/some_dataset path/to/output/directory
import os
import sys
from pathlib import Path

import datasets


dataset_path, converted_path = sys.argv[1:]

dataset = datasets.load_dataset(dataset_path)

def convert(x):
    prompt = x['prompt']
    chosen = [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': x['chosen']}
    ]
    rejected = [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': x['rejected']}
    ]
    return {'chosen': chosen, 'rejected': rejected}

num_proc = min(64, os.cpu_count())
dataset = dataset.map(convert, num_proc=num_proc)
for name, split in dataset.items():
    filepath = Path(converted_path) / f'{name}.json'
    split.to_json(filepath)
