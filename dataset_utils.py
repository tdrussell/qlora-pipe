import os.path
import json
import sys
import glob
import random

sys.path.insert(0, os.path.abspath('axolotl/src'))

import torch
from datasets import Dataset
from axolotl.utils.dict import DictDefault
from axolotl.utils.data import prepare_dataset
import jsonlines
from tqdm import tqdm

from utils import *


# dataset: list of dict with 'text' key
def yield_tokenized_sequences_from_dataset(tokenizer, dataset, sequence_len):
    need = sequence_len
    example_tokens = []
    for item in tqdm(dataset):
        tokens = tokenizer.encode(item['text'])
        assert tokens[-1] != tokenizer.eos_token_id, tokens[-1]
        tokens.append(tokenizer.eos_token_id)
        while len(tokens) > 0:
            taken = tokens[:need]
            tokens = tokens[need:]
            need -= len(taken)
            example_tokens.extend(taken)
            if len(example_tokens) >= sequence_len:
                assert len(example_tokens) == sequence_len
                yield example_tokens
                need = sequence_len
                example_tokens = []
    # yield anything remaining
    # TODO: disabled until I get training working with variable length sequences
    # if len(example_tokens) > 0:
    #     yield example_tokens


def get_examples_from_dataset(tokenizer, dataset, sequence_len):
    result = []
    for tokens in yield_tokenized_sequences_from_dataset(tokenizer, dataset, sequence_len):
        input_ids = torch.tensor(tokens)
        result.append({'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids), 'labels': input_ids})
    return result


def load_dataset_into_dict(path):
    if path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    if path.endswith('.jsonl'):
        with jsonlines.open(path) as reader:
            return list(reader)
    elif path.endswith('.txt'):
        with open(path) as f:
            text = f.read()
        return [{'text': text}]
    else:
        raise NotImplementedError()


def load_dataset_pattern_into_dict(pattern, subsample=None):
    result = []
    paths = glob.glob(pattern)
    if subsample is not None:
        k = int(len(paths) * subsample)
        print(f'Subsampling dataset to {k} entries')
        paths = random.sample(paths, k)
    for path in paths:
        result.extend(load_dataset_into_dict(path))
    return result


def load_raw_dataset(dataset_path, tokenizer, sequence_len, eval_size, cache_dir=None, ignore_cache=False, subsample=None):
    if cache_dir is not None:
        train_path = os.path.join(cache_dir, 'train')
        eval_path = os.path.join(cache_dir, 'eval')
    train_data = None
    eval_data = None
    try:
        if cache_dir is not None and not ignore_cache:
            train_data = Dataset.load_from_disk(train_path)
            eval_data = Dataset.load_from_disk(eval_path)
    except:
        pass
    if train_data is None and eval_data is None:
        dataset = load_dataset_pattern_into_dict(dataset_path, subsample=subsample)

        train_data = Dataset.from_list(get_examples_from_dataset(tokenizer, dataset, sequence_len))
        if eval_size > 0:
            split_datasets = train_data.train_test_split(test_size=eval_size, shuffle=True, seed=42)
            train_data = split_datasets['train']
            eval_data = split_datasets['test']
            if cache_dir is not None:
                train_data.save_to_disk(train_path)
                eval_data.save_to_disk(eval_path)
        else:
            train_data = train_data.shuffle(seed=42)
            if cache_dir is not None:
                train_data.save_to_disk(train_path)

    print(f'train_data size: {len(train_data)}')
    if eval_data is not None:
        print(f'eval_data size: {len(eval_data)}')
    return train_data, eval_data


def load_axolotl_dataset(dataset_path, dataset_type, tokenizer, sequence_len, eval_size):
    cfg = {}
    cfg['datasets'] = [{'path': dataset_path, 'type': dataset_type}]
    cfg['val_set_size'] = eval_size
    cfg['sequence_len'] = sequence_len
    # these two don't matter, but they have to be set
    cfg['batch_size'] = 1
    cfg['num_epochs'] = 1
    cfg = DictDefault(cfg)
    train_data, eval_data, _ = prepare_dataset(cfg, tokenizer)
    return train_data, eval_data


def load_dataset(dataset_path, dataset_type, tokenizer, sequence_len, eval_size, cache_dir=None, ignore_cache=False, subsample=None):
    if dataset_type in ['textfile', 'doclist']:
        with zero_first(is_main_process()):
            train_data, eval_data = load_raw_dataset(dataset_path, tokenizer, sequence_len, eval_size, cache_dir=cache_dir, ignore_cache=ignore_cache, subsample=subsample)
        return train_data, eval_data
    else:
        return load_axolotl_dataset(dataset_path, dataset_type, tokenizer, sequence_len, eval_size)
