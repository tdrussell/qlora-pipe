import os.path
import sys
import glob
import random

sys.path.insert(0, os.path.abspath('axolotl/src'))

import torch
import datasets
from axolotl.utils.dict import DictDefault
from axolotl.utils.data import prepare_dataset
from tqdm import tqdm
import yaml

from utils import *


def yield_sequences_from_token_batch(tokenizer, token_batch, sequence_len):
    need = sequence_len
    example_tokens = []
    for tokens in tqdm(token_batch):
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


def slice_into_chunks(x, sequence_len, overlap=0):
    result = []
    step = sequence_len - overlap
    for i in range(0, len(x), step):
        result.append(x[i:i+sequence_len])
    return result


def load_raw_dataset(dataset_path, tokenizer, sequence_len, eval_size, overlap=0, subsample=None):
    if dataset_path.endswith('.txt'):
        dataset = datasets.load_dataset('text', data_files=dataset_path, sample_by='document')['train']
    elif dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
        dataset = datasets.load_dataset('json', data_files=dataset_path)['train']
    else:
        raise NotImplementedError()

    if subsample:
        dataset = dataset.shuffle(seed=13).select(list(range(int(subsample*len(dataset)))))
    dataset = dataset.map(lambda x: {'input_ids': tokenizer.encode(x['text'], return_tensors='pt').view(-1)}, remove_columns=dataset.column_names)
    # TODO: maybe do it this way instead
    #dataset = dataset.map(lambda x: {'tokens': slice_into_chunks(x['tokens'][0], sequence_len, overlap=overlap)}, batched=True, batch_size=1)
    dataset = dataset.map(lambda x: {'input_ids': list(yield_sequences_from_token_batch(tokenizer, x['input_ids'], sequence_len))}, batched=True)
    dataset = dataset.map(lambda x: {'attention_mask': torch.ones_like(torch.tensor(x['input_ids'])), 'labels': x['input_ids']})
    if eval_size > 0:
        split_datasets = dataset.train_test_split(test_size=eval_size, shuffle=True, seed=42)
        train_data = split_datasets['train']
        eval_data = split_datasets['test']
    else:
        train_data = dataset.shuffle(seed=42)
        eval_data = None
    return train_data, eval_data


def load_axolotl_dataset(dataset_path, tokenizer, sequence_len, eval_size):
    with open(dataset_path, 'r') as f:
        cfg = yaml.safe_load(f.read())
    if 'val_set_size' not in cfg and eval_size:
        cfg['val_set_size'] = eval_size
    if 'sequence_len' not in cfg and sequence_len:
        cfg['sequence_len'] = sequence_len
    # these two don't matter, but they have to be set
    cfg['batch_size'] = 1
    cfg['num_epochs'] = 1
    cfg = DictDefault(cfg)
    train_data, eval_data, *_ = prepare_dataset(cfg, tokenizer)
    if is_main_process():
        print(f'train_data size: {len(train_data)}')
        if eval_data is not None:
            print(f'eval_data size: {len(eval_data)}')
    return train_data, eval_data


def load_dataset(dataset_path, dataset_type, tokenizer, sequence_len, eval_size, subsample=None):
    if dataset_type in ['textfile', 'doclist']:
        with zero_first(is_main_process()):
            train_data, eval_data = load_raw_dataset(dataset_path, tokenizer, sequence_len, eval_size, subsample=subsample)
        return train_data, eval_data
    elif dataset_type == 'axolotl':
        return load_axolotl_dataset(dataset_path, tokenizer, sequence_len, eval_size)
    else:
        raise NotImplementedError()


# for testing
if __name__ == '__main__':
    import transformers
    # from datasets import disable_caching
    # disable_caching()

    tokenizer = transformers.AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, use_fast=False, legacy=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    train_data1, eval_data1 = load_raw_dataset('/home/anon/data/test/txt/*.txt', tokenizer, 100, 0.5)
    train_data2, eval_data2 = load_raw_dataset('/home/anon/data/test/json/*.jsonl', tokenizer, 100, 0.5)
    print(len(train_data1))
    print(len(train_data2))
