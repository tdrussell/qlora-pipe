import math
import sys
import os.path
sys.path.insert(0, os.path.abspath('axolotl/src'))

import torch
from torch.utils.data import DataLoader
import transformers
import accelerate
from deepspeed import comm as dist
from tqdm import tqdm

from axolotl.utils.collators import DataCollatorForSeq2Seq
from utils import *

# A100 wants padding to multiple of 64, other cards are efficient with smaller, so just do 64
PAD_TO_MULTIPLE = 64


def split_batch(batch, pieces):
    example_tuple, labels = batch
    if is_main_process():
        print(f'before GAS splitting, batch size: {example_tuple[0].size(0)}, total tokens: {example_tuple[0].numel()}')
    split_size = example_tuple[0].size(0) // pieces
    split_examples = zip(*(torch.split(tensor, split_size) for tensor in example_tuple))
    return [(ex, None) for ex in split_examples]


# Merge lists a and b, such that for each contiguous piece in the result, the first half comes from
# a and the second half from b. Used for DPO. The splitting must match how split_batch() does it.
def combine_piecewise(a, b, pieces):
    assert len(a) == len(b)
    split_size = len(a) // pieces
    a_chunks = [a[i:i+split_size] for i in range(0, len(a), split_size)]
    b_chunks = [b[i:i+split_size] for i in range(0, len(b), split_size)]
    result = []
    for a_chunk, b_chunk in zip(a_chunks, b_chunks):
        result.extend(a_chunk)
        result.extend(b_chunk)
    return result


# A distributed batch sampler that supports grouping by length
class DistributedBatchSamper(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, num_replicas, rank, batch_size_multiplier=1, shuffle=True, group_by_length=False, seed=0, drop_last=False, batch_size_tokens=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_size_tokens = batch_size_tokens
        self.batch_size_multiplier = batch_size_multiplier
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.group_by_length = group_by_length
        self.seed = seed

        indices = list(enumerate(self.dataset['length']))
        if self.group_by_length:
            indices.sort(key=lambda t: t[1], reverse=True)
        elif self.shuffle:
            # deterministically shuffle based on seed
            g = torch.Generator()
            g.manual_seed(self.seed)
            shuffle_idx = torch.randperm(len(self.dataset), generator=g).tolist()
            indices = [indices[i] for i in shuffle_idx]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        if self.batch_size_tokens:
            global_batch_size_tokens = self.batch_size_tokens * self.num_replicas * self.batch_size_multiplier
            chunk_size = self.num_replicas * self.batch_size_multiplier
            global_batches = []
            current_batch = []
            current_size = 0
            batch_sequence_length = 0
            for i in range(0, len(indices), chunk_size):
                slice = indices[i:i+chunk_size]
                batch_sequence_length = max(batch_sequence_length, int(math.ceil(slice[0][1] / PAD_TO_MULTIPLE)) * PAD_TO_MULTIPLE)
                slice = [(idx, batch_sequence_length) for idx, _ in slice]
                slice_tokens = batch_sequence_length * len(slice)
                if len(current_batch) > 0 and current_size + slice_tokens > global_batch_size_tokens:
                    global_batches.append(current_batch)
                    current_batch = []
                    current_size = 0
                    batch_sequence_length = 0
                current_batch.extend(slice)
                current_size += slice_tokens
        else:
            global_batch_size = self.batch_size * self.num_replicas * self.batch_size_multiplier
            global_batches = [indices[i:i+global_batch_size] for i in range(0, len(indices), global_batch_size)]
        if self.drop_last:
            if self.batch_size_tokens:
                global_batches = global_batches[:-1]
            else:
                global_batches = [b for b in global_batches if len(b) == global_batch_size]
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed+1)
            shuffle_idx = torch.randperm(len(global_batches), generator=g)
            global_batches = [global_batches[i] for i in shuffle_idx]

        # make sure the largest batch comes first to OOM sooner rather than later
        largest_global_batch = 0
        max_tokens = 0
        for global_batch_idx, batch in enumerate(global_batches):
            total_batch_tokens = sum(t[1] for t in batch)
            if total_batch_tokens > max_tokens:
                max_tokens = total_batch_tokens
                largest_global_batch = global_batch_idx
        global_batches[0], global_batches[largest_global_batch] = global_batches[largest_global_batch], global_batches[0]

        batches_for_this_rank = [global_batch[self.rank:len(global_batch):self.num_replicas] for global_batch in global_batches]
        self.indices = [[i for i, _ in batch] for batch in batches_for_this_rank]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class PipelineDataLoader:
    def __init__(self, dataset, tokenizer, batch_size, gradient_accumulation_steps, data_parallel_world_size, data_parallel_rank, shuffle=True, group_by_length=False, pad_to_multiple_of=PAD_TO_MULTIPLE, drop_last=True, batch_size_tokens=None):
        assert data_parallel_rank < data_parallel_world_size
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.batch_size_tokens = batch_size_tokens
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.pad_to_multiple_of = pad_to_multiple_of
        self.data_sampler = DistributedBatchSamper(
            dataset=dataset,
            batch_size=self.batch_size,
            batch_size_tokens=self.batch_size_tokens,
            batch_size_multiplier=self.gradient_accumulation_steps,
            num_replicas=data_parallel_world_size,
            rank=data_parallel_rank,
            shuffle=shuffle,
            group_by_length=group_by_length,
            drop_last=drop_last
        )
        self.reset()

    def reset(self):
        self.epoch = 1
        self.num_batches_pulled = 0
        self._create_dataloader()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data_sampler) * self.gradient_accumulation_steps

    def __next__(self):
        try:
            batch = next(self.data)
        except StopIteration:
            self._create_dataloader()
            batch = next(self.data)
            self.epoch += 1
        return batch

    def _pull_batches_from_dataloader(self):
        for macro_batch in self.dataloader:
            self.num_batches_pulled += 1
            for batch in split_batch(macro_batch, self.gradient_accumulation_steps):
                yield batch

    def _create_dataloader(self):
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        def collate_fn(examples):
            rejected_examples = []
            for example in examples:
                del example['length']
                rejected_example = {}
                for key in list(example.keys()):
                    if 'rejected_' in key:
                        rejected_example[key.strip('rejected_')] = example.pop(key)
                if rejected_example:
                    rejected_examples.append(rejected_example)
            if rejected_examples:
                examples = combine_piecewise(examples, rejected_examples, self.gradient_accumulation_steps)
            batch = data_collator(examples)
            # input to pipeline is (input_ids, attention_mask, labels)
            # this needs to return (features, labels)
            # it is OK if labels is None (the model just returns the loss anyway)
            return ((batch['input_ids'], batch['attention_mask'], batch['labels']), None)
        self.dataloader = DataLoader(
            self.dataset,
            pin_memory=True,
            batch_sampler=self.data_sampler,
            collate_fn=collate_fn,
            #num_workers=self.num_local_io_workers,
        )
        self.data = self._pull_batches_from_dataloader()
        self.num_batches_pulled = 0

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'num_batches_pulled': self.num_batches_pulled,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.num_batches_pulled = state_dict['num_batches_pulled']
        self.dataloader = accelerate.skip_first_batches(self.dataloader, self.num_batches_pulled)
        self.data = self._pull_batches_from_dataloader()

    # Only the first and last stages in the pipeline pull from the dataloader. Parts of the code need
    # to know the epoch, so we synchronize the epoch so the processes that don't use the dataloader
    # know the current epoch.
    def sync_epoch(self):
        process_group = dist.get_world_group()
        result = [None] * dist.get_world_size(process_group)
        torch.distributed.all_gather_object(result, self.epoch, group=process_group)
        max_epoch = -1
        for epoch in result:
            max_epoch = max(epoch, max_epoch)
        self.epoch = max_epoch


# Simple wrapper for PipelineDataLoader that allows for manually advancing to the next
# macro batch. If advance() is not called, it will iterate over the same
# gradient_accumulation_steps microbatches next time. Used for DPO.
class PipelineDataLoaderWrapper:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        return iter(self.data)

    def advance(self):
        self.data = []
        for _ in range(self.dataloader.gradient_accumulation_steps):
            self.data.append(next(self.dataloader))


# for testing
if __name__ == '__main__':
    tokenizer = transformers.AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, use_fast=False, legacy=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'

    from datasets import Dataset
    data = []
    for i in range(1, 41):
        input_ids = torch.tensor([i]*i)
        data.append({'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids), 'labels': input_ids})
    dataset = Dataset.from_list(data)

    # dataloader = PipelineDataLoader(dataset, tokenizer, batch_size=2, gradient_accumulation_steps=2, data_parallel_world_size=1, data_parallel_rank=0, group_by_length=True, pad_to_multiple_of=None)
    # for batch in dataloader:
    #     if dataloader.epoch > 1:
    #         break
    #     print(batch)
    #     print()

    batch_size = 2
    gradient_accumulation_steps = 2
    data_parallel_world_size = 2
    data_parallel_rank = 0
    dataloader = PipelineDataLoader(
        dataset,
        tokenizer,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        data_parallel_world_size=data_parallel_world_size,
        data_parallel_rank=data_parallel_rank,
        shuffle=False,
        group_by_length=False,
        pad_to_multiple_of=None
    )
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])

    state_dict = dataloader.state_dict()
    dataloader = PipelineDataLoader(
        dataset,
        tokenizer,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        data_parallel_world_size=data_parallel_world_size,
        data_parallel_rank=data_parallel_rank,
        shuffle=False,
        group_by_length=False,
        pad_to_multiple_of=None
    )
    dataloader.load_state_dict(state_dict)
    print()
    print('-'*80)
    print()
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])
