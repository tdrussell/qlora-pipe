import math
import os.path
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath('axolotl/src'))

import accelerate
import torch
import transformers
from deepspeed import comm as dist
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from utils.utils import is_main_process


# A100 wants padding to multiple of 64, other cards are efficient with smaller, so just do 64
PAD_TO_MULTIPLE = 64


# Splits an example (feature dict) along the batch dimension into a list of examples.
def split_batch(example, pieces):
    input_ids = example['input_ids']
    if is_main_process():
        print(f'before GAS splitting, input_ids shape: {input_ids.shape}, total tokens: {input_ids.numel()}')
    input_batch_size = input_ids.size(0)
    split_size = input_batch_size // pieces
    examples = [{} for _ in range(pieces)]
    for key, tensor in example.items():
        assert tensor.size(0) == input_batch_size
        for i, tensor_slice in enumerate(torch.split(tensor, split_size)):
            examples[i][key] = tensor_slice
    return examples


# Merge lists of examples a and b, such that for each contiguous piece in the result, the first half comes from
# a and the second half from b. Used for DPO. The splitting must match how split_batch() does it.
def combine_piecewise(a, b, pieces):
    assert len(a) == len(b)
    split_size = len(a) // pieces
    a_chunks = [a[i : i + split_size] for i in range(0, len(a), split_size)]
    b_chunks = [b[i : i + split_size] for i in range(0, len(b), split_size)]
    result = []
    for a_chunk, b_chunk in zip(a_chunks, b_chunks):
        result.extend(a_chunk)
        result.extend(b_chunk)
    return result


# Flattens a list of examples with batch dimension into a list of examples with no batch dimension.
def flatten_examples(examples):
    result = []
    for example in examples:
        batch_size = example['input_ids'].size(0)
        new_examples = [{} for _ in range(batch_size)]
        for key, tensor in example.items():
            assert tensor.size(0) == batch_size
            for i, tensor_slice in enumerate(tensor):
                new_examples[i][key] = tensor_slice
        result.extend(new_examples)
    return result


def example_to_tuple(example):
    return (example['input_ids'], example['attention_mask'], example['labels']), None


def shuffle_list(l, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    shuffle_idx = torch.randperm(len(l), generator=g).tolist()
    new_l = [l[i] for i in shuffle_idx]
    return new_l


def batch_size_tokens_after_padding(batch):
    return max(math.ceil(pair[1] / PAD_TO_MULTIPLE) * PAD_TO_MULTIPLE for pair in batch) * len(batch)


# Supports arbitrary numbers of extra dimensions on the tensors. For example, csm-1b has an extra
# dimension for all the codebooks.
@dataclass
class DataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = None
    label_pad_token_id: int = -100

    def __call__(self, examples):
        batched_example = {}
        for feature in examples[0].keys():
            if feature == 'input_ids':
                pad_value = self.tokenizer.pad_token_id
            elif feature == 'labels':
                pad_value = self.label_pad_token_id
            else:
                pad_value = 0
            tensors = [example[feature] for example in examples]
            batched_example[feature] = self._pad(tensors, pad_value, self.tokenizer.padding_side)
        return batched_example

    def _pad(self, tensors, pad_value, padding_side):
        first_shape = tensors[0].shape[1:]
        assert all(x.shape[1:] == first_shape for x in tensors)
        shape = max(tensors, key=lambda x: x.shape[0]).shape
        max_length = shape[0]
        if self.pad_to_multiple_of is not None:
            max_length = math.ceil(max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of
        bs = len(tensors)
        batched_shape = (bs, max_length) + shape[1:]
        result = torch.full(batched_shape, pad_value)
        for i, tensor in enumerate(tensors):
            length = tensor.shape[0]
            if padding_side == 'right':
                result[i, :length, ...] = tensor
            else:
                result[i, -length:, ...] = tensor
        return result


# A distributed batch sampler that supports grouping by length
class DistributedBatchSamper(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas,
        rank,
        batch_size_multiplier=1,
        shuffle=True,
        group_by_length=False,
        seed=0,
        batch_size_tokens=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_size_tokens = batch_size_tokens
        self.batch_size_multiplier = batch_size_multiplier
        self.num_replicas = num_replicas
        self.rank = rank
        # every global batch must be evenly divisible by this amount
        self.chunk_size = self.num_replicas * self.batch_size_multiplier
        self.shuffle = shuffle
        self.group_by_length = group_by_length
        self.seed = seed

        # Make list of (index, size). Sort or shuffle as needed.
        indices = list(enumerate(self.dataset['length']))
        if self.group_by_length:
            indices.sort(key=lambda t: t[1])
        elif self.shuffle:
            indices = shuffle_list(indices, self.seed)

        # Group indices together into global batches.
        global_batches = []
        current_batch = []
        for i in range(0, len(indices), self.chunk_size):
            slice = indices[i : i + self.chunk_size]
            if len(slice) < self.chunk_size:
                # pad with random examples if slice is too small
                padding_size = self.chunk_size - len(slice)
                shuffled_indices = shuffle_list(indices, self.seed + 1)
                if padding_size < len(shuffled_indices):
                    slice += shuffled_indices[:padding_size]
                else:
                    slice += (shuffled_indices * math.ceil(padding_size / len(shuffled_indices)))[:padding_size]

            if self.should_emit_current_batch(current_batch, slice):
                global_batches.append(current_batch)
                current_batch = []
            current_batch.extend(slice)

        # Emit anything remaining
        if len(current_batch) > 0:
            global_batches.append(current_batch)

        if self.shuffle:
            global_batches = shuffle_list(global_batches, self.seed + 2)

        # make sure the largest batch comes first to OOM sooner rather than later
        largest_global_batch = 0
        max_tokens = 0
        for global_batch_idx, batch in enumerate(global_batches):
            total_batch_tokens = batch_size_tokens_after_padding(batch)
            if total_batch_tokens > max_tokens:
                max_tokens = total_batch_tokens
                largest_global_batch = global_batch_idx
        global_batches[0], global_batches[largest_global_batch] = (
            global_batches[largest_global_batch],
            global_batches[0],
        )

        batches_for_this_rank = [
            global_batch[self.rank : len(global_batch) : self.num_replicas] for global_batch in global_batches
        ]
        self.indices = [[i for i, _ in batch] for batch in batches_for_this_rank]

    def should_emit_current_batch(self, current_batch, slice):
        if not self.batch_size_tokens:
            batch_size_after_appending = len(current_batch) // self.chunk_size + 1
            if batch_size_after_appending > self.batch_size:
                return True
            else:
                return False
        else:
            global_batch_size_tokens = self.batch_size_tokens * self.chunk_size
            current_batch_tokens_after_appending = batch_size_tokens_after_padding(current_batch + slice)
            if len(current_batch) > 0 and current_batch_tokens_after_appending > global_batch_size_tokens:
                return True
            return False

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class PipelineDataLoader:
    def __init__(
        self,
        dataset,
        tokenizer,
        batch_size,
        gradient_accumulation_steps,
        data_parallel_world_size,
        data_parallel_rank,
        shuffle=True,
        group_by_length=False,
        pad_to_multiple_of=PAD_TO_MULTIPLE,
        batch_size_tokens=None,
        return_dict=False,
        rl=False,
    ):
        assert data_parallel_rank < data_parallel_world_size
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.batch_size_tokens = batch_size_tokens
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_dict = return_dict
        self.rl = rl
        self.data_sampler = DistributedBatchSamper(
            dataset=dataset,
            batch_size=self.batch_size,
            batch_size_tokens=self.batch_size_tokens,
            batch_size_multiplier=self.gradient_accumulation_steps,
            num_replicas=data_parallel_world_size,
            rank=data_parallel_rank,
            shuffle=shuffle,
            group_by_length=group_by_length,
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        def collate_fn(examples, gradient_accumulation_steps=self.gradient_accumulation_steps, flatten=False):
            if flatten:
                examples = flatten_examples(examples)
            rejected_examples = []
            for example in examples:
                example.pop('length', None)
                example.pop('token_type_ids', None)
                rejected_example = {}
                for key in list(example.keys()):
                    if 'rejected_' in key:
                        x = example.pop(key)
                        # Just drop the rejected_ entries if not doing RL. This allows normal SFT on just the
                        # accepted completions of a RL dataset.
                        if self.rl:
                            rejected_example[key.strip('rejected_')] = x
                if rejected_example:
                    rejected_examples.append(rejected_example)
            if rejected_examples:
                examples = combine_piecewise(examples, rejected_examples, gradient_accumulation_steps)
            return data_collator(examples)
        self.collate_fn = collate_fn

        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self.recreate_dataloader = False
        self._create_dataloader()
        self.data = self._pull_batches_from_dataloader()

    def reset(self):
        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self.data = self._pull_batches_from_dataloader()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data_sampler) * self.gradient_accumulation_steps

    def __next__(self):
        if self.next_micro_batch is None:
            self.next_micro_batch = next(self.data)
        ret = self.next_micro_batch
        try:
            self.next_micro_batch = next(self.data)
        except StopIteration:
            if self.recreate_dataloader:
                self._create_dataloader()
                self.recreate_dataloader = False
            self.data = self._pull_batches_from_dataloader()
            self.num_batches_pulled = 0
            self.next_micro_batch = next(self.data)
            self.epoch += 1
        return ret

    def _pull_batches_from_dataloader(self):
        for batch in self.dataloader:
            self.num_batches_pulled += 1
            for micro_batch in split_batch(batch, self.gradient_accumulation_steps):
                if self.return_dict:
                    yield micro_batch
                else:
                    # input to pipeline is (input_ids, attention_mask, labels)
                    # this needs to return (features, labels)
                    # it is OK if labels is None (the model just returns the loss anyway)
                    yield example_to_tuple(micro_batch)

    def _create_dataloader(self):
        self.dataloader = DataLoader(
            self.dataset,
            pin_memory=True,
            batch_sampler=self.data_sampler,
            collate_fn=self.collate_fn,
        )

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'num_batches_pulled': self.num_batches_pulled,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        # -1 because by preloading the next micro_batch, it's always going to have one more batch
        # pulled than the actual number of batches iterated by the caller.
        self.num_batches_pulled = state_dict['num_batches_pulled'] - 1
        self._create_dataloader()
        self.dataloader = accelerate.skip_first_batches(self.dataloader, self.num_batches_pulled)
        self.data = self._pull_batches_from_dataloader()
        # Recreate the dataloader after the first pass so that it won't skip
        # batches again (we only want it to skip batches the first time).
        self.recreate_dataloader = True

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


# for testing
if __name__ == '__main__':
    tokenizer = transformers.AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
    tokenizer.pad_token_id = 1000

    from datasets import Dataset

    data = []
    for i in range(1, 41):
        input_ids = torch.tensor([i] * i)
        data.append(
            {
                'input_ids': input_ids,
                'attention_mask': torch.ones_like(input_ids),
                'labels': input_ids,
                'length': len(input_ids),
            }
        )
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
        pad_to_multiple_of=None,
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
        pad_to_multiple_of=None,
    )
    dataloader.load_state_dict(state_dict)
    print()
    print('-' * 80)
    print()
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])
    print(next(dataloader)[0][0])
