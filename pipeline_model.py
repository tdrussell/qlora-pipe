import os

import torch
from torch import nn
import transformers
from deepspeed.accelerator import get_accelerator
from transformers.integrations import get_keys_to_not_convert, replace_with_bnb_linear
from deepspeed.runtime.pipe import module as ds_pipe_module

from utils import is_main_process
from kernels.cross_entropy_loss import Fast_CrossEntropyLoss


def move_data_to_device(module, device):
    # handle lora
    if hasattr(module, 'base_layer'):
        module = module.base_layer
    orig_data = module.weight.data
    module.weight.data = orig_data.to(device, non_blocking=True)
    return orig_data


def set_data(module, data):
    # handle lora
    if hasattr(module, 'base_layer'):
        module = module.base_layer
    module.weight.data = data


def entropy_fn(logits):
    result = []
    # There is a very wide range of chuck sizes that cause no increase in memory reported by
    # nvidia-smi (Torch re-using blocks of memory?). If you try to compute it as one tensor,
    # memory usage is huge. Chuck size of 128 seems good enough for now.
    for logits_chuck in torch.split(logits, 128):
        result.append(torch.distributions.Categorical(logits=logits_chuck).entropy())
    return torch.cat(result).float()


def top_k_accuracy(logits, labels, k_list, ignore_index=-100):
    keep = (labels != ignore_index)
    labels = labels[keep].view(-1, 1)
    max_k = max(k_list)
    _, top_k_predictions = torch.topk(logits, max_k, dim=-1, sorted=True)
    top_k_predictions = top_k_predictions[keep]
    accuracies = []
    for k in k_list:
        accuracies.append(torch.any(top_k_predictions[:, :k] == labels, dim=-1).to(torch.float32).mean())
    return accuracies


class LayerSpec(ds_pipe_module.LayerSpec):
    def __init__(self, typename, *module_args, **module_kwargs):
        super().__init__(typename, *module_args, **module_kwargs)

    def build(self):
        self.module_kwargs.pop('_estimated_size', None)
        return self.typename(*self.module_args, **self.module_kwargs)

    @property
    def estimated_size(self):
        return self.module_kwargs.get('_estimated_size', 1)


class ComputeMetrics(nn.Module):
    def __init__(self, focal_loss_gamma=0):
        super().__init__()
        self.focal_loss_gamma = focal_loss_gamma

    def forward(self, inputs):
        logits, labels = inputs
        shift_logits = logits
        extra_ignored_labels = torch.full((labels.shape[0], 1), -100, device=logits.device)
        shift_labels = torch.hstack((labels[..., 1:], extra_ignored_labels))
        # Flatten the tokens
        vocab_size = shift_logits.size(-1)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss_unreduced = Fast_CrossEntropyLoss.apply(shift_logits, shift_labels)

        valid_loss = (shift_labels >= 0)
        loss_unreduced = loss_unreduced[valid_loss]
        optimized_loss_unreduced = loss_unreduced

        # focal loss
        if (self.focal_loss_gamma > 0):
            p = torch.exp(-optimized_loss_unreduced)
            optimized_loss_unreduced = (1-p)**self.focal_loss_gamma * optimized_loss_unreduced

        with torch.no_grad():
            accuracies = top_k_accuracy(shift_logits, shift_labels, k_list=[1, 5, 20])
            entropy = entropy_fn(shift_logits)[valid_loss]
        loss = optimized_loss_unreduced.mean()
        loss_unreduced = loss_unreduced.detach()
        return loss, loss_unreduced, entropy, *accuracies


class PipelineModel(nn.Module):

    def __init__(self, config, quantization_config):
        self.train_config = config
        self.loader_util = LoaderUtil(config['model'])
        self.focal_loss_gamma = config.get('focal_loss_gamma', 0)
        if self.focal_loss_gamma > 0 and is_main_process():
            print(f'Using focal loss with gamma={self.focal_loss_gamma}')

        if quantization_config is not None:
            modules_to_not_convert = get_keys_to_not_convert(self)
            if not isinstance(modules_to_not_convert, list):
                modules_to_not_convert = [modules_to_not_convert]
            replace_with_bnb_linear(
                self, modules_to_not_convert=modules_to_not_convert, quantization_config=quantization_config
            )
            # Make sure to set this or PEFT (and probably other things) will break in strange ways.
            # We only need this because we do the loading and quanting ourselves.
            self.is_loaded_in_4bit = True

        for name, p in self.named_parameters():
            p.original_name = name

    # need to override this method
    def to_layer_specs(self):
        raise NotImplementedError()


class LoaderUtil:

    def __init__(self, model_path):
        self.model_path = model_path
        self.local_rank = int(os.environ.get("LOCAL_RANK", None))
        assert self.local_rank is not None
        self.device = get_accelerator().device_name(self.local_rank)

        index_file = os.path.join(model_path, transformers.utils.SAFE_WEIGHTS_INDEX_NAME)
        checkpoint_files, checkpoint_metadata = transformers.utils.hub.get_checkpoint_shard_files(
            model_path,
            index_file,
            local_files_only=True
        )
        self.checkpoint_files = checkpoint_files
        self.checkpoint_metadata = checkpoint_metadata
        self.loaded_state_dict = None

    def get_partial_state_dict(self, leaf_file):
        if self.loaded_state_dict is None or leaf_file != self.loaded_state_dict[0]:
            print(f'loading checkpoint file {leaf_file}')
            state_dict = transformers.modeling_utils.load_state_dict(os.path.join(self.model_path, leaf_file))
            self.loaded_state_dict = (leaf_file, state_dict)
        return self.loaded_state_dict[1]

    def load_state_dict_into_module(self, module):
        print(f'load params into module {type(module)}')
        param_renaming_map = {p.original_name: new_name for new_name, p in module.named_parameters()}
        expected_keys = [p.original_name for p in module.parameters()]
        weight_map = self.checkpoint_metadata['weight_map']
        needed_checkpoint_files = set(weight_map[key.replace('orig.', '')] for key in expected_keys)
        for checkpoint_file in needed_checkpoint_files:
            state_dict = self.get_partial_state_dict(checkpoint_file)
            renamed_state_dict = {param_renaming_map[k]: v for k, v in state_dict.items() if k in param_renaming_map}
            # Use some transformers internals to avoid writing a bunch of code ourselves.
            # Might be a bit brittle...
            transformers.modeling_utils._load_state_dict_into_meta_model(
                module,
                renamed_state_dict,
                list(renamed_state_dict.keys()),
                '',
                [name for name, p in module.named_parameters()]
            )

        # Quantization happens here, if needed.
        module.to(self.device)
