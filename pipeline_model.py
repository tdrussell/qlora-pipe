import os
from inspect import signature

import torch
from torch import nn
import transformers
from deepspeed.accelerator import get_accelerator
from transformers.integrations import get_keys_to_not_convert
from deepspeed.runtime.pipe import module as ds_pipe_module
import bitsandbytes as bnb
import accelerate

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
        self.modules_to_not_quantize = get_keys_to_not_convert(self)
        self.loader_util = LoaderUtil(config['model'], quantization_config, self.modules_to_not_quantize)
        self.focal_loss_gamma = config.get('focal_loss_gamma', 0)
        if self.focal_loss_gamma > 0 and is_main_process():
            print(f'Using focal loss with gamma={self.focal_loss_gamma}')

        for name, p in self.named_parameters():
            p.original_name = name

    # need to override this method
    def to_layer_specs(self):
        raise NotImplementedError()


def _replace_with_bnb_linear(parent_modules_map, name, quantization_config):
    '''Replace a Linear layer with a BNB quantized version.'''
    module = parent_modules_map[name]
    with accelerate.init_empty_weights():
        if isinstance(module, nn.Conv1d):
            in_features, out_features = module.weight.shape
        else:
            in_features = module.in_features
            out_features = module.out_features

        if quantization_config.quantization_method() == "llm_int8":
            parent_modules_map[name] = bnb.nn.Linear8bitLt(
                in_features,
                out_features,
                module.bias is not None,
                has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                threshold=quantization_config.llm_int8_threshold,
            )
        else:
            if (
                quantization_config.llm_int8_skip_modules is not None
                and name in quantization_config.llm_int8_skip_modules
            ):
                pass
            else:
                extra_kwargs = (
                    {"quant_storage": quantization_config.bnb_4bit_quant_storage}
                    if "quant_storage" in list(signature(bnb.nn.Linear4bit).parameters)
                    else {}
                )
                parent_modules_map[name] = bnb.nn.Linear4bit(
                    in_features,
                    out_features,
                    module.bias is not None,
                    quantization_config.bnb_4bit_compute_dtype,
                    compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                    quant_type=quantization_config.bnb_4bit_quant_type,
                    **extra_kwargs,
                )
        # Store the module class in case we need to transpose the weight later
        parent_modules_map[name].source_cls = type(module)
        # Force requires grad to False to avoid unexpected errors
        parent_modules_map[name].requires_grad_(False)


# modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/bitsandbytes.py
def _recursively_replace_with_quantized_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
):
    """
    Returns the converted model and a boolean that indicates if the conversion has been successful or not.
    """
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if (isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d)) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                _replace_with_bnb_linear(model._modules, name, quantization_config)

                # copy over the original_name attribute we added earlier (needed for loading weights)
                for orig_name, orig_p in module.named_parameters():
                    if hasattr(orig_p, 'original_name'):
                        for new_name, new_p in model._modules[name].named_parameters():
                            if new_name == orig_name:
                                new_p.original_name = orig_p.original_name

        if len(list(module.children())) > 0:
            _recursively_replace_with_quantized_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)


class LoaderUtil:

    def __init__(self, model_path, quantization_config, modules_to_not_quantize):
        self.model_path = model_path
        self.quantization_config = quantization_config
        self.modules_to_not_quantize = modules_to_not_quantize
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

    def maybe_quantize(self, module):
        if self.quantization_config is None:
            return
        modules_to_not_convert = self.modules_to_not_quantize
        if not isinstance(modules_to_not_convert, list):
            modules_to_not_convert = [modules_to_not_convert]
        _recursively_replace_with_quantized_linear(
            module, modules_to_not_convert=modules_to_not_convert, quantization_config=self.quantization_config
        )
        # Make sure to set this or PEFT (and probably other things) will break in strange ways.
        # We only need this because we do the loading and quanting ourselves.
        self.is_loaded_in_4bit = True

    def load_state_dict_into_module(self, module):
        print(f'load params into module {type(module)}')
        if isinstance(self.quantization_config, transformers.utils.quantization_config.BitsAndBytesConfig):
            # bnb needs to replace with quantized linear before weights are loaded
            self.maybe_quantize(module)
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

        if not isinstance(self.quantization_config, transformers.utils.quantization_config.BitsAndBytesConfig):
            self.maybe_quantize(module)
        module.to(self.device)
