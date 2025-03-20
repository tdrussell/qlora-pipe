import os
from collections import defaultdict
from inspect import signature
import re

import accelerate
import bitsandbytes as bnb
import transformers
from deepspeed.accelerator import get_accelerator
from hqq.core import quantize as hqq_quantize
from torch import nn
from transformers.integrations import get_keys_to_not_convert
from accelerate.utils import set_module_tensor_to_device

import utils.hqq_utils as hqq_utils
from utils.utils import is_main_process


LANGUAGE_MODEL_WEIGHT_PREFIX_REGEX = r'^language_model\.'


class PipelineModel(nn.Module):
    def __init__(self, config, quantization_config, model_config):
        if config['full_fine_tune'] and model_config.tie_word_embeddings:
            raise NotImplementedError('FFT is not supported for models with tied embeddings')
        self.train_config = config
        self.model_config = model_config
        self.modules_to_not_quantize = get_keys_to_not_convert(self)
        self.loader_util = LoaderUtil(config['model'], quantization_config, self.modules_to_not_quantize)
        self.loss_type = config.get('loss_type', 'cross_entropy_loss').lower()
        if rl_config := config.get('rl', None):
            self.loss_type = rl_config['method']
        self.focal_loss_gamma = config.get('focal_loss_gamma', 0)
        if self.focal_loss_gamma > 0 and is_main_process():
            print(f"Optimizing using '{self.loss_type}' with gamma={self.focal_loss_gamma}")
        self.dpo_reference_mode = False
        self.sampling_mode = False

        for name, p in self.named_parameters():
            p.original_name = name

    # need to override this method
    def to_layer_specs(self):
        raise NotImplementedError()

    def set_dpo_reference_mode(self, dpo_reference_mode):
        self.dpo_reference_mode = dpo_reference_mode

    def set_sampling_mode(self, sampling_mode):
        self.sampling_mode = sampling_mode
        # Reset cache when sampling mode is modified. This ensures it's initialized and also clears memory at the end.
        self.cache_dict = defaultdict(transformers.DynamicCache)
        # We could try to use static cache at some point. During early testing with relatively short sequence lengths,
        # it was the same sampling speed as DynamicCache. Note: will need to pass cache_position in transformer layer
        # if using StaticCache.
        # def make_static_cache():
        #     return transformers.StaticCache(
        #         self.model_config,
        #         max_batch_size=1,
        #         max_cache_len=1024,
        #         device='cuda',
        #         dtype=self.dtype,
        #     )
        # self.cache_dict = defaultdict(make_static_cache)

    def set_cache(self, micro_batch_id):
        self.cache = self.cache_dict[micro_batch_id]


def _partial_module_name_match(full_name, list_to_match):
    return any(key in full_name for key in list_to_match)


def _replace_with_quantized_linear(parent_modules_map, name, full_name, quantization_config):
    if isinstance(quantization_config, transformers.BitsAndBytesConfig):
        _replace_with_bnb_linear(parent_modules_map, name, full_name, quantization_config)
    elif isinstance(quantization_config, hqq_utils.CustomHQQConfig):
        _replace_with_hqq_linear(parent_modules_map, name, full_name, quantization_config)
    else:
        raise NotImplementedError(f'Quantization config not implemented: {quantization_config}')


def _replace_with_bnb_linear(parent_modules_map, name, full_name, quantization_config):
    """Replace a Linear layer with a BNB quantized version."""
    if quantization_config.llm_int8_skip_modules is not None and _partial_module_name_match(
        full_name, quantization_config.llm_int8_skip_modules
    ):
        return
    module = parent_modules_map[name]
    with accelerate.init_empty_weights():
        if isinstance(module, nn.Conv1d):
            in_features, out_features = module.weight.shape
        else:
            in_features = module.in_features
            out_features = module.out_features

        if quantization_config.quantization_method() == 'llm_int8':
            parent_modules_map[name] = bnb.nn.Linear8bitLt(
                in_features,
                out_features,
                module.bias is not None,
                has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                threshold=quantization_config.llm_int8_threshold,
            )
        else:
            extra_kwargs = (
                {'quant_storage': quantization_config.bnb_4bit_quant_storage}
                if 'quant_storage' in list(signature(bnb.nn.Linear4bit).parameters)
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


def _replace_with_hqq_linear(parent_modules_map, name, full_name, quantization_config):
    """Replace a Linear layer with a HQQ quantized version."""
    if _partial_module_name_match(full_name, quantization_config.skip_modules):
        return
    module = parent_modules_map[name]
    quant_config_dict = quantization_config.get_dict(full_name)
    hqq_linear = hqq_quantize.HQQLinear(
        module,
        quant_config=quant_config_dict,
        compute_dtype=quantization_config.compute_dtype,
        device=module.weight.device,
        initialize=True,
        del_orig=True,
    )
    # Quantization itself uses a decent amount of VRAM. Temporarily move each quantized parameter to the CPU as we
    # finish, so the quant process doesn't OOM. Deepspeed will move everything to the correct device later.
    hqq_linear.W_q.data = hqq_linear.W_q.data.to('cpu')
    # Store the module class in case we need to transpose the weight later
    hqq_linear.source_cls = type(module)
    # Force requires grad to False to avoid unexpected errors
    hqq_linear.requires_grad_(False)
    parent_modules_map[name] = hqq_linear


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
            current_key_name_str = '.'.join(current_key_name)
            if not any(
                (key + '.' in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                _replace_with_quantized_linear(model._modules, name, current_key_name_str, quantization_config)

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
        self.local_rank = int(os.environ.get('LOCAL_RANK', None))
        assert self.local_rank is not None
        self.device = get_accelerator().device_name(self.local_rank)

        index_file = os.path.join(model_path, transformers.utils.SAFE_WEIGHTS_INDEX_NAME)
        if os.path.exists(index_file):
            checkpoint_files, checkpoint_metadata = transformers.utils.hub.get_checkpoint_shard_files(
                model_path, index_file, local_files_only=True
            )
            self.checkpoint_metadata = checkpoint_metadata
        else:
            self.checkpoint_metadata = None
        self.loaded_state_dict = None

    def get_partial_state_dict(self, leaf_file):
        if self.loaded_state_dict is None or leaf_file != self.loaded_state_dict[0]:
            print(f'loading checkpoint file {leaf_file}')
            state_dict = transformers.modeling_utils.load_state_dict(os.path.join(self.model_path, leaf_file))
            state_dict = {re.sub(LANGUAGE_MODEL_WEIGHT_PREFIX_REGEX, '', k): v for k, v in state_dict.items()}
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
        if isinstance(self.quantization_config, transformers.BitsAndBytesConfig):
            # bnb needs to replace with quantized linear before weights are loaded
            self.maybe_quantize(module)
        param_renaming_map = {p.original_name: new_name for new_name, p in module.named_parameters()}
        expected_keys = [p.original_name for p in module.parameters()]
        # If we have any extra attributes on the parameter, loading with BNB 4bit params breaks, so delete them.
        for p in module.parameters():
            del p.original_name

        if self.checkpoint_metadata is not None:
            weight_map = self.checkpoint_metadata['weight_map']
            weight_map = {re.sub(LANGUAGE_MODEL_WEIGHT_PREFIX_REGEX, '', k): v for k, v in weight_map.items()}
            needed_checkpoint_files = {weight_map[key.replace('orig.', '')] for key in expected_keys}
        else:
            needed_checkpoint_files = ['model.safetensors']

        for checkpoint_file in needed_checkpoint_files:
            state_dict = self.get_partial_state_dict(checkpoint_file)
            renamed_state_dict = {param_renaming_map[k]: v for k, v in state_dict.items() if k in param_renaming_map}
            for name, param in module.named_parameters():
                if name in renamed_state_dict:
                    set_module_tensor_to_device(module, name, device='cpu', value=renamed_state_dict[name])

        module.to(self.device)
        if not isinstance(self.quantization_config, transformers.BitsAndBytesConfig):
            self.maybe_quantize(module)
