import os
import math
from inspect import signature

import torch
from torch import nn
import transformers
from deepspeed.accelerator import get_accelerator
from transformers.integrations import get_keys_to_not_convert
from deepspeed.runtime.pipe import module as ds_pipe_module
import bitsandbytes as bnb
import accelerate
from hqq.core import quantize as hqq_quantize

from utils import is_main_process
from kernels.cross_entropy_loss import Fast_CrossEntropyLoss
import hqq_utils


def move_data_to_device(module, device):
    # handle lora
    if hasattr(module, 'base_layer'):
        module = module.base_layer
    # handle HQQ
    if hasattr(module, 'W_q'):
        orig_data = module.W_q.data
        module.W_q.data = orig_data.to(device, non_blocking=True)
    else:
        orig_data = module.weight.data
        module.weight.data = orig_data.to(device, non_blocking=True)
    return orig_data


def set_data(module, data):
    # handle lora
    if hasattr(module, 'base_layer'):
        module = module.base_layer
    # handle HQQ
    if hasattr(module, 'W_q'):
        module.W_q.data = data
    else:
        module.weight.data = data


def entropy_fn(logits):
    result = []
    # There is a very wide range of chuck sizes that cause no increase in memory reported by
    # nvidia-smi (Torch re-using blocks of memory?). If you try to compute it as one tensor,
    # memory usage is huge. Chuck size of 128 seems good enough for now.
    for logits_chuck in torch.split(logits, 128):
        result.append(torch.distributions.Categorical(logits=logits_chuck).entropy())
    return torch.cat(result)


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
    def __init__(
            self,
            loss_type='cross_entropy_loss',
            focal_loss_gamma=0
        ):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.focal_loss_gamma = focal_loss_gamma

        if self.loss_type == 'cross_entropy_loss' and self.focal_loss_gamma != 0:
            raise ValueError("focal_loss_gamma can't be used with 'cross_entropy_loss' function")
        elif self.loss_type != 'cross_entropy_loss' and self.focal_loss_gamma <= 0:
            raise ValueError("focal_loss_gamma must be greater than 0 for the specified loss function")

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
        valid_loss = (shift_labels >= 0)

        cross_entropy_loss_unreduced = Fast_CrossEntropyLoss.apply(shift_logits, shift_labels)
        cross_entropy_loss_unreduced = cross_entropy_loss_unreduced[valid_loss]

        if self.loss_type == 'cross_entropy_loss':
            loss_unreduced = cross_entropy_loss_unreduced
        elif self.loss_type == 'focal_loss':
            # See https://arxiv.org/abs/1708.02002 (Section 3)
            p = torch.exp(-cross_entropy_loss_unreduced)
            loss_unreduced = (1-p)**self.focal_loss_gamma * cross_entropy_loss_unreduced
        elif self.loss_type == 'focal_loss_star':
            # See https://arxiv.org/abs/1708.02002 (Appendix A/B)
            # NOTE: The use of Beta makes no sense for the multinomial case as it's invariant to translation
            loss_unreduced = Fast_CrossEntropyLoss.apply(shift_logits, shift_labels, self.focal_loss_gamma)
            loss_unreduced = loss_unreduced[valid_loss]
            loss_unreduced = loss_unreduced / self.focal_loss_gamma
        elif self.loss_type == 'inverse_focal_loss':
            # See "Rethinking Calibration of Deep Neural Networks: Do Not Be Afraid of Overconfidence" (Section 5.2)
            # NOTE: The alternative of p^gamma (instead of (1+p)^gamma) might be useful for gradient ascent...
            p = torch.exp(-cross_entropy_loss_unreduced)
            loss_unreduced = (1+p)**self.focal_loss_gamma * cross_entropy_loss_unreduced
        elif self.loss_type == 'exponentiated_cross_entropy_loss':
            # See "Gradient as a Foundation for Building a Loss Function" (Section III.B)
            # NOTE: This is a generalisation of their "Quadratic Cross-Entropy" loss (QCE: gamma=2, CE: gamma=1, etc).
            loss_unreduced = cross_entropy_loss_unreduced**self.focal_loss_gamma / self.focal_loss_gamma
        else:
            raise NotImplementedError(self.loss_type)
        
        with torch.no_grad():
            log_vocab_size = math.log(logits.size(-1))
            entropy = entropy_fn(shift_logits)[valid_loss]
            # Compute normalised entropy so we can compare between models with different vocab sizes
            normalised_entropy = entropy / log_vocab_size         
            # Compute the (negative) log-likelihood using the original *UNADJUSTED* Cross-Entropy loss.
            log_likelihood = cross_entropy_loss_unreduced.mean()
            # Compute McFadden's Pseudo-R² metric using log(vocab_size) as the null log-likelihood.
            mcfaddens_pseudo_r2 = 1 - (log_likelihood / log_vocab_size)
            accuracies = top_k_accuracy(shift_logits, shift_labels, k_list=[1, 5, 20])
        loss = loss_unreduced.mean()
        loss_unreduced = loss_unreduced.detach()
        return loss, loss_unreduced, entropy, normalised_entropy, log_likelihood, mcfaddens_pseudo_r2, *accuracies


class PipelineModel(nn.Module):

    def __init__(self, config, quantization_config, model_config):
        if config['full_fine_tune'] and model_config.tie_word_embeddings:
            raise NotImplementedError('FFT is not supported for models with tied embeddings')
        self.train_config = config
        self.modules_to_not_quantize = get_keys_to_not_convert(self)
        self.loader_util = LoaderUtil(config['model'], quantization_config, self.modules_to_not_quantize)
        self.loss_type = config.get('loss_type', 'cross_entropy_loss').lower()
        self.focal_loss_gamma = config.get('focal_loss_gamma', 0)
        if self.focal_loss_gamma > 0 and is_main_process():
            print(f'Optimizing using \'{self.loss_type}\' with gamma={self.focal_loss_gamma}')

        for name, p in self.named_parameters():
            p.original_name = name

    # need to override this method
    def to_layer_specs(self):
        raise NotImplementedError()


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
    '''Replace a Linear layer with a BNB quantized version.'''
    if quantization_config.llm_int8_skip_modules is not None and _partial_module_name_match(full_name, quantization_config.llm_int8_skip_modules):
        return
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


def _replace_with_hqq_linear(parent_modules_map, name, full_name, quantization_config):
    '''Replace a Linear layer with a HQQ quantized version.'''
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
        del_orig=True
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
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
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
        self.local_rank = int(os.environ.get("LOCAL_RANK", None))
        assert self.local_rank is not None
        self.device = get_accelerator().device_name(self.local_rank)

        index_file = os.path.join(model_path, transformers.utils.SAFE_WEIGHTS_INDEX_NAME)
        if os.path.exists(index_file):
            checkpoint_files, checkpoint_metadata = transformers.utils.hub.get_checkpoint_shard_files(
                model_path,
                index_file,
                local_files_only=True
            )
            self.checkpoint_metadata = checkpoint_metadata
        else:
            self.checkpoint_metadata = None
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
            needed_checkpoint_files = set(weight_map[key.replace('orig.', '')] for key in expected_keys)
        else:
            needed_checkpoint_files = ['model.safetensors']

        for checkpoint_file in needed_checkpoint_files:
            state_dict = self.get_partial_state_dict(checkpoint_file)
            renamed_state_dict = {param_renaming_map[k]: v for k, v in state_dict.items() if k in param_renaming_map}
            # Use some transformers internals to avoid writing a bunch of code ourselves.
            # Might be a bit brittle...
            transformers.modeling_utils._load_state_dict_into_meta_model(
                module,
                renamed_state_dict,
                '',
                list(renamed_state_dict.keys()),
            )

        module.to(self.device)
        if not isinstance(self.quantization_config, transformers.BitsAndBytesConfig):
            self.maybe_quantize(module)
