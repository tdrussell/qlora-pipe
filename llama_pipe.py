import os

import torch
from torch import nn
import torch.nn.functional as F
import transformers
import accelerate
from deepspeed.runtime.pipe.module import LayerSpec
from deepspeed.accelerator import get_accelerator
from transformers.integrations import get_keys_to_not_convert, replace_with_bnb_linear


class EmbeddingPipe(nn.Module):
    def __init__(self, loader_util, orig, prepare_decoder_attention_mask):
        super().__init__()
        self.orig = orig
        self._prepare_decoder_attention_mask = prepare_decoder_attention_mask
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        input_ids, attention_mask, position_ids, labels = inputs
        inputs_embeds = self.orig(input_ids)
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, 0
        )
        hidden_states = self.orig(input_ids)
        # We have to do this so activation checkpointing with reentrant checkpoint function (the default) works.
        # We could just use non-reentrant instead, but that has some weird bug with flash attn where the memory usage is very high.
        hidden_states.requires_grad_(True)
        # Without flash attn, the attention_mask is a float. With pipeline parallel, any float tensors sent across GPUs must have requires_grad.
        # This is a workaround, theoretically there's no reason to require this.
        if torch.is_floating_point(attention_mask):
            attention_mask.requires_grad_(True)
        return hidden_states, attention_mask, position_ids, labels


class LlamaRMSNormPipe(nn.Module):
    def __init__(self, loader_util, orig):
        super().__init__()
        self.orig = orig
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, _, _, labels = inputs
        return self.orig(hidden_states), labels


class LmHeadPipe(nn.Module):
    def __init__(self, loader_util, orig):
        super().__init__()
        self.orig = orig
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, labels = inputs
        return self.orig(hidden_states), labels


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


class LlamaDecoderLayerPipe(nn.Module):
    def __init__(self, loader_util, orig):
        super().__init__()
        self.orig = orig
        self.mlp_offloaded_to_cpu = False
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, attention_mask, position_ids, labels = inputs
        if self.mlp_offloaded_to_cpu:
            cpu_up_proj = move_data_to_device(self.orig.mlp.up_proj, hidden_states.device)
            cpu_down_proj = move_data_to_device(self.orig.mlp.down_proj, hidden_states.device)
            cpu_gate_proj = move_data_to_device(self.orig.mlp.gate_proj, hidden_states.device)
        result = (self.orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0], attention_mask, position_ids, labels)
        if self.mlp_offloaded_to_cpu:
            set_data(self.orig.mlp.up_proj, cpu_up_proj)
            set_data(self.orig.mlp.down_proj, cpu_down_proj)
            set_data(self.orig.mlp.gate_proj, cpu_gate_proj)
        return result

    def offload_mlp_to_cpu(self):
        self.mlp_offloaded_to_cpu = True
        move_data_to_device(self.orig.mlp.up_proj, 'cpu')
        move_data_to_device(self.orig.mlp.down_proj, 'cpu')
        move_data_to_device(self.orig.mlp.gate_proj, 'cpu')


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


class LlamaForCausalLMPipe(transformers.LlamaForCausalLM):
    def __init__(self, model_path):
        config = transformers.LlamaConfig.from_pretrained(model_path)
        super().__init__(config)
        self.loader_util = LoaderUtil(model_path, self)

    def compute_metrics(self, inputs):
        logits, labels = inputs
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss_unreduced = torch.nn.CrossEntropyLoss(reduction='none')(shift_logits, shift_labels)
        # if we mask the labels, those loss values will be 0
        valid_loss = (loss_unreduced != 0)
        loss_unreduced = loss_unreduced[valid_loss]
        with torch.no_grad():
            accuracies = top_k_accuracy(shift_logits, shift_labels, k_list=[1, 5, 20])
            entropy = entropy_fn(shift_logits)[valid_loss]
        loss = loss_unreduced.mean()
        loss_unreduced = loss_unreduced.detach()
        return loss, loss_unreduced, entropy, *accuracies

    def to_layers(self):
        def initial_layer(inputs):
            input_ids, attention_mask, labels = inputs
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            return input_ids, attention_mask, position_ids, labels

        result = [
            initial_layer,
            EmbeddingPipe(self.model.embed_tokens, self.model._prepare_decoder_attention_mask),
        ]
        for block in self.model.layers:
            result.append(LlamaDecoderLayerPipe(block))
        result.append(LlamaRMSNormPipe(self.model.norm))
        result.append(LmHeadPipe(self.lm_head))
        result.append(self.compute_metrics)
        self.layer_list = result
        return result

    def to_layer_specs(self):
        def initial_layer(inputs):
            input_ids, attention_mask, labels = inputs
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            return input_ids, attention_mask, position_ids, labels

        result = [
            initial_layer,
            LayerSpec(EmbeddingPipe, self.loader_util, self.model.embed_tokens, self.model._prepare_decoder_attention_mask),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm))
        result.append(LayerSpec(LmHeadPipe, self.loader_util, self.lm_head))
        result.append(self.compute_metrics)
        return result


def load_pipeline_model(model_path, quantization_config=None, dtype=None):
    with accelerate.init_empty_weights():
        model = LlamaForCausalLMPipe(model_path)

    modules_to_not_convert = get_keys_to_not_convert(model)
    if not isinstance(modules_to_not_convert, list):
        modules_to_not_convert = [modules_to_not_convert]
    model = replace_with_bnb_linear(
        model, modules_to_not_convert=modules_to_not_convert, quantization_config=quantization_config
    )
    # Make sure to set this or PEFT (and probably other things) will break in strange ways.
    # We only need this because we do the loading and quanting ourselves.
    model.is_loaded_in_4bit = True
    if dtype is not None:
        model = model.to(dtype)

    for name, p in model.named_parameters():
        p.original_name = name

    return model


class LoaderUtil:

    def __init__(self, model_path, parent_model):
        self.model_path = model_path
        self.parent_model = parent_model
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