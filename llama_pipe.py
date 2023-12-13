import torch
from torch import nn
import torch.nn.functional as F
import transformers
import accelerate
from deepspeed.runtime.pipe.module import LayerSpec

from pipeline_model import PipelineModel


class EmbeddingPipe(nn.Module):
    def __init__(self, loader_util, orig, use_flash_attention_2):
        super().__init__()
        self.orig = orig
        self.use_flash_attention_2 = use_flash_attention_2
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        input_ids, attention_mask, position_ids, labels = inputs
        inputs_embeds = self.orig(input_ids)
        batch_size, seq_length = input_ids.shape

        if self.use_flash_attention_2:
            # 2d mask is passed through the layers
            assert attention_mask is not None
            assert len(attention_mask.size()) == 2
        else:
            # 4d mask is passed through the layers
            attention_mask = transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, 0
            )

        hidden_states = inputs_embeds
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


# A little bit of inheritance and MRO trickery since LlamaForCausalLM.__init__ only takes a
# positional argument. We inherit PipelineModel first, but call LlamaForCausalLM init first,
# and make sure PipelineModel doesn't have a super().__init__() call.
class LlamaForCausalLMPipe(PipelineModel, transformers.LlamaForCausalLM):
    def __init__(self, model_path, quantization_config):
        config = transformers.LlamaConfig.from_pretrained(model_path)
        config._attn_implementation = 'flash_attention_2'
        # we can't be float32 when constructing the model or it complains because
        # of flash attention
        torch.set_default_dtype(torch.bfloat16)
        with accelerate.init_empty_weights():
            transformers.LlamaForCausalLM.__init__(self, config)
        PipelineModel.__init__(self, model_path, quantization_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        def initial_layer(inputs):
            input_ids, attention_mask, labels = inputs
            batch_size, seq_length = input_ids.shape[:2]
            device = input_ids.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
            return input_ids, attention_mask, position_ids, labels

        result = [
            initial_layer,
            LayerSpec(EmbeddingPipe, self.loader_util, self.model.embed_tokens, self.model._use_flash_attention_2),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm))
        result.append(LayerSpec(LmHeadPipe, self.loader_util, self.lm_head))
        result.append(self.compute_metrics)
        return result
