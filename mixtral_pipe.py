import torch
from torch import nn
import torch.nn.functional as F
import transformers
import accelerate
from deepspeed.runtime.pipe.module import LayerSpec
from transformers.models.mixtral import modeling_mixtral

from pipeline_model import PipelineModel


class EmbeddingPipe(nn.Module):
    def __init__(self, loader_util, orig, attn_implementation, sliding_window):
        super().__init__()
        self.orig = orig
        self.attn_implementation = attn_implementation
        self.sliding_window = sliding_window
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        input_ids, attention_mask, position_ids, labels = inputs
        inputs_embeds = self.orig(input_ids)
        batch_size, seq_length = input_ids.shape

        if self.attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            assert attention_mask is not None
            assert len(attention_mask.size()) == 2
        else:
            # 4d mask is passed through the layers
            attention_mask = transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                0,
                sliding_window=self.sliding_window,
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


# TODO: make MLP offloading work (right now it's just copied from LlamaDecoderLayerPipe)
class MixtralDecoderLayerPipe(nn.Module):
    def __init__(self, loader_util, orig):
        super().__init__()
        self.orig = orig
        self.mlp_offloaded_to_cpu = False
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, attention_mask, position_ids, labels = inputs[:4]
        input_router_logits = inputs[4:]
        if self.mlp_offloaded_to_cpu:
            cpu_up_proj = move_data_to_device(self.orig.mlp.up_proj, hidden_states.device)
            cpu_down_proj = move_data_to_device(self.orig.mlp.down_proj, hidden_states.device)
            cpu_gate_proj = move_data_to_device(self.orig.mlp.gate_proj, hidden_states.device)
        hidden_states, router_logits = self.orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids, output_router_logits=True)
        router_logits = router_logits.to(torch.float32)
        router_logits = input_router_logits + (router_logits,)
        result = (hidden_states, attention_mask, position_ids, labels, *router_logits)
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


class MixtralRMSNormPipe(nn.Module):
    def __init__(self, loader_util, orig):
        super().__init__()
        self.orig = orig
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, _, _, labels, *router_logits = inputs
        return self.orig(hidden_states), labels, *router_logits


class LmHeadPipe(nn.Module):
    def __init__(self, loader_util, orig):
        super().__init__()
        self.orig = orig
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, labels, *router_logits = inputs
        return self.orig(hidden_states), labels, *router_logits


def load_balancing_loss_func(gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2) -> float:
    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        stacked_gate_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(stacked_gate_logits, dim=-1) # [num_layers, num_tokens, num_experts]
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1) # [num_layers, num_tokens, top_k]
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts) # [num_layers, num_tokens, top_k, num_experts]
    # For a given token, determine if it was routed to a given expert. Think of this as a collection of top_k-hot vectors.
    expert_mask = torch.max(expert_mask, dim=-2).values.float() # [num_layers, num_tokens, num_experts]
    tokens_per_layer_and_expert = torch.mean(expert_mask, dim=-2) # [num_layers, num_experts]
    router_prob_per_layer_and_expert = torch.mean(routing_weights, dim=-2) # [num_layers, num_experts]
    return torch.mean(tokens_per_layer_and_expert * router_prob_per_layer_and_expert) * num_experts**2


class MixtralForCausalLMPipe(PipelineModel, transformers.MixtralForCausalLM):
    def __init__(self, model_path, quantization_config, load_balancing_loss_coef=None, **kwargs):
        config = transformers.MixtralConfig.from_pretrained(model_path)
        config._attn_implementation = 'flash_attention_2'
        # we can't be float32 when constructing the model or it complains because
        # of flash attention
        torch.set_default_dtype(torch.bfloat16)
        with accelerate.init_empty_weights():
            transformers.MixtralForCausalLM.__init__(self, config)
        PipelineModel.__init__(self, model_path, quantization_config, **kwargs)
        torch.set_default_dtype(torch.float32)
        self.load_balancing_loss_coef = load_balancing_loss_coef

    def compute_metrics(self, inputs):
        logits, labels, *router_logits = inputs
        router_logits = tuple(router_logits)
        metrics = super().compute_metrics((logits, labels))
        if self.load_balancing_loss_coef is not None:
            aux_loss = modeling_mixtral.load_balancing_loss_func(
                router_logits, self.num_experts, self.num_experts_per_tok
            )
            alternate_aux_loss = load_balancing_loss_func(router_logits, self.num_experts, self.num_experts_per_tok)
            loss = metrics[0]
            loss += self.load_balancing_loss_coef * aux_loss
            metrics = (loss, *metrics[1:], aux_loss, alternate_aux_loss)
        return metrics

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
            LayerSpec(EmbeddingPipe, self.loader_util, self.model.embed_tokens, self.model.config._attn_implementation, self.model.config.sliding_window),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(MixtralDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(MixtralRMSNormPipe, self.loader_util, self.model.norm))
        result.append(LayerSpec(LmHeadPipe, self.loader_util, self.lm_head))
        result.append(self.compute_metrics)
        return result
