import torch
from torch import nn
import transformers
import accelerate
from transformers.models.mixtral import modeling_mixtral

from pipeline_model import ComputeMetrics, LayerSpec, PipelineModel, move_data_to_device, set_data
from utils import DTYPE_MAP


class EmbeddingPipe(nn.Module):
    def __init__(self, loader_util, orig, attn_implementation, sliding_window, embedding_on_cpu=False):
        super().__init__()
        self.orig = orig
        self.attn_implementation = attn_implementation
        self.sliding_window = sliding_window
        self.embedding_on_cpu = embedding_on_cpu
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        input_ids, attention_mask, position_ids, labels = inputs
        original_device = input_ids.device
        if self.embedding_on_cpu:
            self.orig.to('cpu')
            input_ids = input_ids.to('cpu')
        inputs_embeds = self.orig(input_ids).to(original_device)
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


def move_experts_to_device(experts, device, num_experts_to_offload):
    orig_data = []
    for i in range(num_experts_to_offload):
        orig_w1 = move_data_to_device(experts[i].w1, device)
        orig_w2 = move_data_to_device(experts[i].w2, device)
        orig_w3 = move_data_to_device(experts[i].w3, device)
        orig_data.append((orig_w1, orig_w2, orig_w3))
    return orig_data


def set_experts_data(experts, orig_data):
    for i, (orig_w1, orig_w2, orig_w3) in enumerate(orig_data):
        set_data(experts[i].w1, orig_w1)
        set_data(experts[i].w2, orig_w2)
        set_data(experts[i].w3, orig_w3)


class MixtralDecoderLayerPipe(nn.Module):
    def __init__(self, loader_util, orig, num_experts_to_offload):
        super().__init__()
        self.orig = orig
        self.mlp_offloaded_to_cpu = False
        self.num_experts_to_offload = num_experts_to_offload
        loader_util.load_state_dict_into_module(self)

    # See note on MLP offloading in llama_pipe.py
    def forward(self, inputs):
        def set_cpu_data():
            set_experts_data(self.orig.block_sparse_moe.experts, orig_data)
        def set_cpu_data_hook(grad):
            set_cpu_data()
            return None

        hidden_states, attention_mask, position_ids, labels = inputs[:4]
        input_router_logits = inputs[4:]
        if self.mlp_offloaded_to_cpu:
            if hidden_states.requires_grad:
                hidden_states.register_hook(set_cpu_data_hook)
            orig_data = move_experts_to_device(self.orig.block_sparse_moe.experts, hidden_states.device, self.num_experts_to_offload)
        hidden_states, router_logits = self.orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids, output_router_logits=True)
        # TODO: fix unsloth gradient checkpointing when we return router logits
        #router_logits = router_logits.to(torch.float32)
        #router_logits = input_router_logits + (router_logits,)
        #result = (hidden_states, attention_mask, position_ids, labels, *router_logits)
        result = (hidden_states, attention_mask, position_ids, labels)
        if self.mlp_offloaded_to_cpu and not torch.is_grad_enabled():
            set_cpu_data()
        return result

    def offload_mlp_to_cpu(self):
        self.mlp_offloaded_to_cpu = True
        move_experts_to_device(self.orig.block_sparse_moe.experts, 'cpu', self.num_experts_to_offload)


class MixtralRMSNormPipe(nn.Module):
    def __init__(self, loader_util, orig):
        super().__init__()
        self.orig = orig
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, _, _, labels, *router_logits = inputs
        return self.orig(hidden_states), labels, *router_logits


class LmHeadPipe(nn.Module):
    def __init__(self, loader_util, lm_head):
        super().__init__()
        self.lm_head = lm_head
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, labels, *router_logits = inputs
        return self.lm_head(hidden_states), labels, *router_logits


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


class MixtralComputeMetrics(ComputeMetrics):
    def __init__(self, load_balancing_loss_coef, num_experts, num_experts_per_tok, **kwargs):
        super().__init__(**kwargs)
        self.load_balancing_loss_coef = load_balancing_loss_coef
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, inputs):
        logits, labels, *router_logits = inputs
        router_logits = tuple(router_logits)
        metrics = super().forward((logits, labels))
        if self.load_balancing_loss_coef is not None:
            aux_loss = modeling_mixtral.load_balancing_loss_func(
                router_logits, self.num_experts, self.num_experts_per_tok
            )
            alternate_aux_loss = load_balancing_loss_func(router_logits, self.num_experts, self.num_experts_per_tok)
            loss = metrics[0]
            loss += self.load_balancing_loss_coef * aux_loss
            metrics = (loss, *metrics[1:], aux_loss, alternate_aux_loss)
        return metrics


class MixtralForCausalLMPipe(PipelineModel, transformers.MixtralForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.MixtralConfig.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.MixtralForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)
        self.load_balancing_loss_coef = config.get('load_balancing_loss_coef', None)
        self.num_experts_to_offload = self.num_experts
        if 'offload_mlp_to_cpu' in config and type(config['offload_mlp_to_cpu']) == int:
            self.num_experts_to_offload = config['offload_mlp_to_cpu']

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
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model.config._attn_implementation,
                self.model.config.sliding_window,
                embedding_on_cpu=not self.train_config['full_fine_tune']
            )
        ]
        for block in self.model.layers:
            result.append(LayerSpec(MixtralDecoderLayerPipe, self.loader_util, block, self.num_experts_to_offload))
        result.append(LayerSpec(MixtralRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(LayerSpec(LmHeadPipe, self.loader_util, self.lm_head, _estimated_size=0))
        result.append(
            LayerSpec(
                MixtralComputeMetrics,
                load_balancing_loss_coef=self.load_balancing_loss_coef,
                num_experts=self.num_experts,
                num_experts_per_tok=self.num_experts_per_tok,
                loss_function=self.loss_function,
                use_gradient_ascent=self.use_gradient_ascent,
                focal_loss_gamma=self.focal_loss_gamma,
                _estimated_size=0
            )
        )
        return result
