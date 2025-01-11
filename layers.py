import math

import torch
import torch.nn.functional as F
import transformers
from deepspeed.runtime.pipe import module as ds_pipe_module
from torch import nn

from kernels.cross_entropy_loss import Fast_CrossEntropyLoss


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


def entropy_fn(logits):
    result = []
    # There is a very wide range of chuck sizes that cause no increase in memory reported by
    # nvidia-smi (Torch re-using blocks of memory?). If you try to compute it as one tensor,
    # memory usage is huge. Chuck size of 128 seems good enough for now.
    for logits_chuck in torch.split(logits, 128):
        result.append(torch.distributions.Categorical(logits=logits_chuck).entropy())
    return torch.cat(result).float()


def top_k_accuracy(logits, labels, k_list, ignore_index=-100):
    keep = labels != ignore_index
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


# TODO: consider using Liger-Kernel fused loss implementations. The inputs are already set up to support this.
# Would save VRAM, but some metrics could no longer be computed (e.g. entropy, accuracies).
class OutputLayer(nn.Module):
    def __init__(
        self,
        pipeline_model,
        loader_util,
        lm_head,
        logit_scale=1.0,
        loss_type='cross_entropy_loss',
        focal_loss_gamma=0,
        tie_weights=None,
        logit_softcapping=None,
    ):
        super().__init__()
        # Assign list to prevent registering the nn.Module
        self.pipeline_model = [pipeline_model]
        # Unlike the other wrapper classes, this is called lm_head and not orig. Because this is directly a
        # nn.Linear layer, it needs to keep the same attribute name so quantization knows not to quantize it.
        self.lm_head = lm_head
        self.logit_scale = logit_scale
        self.loss_type = loss_type.lower()
        self.focal_loss_gamma = focal_loss_gamma
        if tie_weights:
            self.lm_head.weight.original_name = tie_weights
        self.logit_softcapping = logit_softcapping
        loader_util.load_state_dict_into_module(self)

        if self.loss_type == 'cross_entropy_loss' and self.focal_loss_gamma != 0:
            raise ValueError("focal_loss_gamma can't be used with 'cross_entropy_loss' function")

    def forward(self, inputs):
        hidden_states, labels = inputs
        if self.pipeline_model[0].sampling_mode:
            # When sampling only compute the last logits.
            hidden_states = hidden_states[:, -1:, :]
        labels = labels.to(hidden_states.device)
        if self.logit_scale == 1.0:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(self.logit_scale * hidden_states)
        if self.logit_softcapping is not None and self.logit_softcapping > 0:
            logits = logits / self.logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.logit_softcapping

        if self.pipeline_model[0].sampling_mode:
            return logits

        extra_ignored_labels = torch.full((labels.shape[0], 1), -100, device=logits.device)
        labels = torch.hstack((labels[..., 1:], extra_ignored_labels))
        # Flatten the tokens
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)
        flat_loss_mask = flat_labels >= 0

        cross_entropy_loss = Fast_CrossEntropyLoss.apply(flat_logits, flat_labels)

        loss = None
        if self.loss_type == 'cross_entropy_loss':
            cross_entropy_loss = cross_entropy_loss[flat_loss_mask]
            loss_unreduced = cross_entropy_loss
        elif self.loss_type == 'focal_loss':
            cross_entropy_loss = cross_entropy_loss[flat_loss_mask]
            # See https://arxiv.org/abs/1708.02002 (Section 3)
            p = torch.exp(-cross_entropy_loss)
            loss_unreduced = (1 - p) ** self.focal_loss_gamma * cross_entropy_loss
        elif self.loss_type == 'focal_loss_star':
            cross_entropy_loss = cross_entropy_loss[flat_loss_mask]
            # See https://arxiv.org/abs/1708.02002 (Appendix A/B)
            # NOTE: The use of Beta makes no sense for the multinomial case as it's invariant to translation
            loss_unreduced = Fast_CrossEntropyLoss.apply(flat_logits, flat_labels, self.focal_loss_gamma)
            loss_unreduced = loss_unreduced[flat_loss_mask]
            loss_unreduced = loss_unreduced / self.focal_loss_gamma
        elif self.loss_type == 'inverse_focal_loss':
            cross_entropy_loss = cross_entropy_loss[flat_loss_mask]
            # See "Rethinking Calibration of Deep Neural Networks: Do Not Be Afraid of Overconfidence" (Section 5.2)
            # NOTE: The alternative of p^gamma (instead of (1+p)^gamma) might be useful for gradient ascent...
            p = torch.exp(-cross_entropy_loss)
            loss_unreduced = (1 + p) ** self.focal_loss_gamma * cross_entropy_loss
        elif self.loss_type == 'exponentiated_cross_entropy_loss':
            cross_entropy_loss = cross_entropy_loss[flat_loss_mask]
            # See "Gradient as a Foundation for Building a Loss Function" (Section III.B)
            # NOTE: This is a generalisation of their "Quadratic Cross-Entropy" loss (QCE: gamma=2, CE: gamma=1, etc).
            loss_unreduced = cross_entropy_loss**self.focal_loss_gamma / self.focal_loss_gamma
        elif self.loss_type == 'dpo':
            rl_config = self.pipeline_model[0].train_config['rl']
            cross_entropy_loss = cross_entropy_loss.view_as(labels)  # unflatten
            loss_mask = labels >= 0
            logps = -(cross_entropy_loss * loss_mask).sum(-1)
            half = cross_entropy_loss.size(0) // 2
            chosen_logps = logps[:half]
            rejected_logps = logps[half:]

            if self.pipeline_model[0].dpo_reference_mode:
                self.reference_chosen_logps = chosen_logps.detach()
                self.reference_rejected_logps = rejected_logps.detach()
                return torch.tensor(0.0, device=logits.device)

            # log the language modeling loss metrics on the chosen completion
            cross_entropy_loss = cross_entropy_loss[:half].flatten()[loss_mask[:half].flatten()]
            loss_unreduced = cross_entropy_loss
            flat_logits = logits[:half].view(-1, vocab_size)
            flat_labels = labels[:half].view(-1)
            flat_loss_mask = flat_labels >= 0

            policy_chosen_logps = chosen_logps
            policy_rejected_logps = rejected_logps
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = self.reference_chosen_logps - self.reference_rejected_logps
            del self.reference_chosen_logps
            del self.reference_rejected_logps
            dpo_logits = pi_logratios - ref_logratios
            loss = -F.logsigmoid(rl_config['dpo_beta'] * dpo_logits).mean()
        else:
            raise NotImplementedError(self.loss_type)

        with torch.no_grad():
            log_vocab_size = math.log(logits.size(-1))
            entropy = entropy_fn(flat_logits)[flat_loss_mask]
            # Compute normalised entropy so we can compare between models with different vocab sizes
            normalised_entropy = entropy / log_vocab_size
            # Compute the (negative) log-likelihood using the original *UNADJUSTED* Cross-Entropy loss.
            log_likelihood = cross_entropy_loss.mean()
            # Compute McFadden's Pseudo-R² metric using log(vocab_size) as the null log-likelihood.
            mcfaddens_pseudo_r2 = 1 - (log_likelihood / log_vocab_size)
            accuracies = top_k_accuracy(flat_logits, flat_labels, k_list=[1, 5, 20])
            # Compute the norms of the (pre-logit-scaled) hidden states
            hidden_state_norms = torch.norm(hidden_states.float(), dim=-1)
            hidden_state_norms = hidden_state_norms.view(-1)[flat_loss_mask]
        if loss is None:
            # Normal language modeling loss types (e.g. not DPO)
            loss = loss_unreduced.mean()
        loss_unreduced = loss_unreduced.detach()
        return (
            loss,
            loss_unreduced,
            hidden_state_norms,
            entropy,
            normalised_entropy,
            log_likelihood,
            mcfaddens_pseudo_r2,
            *accuracies,
        )


def load_balancing_loss_func(gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2) -> float:
    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        stacked_gate_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(stacked_gate_logits, dim=-1)  # [num_layers, num_tokens, num_experts]
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)  # [num_layers, num_tokens, top_k]
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_experts
    )  # [num_layers, num_tokens, top_k, num_experts]
    # For a given token, determine if it was routed to a given expert. Think of this as a collection of top_k-hot vectors.
    expert_mask = torch.max(expert_mask, dim=-2).values.float()  # [num_layers, num_tokens, num_experts]
    tokens_per_layer_and_expert = torch.mean(expert_mask, dim=-2)  # [num_layers, num_experts]
    router_prob_per_layer_and_expert = torch.mean(routing_weights, dim=-2)  # [num_layers, num_experts]
    return torch.mean(tokens_per_layer_and_expert * router_prob_per_layer_and_expert) * num_experts**2


class MixtralOutputLayer(OutputLayer):
    def __init__(
        self,
        pipeline_model,
        loader_util,
        lm_head,
        load_balancing_loss_coef,
        num_experts,
        num_experts_per_tok,
        **kwargs,
    ):
        super().__init__(pipeline_model, loader_util, lm_head, **kwargs)
        self.load_balancing_loss_coef = load_balancing_loss_coef
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, inputs):
        hidden_states, labels, *router_logits = inputs
        router_logits = tuple(router_logits)
        outputs = super().forward((hidden_states, labels))
        if self.pipeline_model[0].sampling_mode:
            return outputs
        if self.load_balancing_loss_coef is not None:
            aux_loss = transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func(
                router_logits, self.num_experts, self.num_experts_per_tok
            )
            alternate_aux_loss = load_balancing_loss_func(router_logits, self.num_experts, self.num_experts_per_tok)
            loss = outputs[0]
            loss += self.load_balancing_loss_coef * aux_loss
            outputs = (loss, *outputs[1:], aux_loss, alternate_aux_loss)
        return outputs


class InputLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = [model]
        self.embed_tokens = model.model.embed_tokens
        if self.model.model.config.model_type == 'llama':
            self.rotary_emb = model.model.rotary_emb
        self.embedding_on_cpu = not self.model.train_config['full_fine_tune']
        self.model.loader_util.load_state_dict_into_module(self)

    @property
    def model(self):
        return self._model[0]

    def forward(self, inputs):
        past_key_values = None
        cache_position = None
        use_cache = self.model.sampling_mode

        input_ids, attention_mask, labels = inputs[:3]
        if self.model.sampling_mode:
            micro_batch_id = inputs[3].item()
            self.model.set_cache(micro_batch_id)
        device = input_ids.device
        if self.embedding_on_cpu:
            self.embed_tokens.to('cpu')
            input_ids = input_ids.to('cpu')

        inputs_embeds = self.embed_tokens(input_ids).to(device)
        if use_cache:
            past_key_values = self.model.cache

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)

        original_attention_mask = attention_mask
        if self.model.model.config.model_type == 'mistral':
            attention_mask = self.model.model._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, None
            )
        else:
            attention_mask = self.model.model._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, None
            )
        if attention_mask is None:
            # With FA, attention_mask can end up being None. But with deepspeed we can't pass None
            # between GPUs. So force it back to the original attention_mask.
            # TODO: with pipeline parallelism all tensors have to be the same shape between micro batches.
            # Can we handle None attention_mask better? Sending empty tensor and converting it later will not
            # work, due to static shape constraints. Maybe fill the original attention mask with some signal value?
            attention_mask = original_attention_mask
        hidden_states = inputs_embeds
        if self.model.model.config.model_type == 'gemma2':
            normalizer = torch.tensor(self.model.model.config.hidden_size**0.5, dtype=hidden_states.dtype)
            hidden_states = hidden_states * normalizer
        if self.model.model.config.model_type == 'llama':
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            position_ids = torch.tensor([], device=device)
        else:
            cos, sin = torch.tensor([], device=device), torch.tensor([], device=device)

        # Deepspeed requirement. Hidden states must require grad.
        hidden_states.requires_grad_(True)
        return hidden_states, attention_mask, position_ids, cos, sin, labels


class LlamaRMSNormPipe(nn.Module):
    def __init__(self, loader_util, orig):
        super().__init__()
        self.orig = orig
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, _, _, _, _, labels, *router_logits = inputs
        return self.orig(hidden_states), labels, *router_logits


class LlamaDecoderLayerPipe(nn.Module):
    def __init__(self, pipeline_model, loader_util, orig):
        super().__init__()
        self.pipeline_model = [pipeline_model]
        self.orig = orig
        self.mlp_offloaded_to_cpu = False
        loader_util.load_state_dict_into_module(self)

    # A note on MLP offloading:
    # We take advantage of how activation checkpointing works with reentrant checkpointing functions.
    # During the forward pass, if gradients are disabled (eval or first forward pass of activation checkpointing)
    # we offload the weights back to CPU at the end of the function. If gradients are enabled (second forward pass
    # of activation checkpointing) we leave the weights on GPU, and use a backward hook to offload to CPU after the
    # backward pass of this function is completed. This way the weights stay on the GPU for the backward pass.
    def forward(self, inputs):
        def move_mlp_to_cpu_hook(grad):
            self.move_mlp_to_cpu()
            return None

        hidden_states, input_attention_mask, position_ids, cos, sin, labels = inputs
        if self.mlp_offloaded_to_cpu:
            if hidden_states.requires_grad:
                hidden_states.register_hook(move_mlp_to_cpu_hook)
            self.move_mlp_to_device(hidden_states.device)
        kwargs = {}
        if input_attention_mask.numel() == 0:
            # We can't pass None between pipeline layers, so this signals that attention_mask should be None.
            kwargs['attention_mask'] = None
        else:
            kwargs['attention_mask'] = input_attention_mask
        if self.pipeline_model[0].model.config.model_type == 'llama':
            # Llama model uses precomputed RoPE embeddings.
            kwargs['position_embeddings'] = (cos, sin)
        else:
            # Everything else (unless this is gradually changing?) still uses position_ids.
            kwargs['position_ids'] = position_ids
        if self.pipeline_model[0].sampling_mode:
            kwargs['use_cache'] = True
            kwargs['past_key_value'] = self.pipeline_model[0].cache
        result = (self.orig(hidden_states, **kwargs)[0], input_attention_mask, position_ids, cos, sin, labels)
        if self.mlp_offloaded_to_cpu and not torch.is_grad_enabled():
            self.move_mlp_to_cpu()
        return result

    def move_mlp_to_cpu(self):
        # If it's already been moved to CPU once, just set the data to avoid a transfer.
        if self.mlp_offloaded_to_cpu:
            set_data(self.orig.mlp.up_proj, self.cpu_up_proj)
            set_data(self.orig.mlp.down_proj, self.cpu_down_proj)
            set_data(self.orig.mlp.gate_proj, self.cpu_gate_proj)
            return

        move_data_to_device(self.orig.mlp.up_proj, 'cpu')
        move_data_to_device(self.orig.mlp.down_proj, 'cpu')
        move_data_to_device(self.orig.mlp.gate_proj, 'cpu')
        self.mlp_offloaded_to_cpu = True

    def move_mlp_to_device(self, device):
        self.cpu_up_proj = move_data_to_device(self.orig.mlp.up_proj, device)
        self.cpu_down_proj = move_data_to_device(self.orig.mlp.down_proj, device)
        self.cpu_gate_proj = move_data_to_device(self.orig.mlp.gate_proj, device)


class Phi3DecoderLayerPipe(LlamaDecoderLayerPipe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def move_mlp_to_cpu(self):
        if self.mlp_offloaded_to_cpu:
            set_data(self.orig.mlp.gate_up_proj, self.cpu_gate_up_proj)
            set_data(self.orig.mlp.down_proj, self.cpu_down_proj)
            return

        move_data_to_device(self.orig.mlp.gate_up_proj, 'cpu')
        move_data_to_device(self.orig.mlp.down_proj, 'cpu')
        self.mlp_offloaded_to_cpu = True

    def move_mlp_to_device(self, device):
        self.cpu_gate_up_proj = move_data_to_device(self.orig.mlp.gate_up_proj, device)
        self.cpu_down_proj = move_data_to_device(self.orig.mlp.down_proj, device)


class MixtralDecoderLayerPipe(LlamaDecoderLayerPipe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_experts_to_offload = self.pipeline_model[0].num_experts_to_offload

    def forward(self, inputs):
        def move_mlp_to_cpu_hook(grad):
            self.move_mlp_to_cpu()
            return None

        hidden_states, input_attention_mask, position_ids, cos, sin, labels, *input_router_logits = inputs
        if self.mlp_offloaded_to_cpu:
            if hidden_states.requires_grad:
                hidden_states.register_hook(move_mlp_to_cpu_hook)
            self.move_mlp_to_device(hidden_states.device)
        kwargs = {}
        if input_attention_mask.numel() == 0:
            # We can't pass None between pipeline layers, so this signals that attention_mask should be None.
            kwargs['attention_mask'] = None
        else:
            kwargs['attention_mask'] = input_attention_mask
        if self.pipeline_model[0].model.config.model_type == 'llama':
            # Llama model uses precomputed RoPE embeddings.
            kwargs['position_embeddings'] = (cos, sin)
        else:
            # Everything else (unless this is gradually changing?) still uses position_ids.
            kwargs['position_ids'] = position_ids
        if self.pipeline_model[0].sampling_mode:
            kwargs['use_cache'] = True
            kwargs['past_key_value'] = self.pipeline_model[0].cache
        hidden_states, router_logits = self.orig(hidden_states, output_router_logits=True, **kwargs)
        # TODO: fix unsloth gradient checkpointing when we return router logits
        # router_logits = router_logits.to(torch.float32)
        # router_logits = input_router_logits + (router_logits,)
        # result = (hidden_states, input_attention_mask, position_ids, cos, sin, labels, *router_logits)
        result = (hidden_states, input_attention_mask, position_ids, cos, sin, labels)
        if self.mlp_offloaded_to_cpu and not torch.is_grad_enabled():
            self.move_mlp_to_cpu()
        return result

    def move_mlp_to_cpu(self):
        if self.mlp_offloaded_to_cpu:
            set_experts_data(self.orig.block_sparse_moe.experts, self.orig_data)
            return

        move_experts_to_device(self.orig.block_sparse_moe.experts, 'cpu', self.num_experts_to_offload)
        self.mlp_offloaded_to_cpu = True

    def move_mlp_to_device(self, device):
        self.orig_data = move_experts_to_device(
            self.orig.block_sparse_moe.experts, device, self.num_experts_to_offload
        )
