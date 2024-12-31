import torch
from torch import nn
import transformers
import accelerate

from pipeline_model import OutputLayer, LayerSpec, PipelineModel, move_data_to_device, set_data
from utils import DTYPE_MAP

class EmbeddingPipe(nn.Module):
    def __init__(self, loader_util, orig, model, embedding_on_cpu=False):
        super().__init__()
        self.orig = orig
        # The original model object, e.g. LlamaModel. Use a list so the nn.Module isn't registered to this module.
        self.model = [model]
        self.embedding_on_cpu = embedding_on_cpu
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        input_ids, attention_mask, position_ids, labels = inputs
        original_device = input_ids.device
        if self.embedding_on_cpu:
            self.orig.to('cpu')
            input_ids = input_ids.to('cpu')
        inputs_embeds = self.orig(input_ids).to(original_device)

        original_attention_mask = attention_mask
        past_key_values = None  # always None for training
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if self.model[0].config.model_type == 'mistral':
            attention_mask = self.model[0]._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, False, False
            )
        else:
            attention_mask = self.model[0]._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, False
            )
        if attention_mask is None:
            # With FA, attention_mask can end up being None. But with deepspeed we can't pass None
            # between GPUs. So force it back to the original attention_mask.
            attention_mask = original_attention_mask

        hidden_states = inputs_embeds
        if self.model[0].config.model_type == 'gemma2':
            normalizer = torch.tensor(self.model[0].config.hidden_size**0.5, dtype=hidden_states.dtype)
            hidden_states = hidden_states * normalizer

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


class LlamaDecoderLayerPipe(nn.Module):
    def __init__(self, loader_util, orig):
        super().__init__()
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
        def set_cpu_data():
            set_data(self.orig.mlp.up_proj, cpu_up_proj)
            set_data(self.orig.mlp.down_proj, cpu_down_proj)
            set_data(self.orig.mlp.gate_proj, cpu_gate_proj)
        def set_cpu_data_hook(grad):
            set_cpu_data()
            return None

        hidden_states, attention_mask, position_ids, labels = inputs
        if self.mlp_offloaded_to_cpu:
            if hidden_states.requires_grad:
                hidden_states.register_hook(set_cpu_data_hook)
            cpu_up_proj = move_data_to_device(self.orig.mlp.up_proj, hidden_states.device)
            cpu_down_proj = move_data_to_device(self.orig.mlp.down_proj, hidden_states.device)
            cpu_gate_proj = move_data_to_device(self.orig.mlp.gate_proj, hidden_states.device)
        result = (self.orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0], attention_mask, position_ids, labels)
        if self.mlp_offloaded_to_cpu and not torch.is_grad_enabled():
            set_cpu_data()
        return result

    def offload_mlp_to_cpu(self):
        self.mlp_offloaded_to_cpu = True
        move_data_to_device(self.orig.mlp.up_proj, 'cpu')
        move_data_to_device(self.orig.mlp.down_proj, 'cpu')
        move_data_to_device(self.orig.mlp.gate_proj, 'cpu')


class Phi3DecoderLayerPipe(nn.Module):
    def __init__(self, loader_util, orig):
        super().__init__()
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
        def set_cpu_data():
            set_data(self.orig.mlp.gate_up_proj, cpu_up_proj)
            set_data(self.orig.mlp.down_proj, cpu_down_proj)
        def set_cpu_data_hook(grad):
            set_cpu_data()
            return None

        hidden_states, attention_mask, position_ids, labels = inputs
        if self.mlp_offloaded_to_cpu:
            if hidden_states.requires_grad:
                hidden_states.register_hook(set_cpu_data_hook)
            cpu_up_proj = move_data_to_device(self.orig.mlp.gate_up_proj, hidden_states.device)
            cpu_down_proj = move_data_to_device(self.orig.mlp.down_proj, hidden_states.device)
        result = (self.orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0], attention_mask, position_ids, labels)
        if self.mlp_offloaded_to_cpu and not torch.is_grad_enabled():
            set_cpu_data()
        return result

    def offload_mlp_to_cpu(self):
        self.mlp_offloaded_to_cpu = True
        move_data_to_device(self.orig.mlp.gate_up_proj, 'cpu')
        move_data_to_device(self.orig.mlp.down_proj, 'cpu')


# A little bit of inheritance and MRO trickery since LlamaForCausalLM.__init__ only takes a
# positional argument. We inherit PipelineModel first, but call LlamaForCausalLM init first,
# and make sure PipelineModel doesn't have a super().__init__() call.
class LlamaForCausalLMPipe(PipelineModel, transformers.LlamaForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.LlamaConfig.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.LlamaForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
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
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=not self.train_config['full_fine_tune']
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(
            LayerSpec(
                OutputLayer,
                self,
                self.loader_util,
                self.lm_head,
                loss_type=self.loss_type,
                focal_loss_gamma=self.focal_loss_gamma,
                tie_weights='model.embed_tokens.weight' if self.config.tie_word_embeddings else None,
            )
        )
        return result


class Qwen2ForCausalLMPipe(PipelineModel, transformers.Qwen2ForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.Qwen2Config.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.Qwen2ForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
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
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=not self.train_config['full_fine_tune']
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(
            LayerSpec(
                OutputLayer,
                self,
                self.loader_util,
                self.lm_head,
                loss_type=self.loss_type,
                focal_loss_gamma=self.focal_loss_gamma,
                tie_weights='model.embed_tokens.weight' if self.config.tie_word_embeddings else None,
            )
        )
        return result

class CohereForCausalLMPipe(PipelineModel, transformers.CohereForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.CohereConfig.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.CohereForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        # the embedding table for this model is huge; load balance it better with some heuristics
        embedding_relative_size = 4

        def initial_layer(inputs):
            input_ids, attention_mask, labels = inputs
            batch_size, seq_length = input_ids.shape[:2]
            device = input_ids.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
            return input_ids, attention_mask, position_ids, labels

        embedding_on_cpu = not self.train_config['full_fine_tune']
        result = [
            initial_layer,
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=embedding_on_cpu,
                _estimated_size=1 if embedding_on_cpu else embedding_relative_size,
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(
            LayerSpec(
                OutputLayer,
                self,
                self.loader_util,
                self.lm_head,
                logit_scale=self.logit_scale,
                loss_type=self.loss_type,
                focal_loss_gamma=self.focal_loss_gamma,
                tie_weights='model.embed_tokens.weight' if self.config.tie_word_embeddings else None,
                _estimated_size=embedding_relative_size,
            )
        )
        return result


class Phi3ForCausalLMPipe(PipelineModel, transformers.Phi3ForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.Phi3Config.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.Phi3ForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
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
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=not self.train_config['full_fine_tune']
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(Phi3DecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(
            LayerSpec(
                OutputLayer,
                self,
                self.loader_util,
                self.lm_head,
                loss_type=self.loss_type,
                focal_loss_gamma=self.focal_loss_gamma,
            )
        )
        return result

class Gemma2ForCausalLMPipe(PipelineModel, transformers.Gemma2ForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.Gemma2Config.from_pretrained(config['model'])
        # TODO: change this when Gemma works with other attn implementations
        model_config._attn_implementation = 'eager'
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.Gemma2ForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        # the embedding table for this model is huge; load balance it better with some heuristics
        # this value optimized for LoRA, pipeline_stages=2
        embedding_relative_size = 8

        def initial_layer(inputs):
            input_ids, attention_mask, labels = inputs
            batch_size, seq_length = input_ids.shape[:2]
            device = input_ids.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
            return input_ids, attention_mask, position_ids, labels

        embedding_on_cpu = not self.train_config['full_fine_tune']
        result = [
            initial_layer,
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=embedding_on_cpu,
                _estimated_size=1 if embedding_on_cpu else embedding_relative_size,
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(
            LayerSpec(
                OutputLayer,
                self,
                self.loader_util,
                self.lm_head,
                loss_type=self.loss_type,
                focal_loss_gamma=self.focal_loss_gamma,
                tie_weights='model.embed_tokens.weight' if self.config.tie_word_embeddings else None,
                logit_softcapping=self.config.final_logit_softcapping,
                _estimated_size=embedding_relative_size,
            )
        )
        return result


class MistralForCausalLMPipe(PipelineModel, transformers.MistralForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.MistralConfig.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.MistralForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
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
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=not self.train_config['full_fine_tune']
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(
            LayerSpec(
                OutputLayer,
                self,
                self.loader_util,
                self.lm_head,
                loss_type=self.loss_type,
                focal_loss_gamma=self.focal_loss_gamma,
            )
        )
        return result
