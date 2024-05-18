import torch
from torch import nn
import transformers
import accelerate

from pipeline_model import ComputeMetrics, LayerSpec, PipelineModel, move_data_to_device, set_data
from utils import DTYPE_MAP

class EmbeddingPipe(nn.Module):
    def __init__(self, loader_util, orig, attn_implementation, embedding_on_cpu=False):
        super().__init__()
        self.orig = orig
        self.attn_implementation = attn_implementation
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
    def __init__(self, loader_util, orig, logit_scale=1.0, tie_weights=None):
        super().__init__()
        self.orig = orig
        self.logit_scale = logit_scale
        if tie_weights:
            self.orig.weight.original_name = tie_weights
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, labels = inputs
        return self.orig(hidden_states*self.logit_scale), labels


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
    def __init__(self, config, quantization_config, **kwargs):
        model_config = transformers.LlamaConfig.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.LlamaForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, **kwargs)
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
                self.model.config._attn_implementation,
                embedding_on_cpu=not self.train_config['full_fine_tune']
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(LayerSpec(LmHeadPipe, self.loader_util, self.lm_head))
        result.append(LayerSpec(ComputeMetrics, focal_loss_gamma=self.focal_loss_gamma))
        return result


class Qwen2ForCausalLMPipe(PipelineModel, transformers.Qwen2ForCausalLM):
    def __init__(self, config, quantization_config, **kwargs):
        model_config = transformers.Qwen2Config.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.Qwen2ForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, **kwargs)
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
                self.model.config._attn_implementation,
                embedding_on_cpu=not self.train_config['full_fine_tune']
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(LayerSpec(LmHeadPipe, self.loader_util, self.lm_head))
        result.append(LayerSpec(ComputeMetrics, focal_loss_gamma=self.focal_loss_gamma))
        return result

class CohereForCausalLMPipe(PipelineModel, transformers.CohereForCausalLM):
    def __init__(self, config, quantization_config, **kwargs):
        model_config = transformers.CohereConfig.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.CohereForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, **kwargs)
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
                self.model.config._attn_implementation,
                embedding_on_cpu=embedding_on_cpu,
                _estimated_size=1 if embedding_on_cpu else embedding_relative_size,
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(LayerSpec(
            LmHeadPipe,
            self.loader_util,
            self.lm_head,
            logit_scale=self.logit_scale,
            tie_weights='model.embed_tokens.weight',
            _estimated_size=embedding_relative_size
        ))
        result.append(LayerSpec(ComputeMetrics, focal_loss_gamma=self.focal_loss_gamma))
        return result
