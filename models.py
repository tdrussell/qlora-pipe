import accelerate
import torch
import transformers

from layers import (
    InputLayer,
    LayerSpec,
    LlamaDecoderLayerPipe,
    LlamaRMSNormPipe,
    MixtralDecoderLayerPipe,
    MixtralOutputLayer,
    OutputLayer,
    Phi3DecoderLayerPipe,
    Gemma3InputLayer,
    Gemma3DecoderLayerPipe,
    Gemma3RMSNormPipe,
)
from pipeline_model import PipelineModel
from utils import DTYPE_MAP

DEFAULT_ATTN_IMPLEMENTATION = 'flash_attention_2'


# A little bit of inheritance and MRO trickery since LlamaForCausalLM.__init__ only takes a
# positional argument. We inherit PipelineModel first, but call LlamaForCausalLM init first,
# and make sure PipelineModel doesn't have a super().__init__() call.
class LlamaForCausalLMPipe(PipelineModel, transformers.LlamaForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.LlamaConfig.from_pretrained(config['model'])
        model_config._attn_implementation = config.get('attn_implementation', DEFAULT_ATTN_IMPLEMENTATION)
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.LlamaForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        embedding_relative_size = 4
        embedding_on_cpu = not self.train_config['full_fine_tune']
        result = [LayerSpec(InputLayer, self, _estimated_size=0 if embedding_on_cpu else embedding_relative_size)]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self, self.loader_util, block))
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
                _estimated_size=embedding_relative_size,
            )
        )
        return result


class Qwen2ForCausalLMPipe(PipelineModel, transformers.Qwen2ForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.Qwen2Config.from_pretrained(config['model'])
        model_config._attn_implementation = config.get('attn_implementation', DEFAULT_ATTN_IMPLEMENTATION)
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.Qwen2ForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        result = [LayerSpec(InputLayer, self)]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self, self.loader_util, block))
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
        model_config._attn_implementation = config.get('attn_implementation', DEFAULT_ATTN_IMPLEMENTATION)
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.CohereForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        # the embedding table for this model is huge; load balance it better with some heuristics
        embedding_relative_size = 4
        embedding_on_cpu = not self.train_config['full_fine_tune']
        result = [LayerSpec(InputLayer, self, _estimated_size=1 if embedding_on_cpu else embedding_relative_size)]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self, self.loader_util, block))
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
        model_config._attn_implementation = config.get('attn_implementation', DEFAULT_ATTN_IMPLEMENTATION)
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.Phi3ForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        result = [LayerSpec(InputLayer, self)]
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
        model_config._attn_implementation = config.get('attn_implementation', DEFAULT_ATTN_IMPLEMENTATION)
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.Gemma2ForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        # the embedding table for this model is huge; load balance it better with some heuristics
        # this value optimized for LoRA, pipeline_stages=2
        embedding_relative_size = 8
        embedding_on_cpu = not self.train_config['full_fine_tune']
        result = [LayerSpec(InputLayer, self, _estimated_size=1 if embedding_on_cpu else embedding_relative_size)]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self, self.loader_util, block))
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
        model_config._attn_implementation = config.get('attn_implementation', DEFAULT_ATTN_IMPLEMENTATION)
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.MistralForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        result = [LayerSpec(InputLayer, self)]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self, self.loader_util, block))
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


class MixtralForCausalLMPipe(PipelineModel, transformers.MixtralForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.MixtralConfig.from_pretrained(config['model'])
        model_config._attn_implementation = config.get('attn_implementation', DEFAULT_ATTN_IMPLEMENTATION)
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.MixtralForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)
        self.load_balancing_loss_coef = config.get('load_balancing_loss_coef', None)
        self.num_experts_to_offload = self.num_experts
        if 'offload_mlp_to_cpu' in config and isinstance(config['offload_mlp_to_cpu'], int):
            self.num_experts_to_offload = config['offload_mlp_to_cpu']

    def to_layer_specs(self):
        result = [LayerSpec(InputLayer, self)]
        for block in self.model.layers:
            result.append(LayerSpec(MixtralDecoderLayerPipe, self, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(
            LayerSpec(
                MixtralOutputLayer,
                self,
                self.loader_util,
                self.lm_head,
                load_balancing_loss_coef=self.load_balancing_loss_coef,
                num_experts=self.num_experts,
                num_experts_per_tok=self.num_experts_per_tok,
                loss_type=self.loss_type,
                focal_loss_gamma=self.focal_loss_gamma,
            )
        )
        return result

class Gemma3ForCausalLMPipe(PipelineModel, transformers.Gemma3ForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.Gemma3TextConfig.from_pretrained(config['model'])
        model_config._attn_implementation = config.get('attn_implementation', DEFAULT_ATTN_IMPLEMENTATION)
        torch.set_default_dtype(DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')])
        with accelerate.init_empty_weights():
            transformers.Gemma3ForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        # the embedding table for this model is huge; load balance it better with some heuristics
        # this value optimized for LoRA, pipeline_stages=2
        embedding_relative_size = 8
        embedding_on_cpu = not self.train_config['full_fine_tune']
        result = [LayerSpec(Gemma3InputLayer, self, _estimated_size=1 if embedding_on_cpu else embedding_relative_size)]
        for block in self.model.layers:
            result.append(LayerSpec(Gemma3DecoderLayerPipe, self, self.loader_util, block))
        result.append(LayerSpec(Gemma3RMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
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