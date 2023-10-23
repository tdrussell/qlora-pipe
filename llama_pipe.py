import torch
from torch import nn
import transformers


class EmbeddingPipe(nn.Module):
    def __init__(self, orig, prepare_decoder_attention_mask):
        super().__init__()
        self.orig = orig
        self._prepare_decoder_attention_mask = prepare_decoder_attention_mask
    
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
    def __init__(self, orig):
        super().__init__()
        self.orig = orig

    def forward(self, inputs):
        hidden_states, _, _, labels = inputs
        return self.orig(hidden_states), labels
    

class LmHeadPipe(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.orig = orig

    def forward(self, inputs):
        hidden_states, labels = inputs
        return self.orig(hidden_states), labels


class LlamaDecoderLayerPipe(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.orig = orig
        self.offload_mlp_to_cpu = False
    
    def forward(self, inputs):
        hidden_states, attention_mask, position_ids, labels = inputs
        if self.offload_mlp_to_cpu:
            cpu_up_proj = self.orig.mlp.up_proj.weight.data
            cpu_down_proj = self.orig.mlp.down_proj.weight.data
            cpu_gate_proj = self.orig.mlp.gate_proj.weight.data
            self.orig.mlp.up_proj.weight.data = cpu_up_proj.to(hidden_states.device, non_blocking=True)
            self.orig.mlp.down_proj.weight.data = cpu_down_proj.to(hidden_states.device, non_blocking=True)
            self.orig.mlp.gate_proj.weight.data = cpu_gate_proj.to(hidden_states.device, non_blocking=True)
        result = (self.orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0], attention_mask, position_ids, labels)
        if self.offload_mlp_to_cpu:
            self.orig.mlp.up_proj.weight.data = cpu_up_proj
            self.orig.mlp.down_proj.weight.data = cpu_down_proj
            self.orig.mlp.gate_proj.weight.data = cpu_gate_proj
        return result


class LlamaForCausalLMPipe(transformers.LlamaForCausalLM):
    def compute_loss(self, inputs):
        logits, labels = inputs
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = transformers.models.llama.modeling_llama.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

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
        result.append(self.compute_loss)
        self.layer_list = result
        return result
    
    def offload_mlp_to_cpu(self):
        for layer in self.layer_list:
            if type(layer) != LlamaDecoderLayerPipe:
                continue
            layer.offload_mlp_to_cpu = True
            layer.orig.mlp.up_proj.weight.data = layer.orig.mlp.up_proj.weight.data.to('cpu')
            layer.orig.mlp.down_proj.weight.data = layer.orig.mlp.down_proj.weight.data.to('cpu')
            layer.orig.mlp.gate_proj.weight.data = layer.orig.mlp.gate_proj.weight.data.to('cpu')
        torch.cuda.empty_cache()