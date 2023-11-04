import torch
from torch import nn
import torch.nn.functional as F
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


def entropy_fn(logits):
    result = 0.
    # There is a very wide range of chuck sizes that cause no increase in memory reported by
    # nvidia-smi (Torch re-using blocks of memory?). If you try to compute it as one tensor,
    # memory usage is huge. Chuck size of 128 seems good enough for now.
    for logits_chuck in torch.split(logits, 128):
        result += torch.distributions.Categorical(logits=logits_chuck).entropy().sum()
    return result / logits.size(0)


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

        loss_fct = transformers.models.llama.modeling_llama.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)
        with torch.no_grad():
            accuracies = top_k_accuracy(shift_logits, shift_labels, k_list=[1, 5, 20])
            entropy = entropy_fn(shift_logits)
        return loss, entropy, *accuracies

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
    
    def offload_mlp_to_cpu(self):
        for layer in self.layer_list:
            if type(layer) != LlamaDecoderLayerPipe:
                continue
            layer.offload_mlp_to_cpu = True
            layer.orig.mlp.up_proj.weight.data = layer.orig.mlp.up_proj.weight.data.to('cpu')
            layer.orig.mlp.down_proj.weight.data = layer.orig.mlp.down_proj.weight.data.to('cpu')
            layer.orig.mlp.gate_proj.weight.data = layer.orig.mlp.gate_proj.weight.data.to('cpu')
        torch.cuda.empty_cache()