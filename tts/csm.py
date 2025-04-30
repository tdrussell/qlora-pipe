from dataclasses import dataclass
import json
from pathlib import Path
import math

import torch
from torch import nn
import torchtune
from torchtune.models import llama3_2
import transformers
import safetensors
import accelerate
from tokenizers.processors import TemplateProcessing

from kernels.cross_entropy_loss import Fast_CrossEntropyLoss
from utils.utils import DTYPE_MAP


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


def compute_metrics(logits, labels, h):
    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = labels.view(-1)
    flat_loss_mask = (flat_labels >= 0)
    #cross_entropy_loss = Fast_CrossEntropyLoss.apply(flat_logits, flat_labels)
    cross_entropy_loss = torch.nn.functional.cross_entropy(flat_logits, flat_labels, reduction='none')
    cross_entropy_loss = cross_entropy_loss[flat_loss_mask]
    loss_unreduced = cross_entropy_loss
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
        hidden_state_norms = torch.norm(h.float(), dim=-1)
        hidden_state_norms = hidden_state_norms.view(-1)[flat_loss_mask]
    loss = loss_unreduced.mean()
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


def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
}


def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    r = mask[input_pos, :]
    return r


@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int


class CSM(nn.Module):
    def __init__(self, config, quantization_config):
        dtype = DTYPE_MAP[config.get('model_weight_dtype', 'bfloat16')]
        torch.set_default_dtype(dtype)
        super().__init__()
        model_path = Path(config['model']['path'])
        with open(model_path / 'config.json') as f:
            # Can't use attribute 'config' or PEFT breaks.
            self.custom_config = ModelArgs(**json.load(f))

        with accelerate.init_empty_weights():
            self.backbone, backbone_dim = _prepare_transformer(FLAVORS[self.custom_config.backbone_flavor]())
            self.decoder, decoder_dim = _prepare_transformer(FLAVORS[self.custom_config.decoder_flavor]())

            self.text_embeddings = nn.Embedding(self.custom_config.text_vocab_size, backbone_dim)
            self.audio_embeddings = nn.Embedding(self.custom_config.audio_vocab_size * self.custom_config.audio_num_codebooks, backbone_dim)

            self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
            self.codebook0_head = nn.Linear(backbone_dim, self.custom_config.audio_vocab_size, bias=False)
            self.audio_head = nn.Parameter(torch.empty(self.custom_config.audio_num_codebooks - 1, decoder_dim, self.custom_config.audio_vocab_size))

        state_dict = {}
        with safetensors.safe_open(model_path / 'model.safetensors', framework='pt', device='cpu') as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key).to(dtype)
        self.load_state_dict(state_dict, assign=True)

        device = next(self.parameters()).device
        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device))
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.custom_config.audio_num_codebooks, device))
        torch.set_default_dtype(torch.float32)

    @classmethod
    def get_tokenizer(cls, model_config: str | dict):
        # Training currently requires padding_side='right' because it uses default causal masking.
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_config['llama3_path'], local_files_only=True, model_max_length=int(1e30), padding_side='right'
        )
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
        )
        return tokenizer

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.custom_config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.custom_config.audio_vocab_size * torch.arange(self.custom_config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.custom_config.audio_num_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds], dim=-2)

    def to_layer_specs(self):
        layers = [InputLayer(self)]
        for layer in self.backbone.layers:
            layers.append(DecoderLayer(layer))
        layers.append(RMSNorm(self.backbone.norm))
        layers.append(OutputLayer(self))
        return layers


class InputLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_embeddings = model.text_embeddings
        self.audio_embeddings = model.audio_embeddings
        self.register_buffer('backbone_causal_mask', model.backbone_causal_mask)
        self.custom_config = model.custom_config

    def forward(self, inputs):
        tokens, tokens_mask, labels = inputs[:3]
        # handle padding tokens
        tokens = tokens * tokens_mask

        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)
        audio_tokens = tokens[:, :, :-1] + (
            self.custom_config.audio_vocab_size * torch.arange(self.custom_config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.custom_config.audio_num_codebooks, -1
        )
        embeds = torch.cat([audio_embeds, text_embeds], dim=-2)

        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h.requires_grad_(True)

        # bs, seq_len, _ = h.shape
        # input_pos = torch.arange(seq_len).repeat(bs, 1).long().to(tokens.device)
        # backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)[:, :, :seq_len]

        return h, audio_tokens, labels


class DecoderLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, inputs):
        h, audio_tokens, labels = inputs
        h = self.layer(h)
        return h, audio_tokens, labels


class RMSNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, inputs):
        h, audio_tokens, labels = inputs
        return self.norm(h), audio_tokens, labels


class OutputLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.projection = model.projection
        self.codebook0_head = model.codebook0_head
        self.decoder = model.decoder
        self.audio_head = model.audio_head
        self.audio_embeddings = model.audio_embeddings
        self.projection = model.projection
        self.custom_config = model.custom_config

    def forward(self, inputs):
        h, audio_tokens, labels = inputs
        extra_ignored_labels = torch.full((labels.shape[0], 1, labels.shape[-1]), -100, device=h.device)
        labels = torch.hstack((labels[..., 1:, :], extra_ignored_labels))

        c0_logits = self.codebook0_head(h).to(torch.float32)
        c0_labels = labels[:, :, 0]
        c0_metrics = compute_metrics(c0_logits, c0_labels, h)

        audio_select = torch.where(c0_labels.flatten() >= 0)
        audio_select = audio_select[torch.randperm(len(audio_select))]
        sample_size = len(audio_select) // 8
        audio_select = audio_select[:sample_size]

        h = h.view(-1, h.size(-1))
        h = h[audio_select]  # [num_items, dim]
        # Shift and pad to align with labels. audio_select comes from flattened labels.
        audio_tokens = torch.hstack(
            (audio_tokens[:, 1:, :], torch.zeros((audio_tokens.size(0), 1, audio_tokens.size(-1)), device=audio_tokens.device, dtype=audio_tokens.dtype))
        )
        assert audio_tokens.size(1) == labels.size(1)
        audio_tokens = audio_tokens.view(-1, audio_tokens.size(-1))
        audio_tokens = audio_tokens[audio_select]
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            audio_tokens.size(0), self.custom_config.audio_num_codebooks, -1
        )  # [num_items, num_codebooks, dim]
        h = h.unsqueeze(1)
        h = torch.cat([h, audio_embeds], dim=1)
        h = self.decoder(self.projection(h)).to(h.dtype)  # [num_items, num_codebooks+1, dim]
        h = torch.permute(h, (1, 0, 2))
        # Index 0 is backbone hidden state. Final index has nothing to predict.
        h = h[1:-1, ...]
        ci_logits = torch.bmm(h, self.audio_head).to(torch.float32)  # [num_codebooks-1, num_items, num_logits]
        labels = labels.view(-1, labels.size(-1))
        labels = labels[audio_select]
        labels = labels[:, 1:-1].t().contiguous()  # [num_codebooks-1, num_items]
        ci_metrics = compute_metrics(ci_logits, labels, h)

        # TODO: make the weighting configurable?
        loss = 0.3 * c0_metrics[0] + ci_metrics[0]
        return (loss, *c0_metrics[1:])
