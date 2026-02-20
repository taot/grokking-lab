import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class AttentionOutput:
    """Attention layer output with optional intermediate values."""

    output: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None  # (batch, h, seq, seq)


@dataclass
class DecoderBlockOutput:
    """Decoder block output with optional intermediate values."""

    output: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None  # (batch, h, seq, seq)
    mlp_activations: Optional[torch.Tensor] = None  # (batch, seq, 4*d)


@dataclass
class TransformerOutput:
    """Transformer output with optional intermediate values."""

    logits: torch.Tensor
    attention_weights: Optional[list[torch.Tensor]] = None  # per layer
    mlp_activations: Optional[list[torch.Tensor]] = None  # per layer


def assert_dimension(x: torch.Tensor, dim: tuple[int, ...]) -> None:
    assert x.shape == dim


class Attention(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()

        self.d = d
        self.h = h
        assert self.d % self.h == 0
        self.d_head = self.d // self.h

        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_proj = nn.Linear(d, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor | AttentionOutput:
        batch_size, seq_len = x.shape[:2]
        assert_dimension(x, (batch_size, seq_len, self.d))

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, seq_len, self.h, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.h, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.h, self.d_head).transpose(1, 2)
        assert_dimension(q, (batch_size, self.h, seq_len, self.d_head))
        assert_dimension(k, (batch_size, self.h, seq_len, self.d_head))
        assert_dimension(v, (batch_size, self.h, seq_len, self.d_head))

        weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        assert_dimension(weights, (batch_size, self.h, seq_len, seq_len))

        causal = torch.ones(
            (seq_len, seq_len), device=weights.device, dtype=torch.bool
        ).tril()
        attn_mask = causal[None, None, :, :]

        if mask is not None:
            mask = mask.to(device=weights.device)
            if mask.dtype != torch.bool:
                mask = mask.to(dtype=torch.bool)
            if mask.dim() == 2:
                mask = mask[None, None, :, :]
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]
            elif mask.dim() != 4:
                raise ValueError(f"Invalid mask shape: {tuple(mask.shape)}")
            attn_mask = attn_mask & mask

        weights = weights.masked_fill(~attn_mask, torch.finfo(weights.dtype).min)

        weights_softmax = torch.softmax(weights, dim=-1)
        attn = torch.matmul(weights_softmax, v)
        assert_dimension(attn, (batch_size, self.h, seq_len, self.d_head))

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d)
        out = self.W_proj(attn)

        if return_weights:
            return AttentionOutput(output=out, attention_weights=weights_softmax)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()

        self.attention = Attention(d=d, h=h)

        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)
        self.relu = nn.ReLU()

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False
    ) -> torch.Tensor | DecoderBlockOutput:
        if return_intermediates:
            attn_result = self.attention(x, mask=None, return_weights=True)
            attn_out = attn_result.output
            attn_weights = attn_result.attention_weights
        else:
            attn_out = self.attention(x, mask=None, return_weights=False)
            attn_weights = None

        x = x + attn_out

        mlp_pre = self.ff1(x)
        mlp_act = self.relu(mlp_pre)
        ff_out = self.ff2(mlp_act)
        x = x + ff_out

        if return_intermediates:
            return DecoderBlockOutput(
                output=x,
                attention_weights=attn_weights,
                mlp_activations=mlp_act,
            )
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        output_size: int,
        d: int,
        n_layers: int,
        h: int,
        max_seq_len: int = 3,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.output_size = output_size
        self.d = d
        self.h = h

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d)
        self.pos_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=d)

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d=d, h=h) for _ in range(n_layers)]
        )
        self.linear = nn.Linear(d, self.output_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, return_intermediates: bool = False
    ) -> torch.Tensor | TransformerOutput:
        seq_len = input_ids.shape[-1]
        pos_ids = torch.arange(seq_len, device=input_ids.device)

        token_emb = self.embedding(input_ids)
        pos_emb = self.pos_embedding(pos_ids).unsqueeze(0)
        x = token_emb + pos_emb

        all_attn_weights: list[torch.Tensor] = []
        all_mlp_activations: list[torch.Tensor] = []

        for decoder in self.decoder_blocks:
            if return_intermediates:
                result = decoder(x, return_intermediates=True)
                x = result.output
                if result.attention_weights is not None:
                    all_attn_weights.append(result.attention_weights)
                if result.mlp_activations is not None:
                    all_mlp_activations.append(result.mlp_activations)
            else:
                x = decoder(x)

        logits = self.linear(x)
        final_logits = logits[:, -1, :]

        if return_intermediates:
            return TransformerOutput(
                logits=final_logits,
                attention_weights=all_attn_weights if all_attn_weights else None,
                mlp_activations=all_mlp_activations if all_mlp_activations else None,
            )
        return final_logits
