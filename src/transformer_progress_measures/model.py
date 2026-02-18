import math
from typing import Optional

import torch
from torch import nn


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
        self.W_proj = nn.Linear(d, d)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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
        return out


class DecoderBlock(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d)
        self.attention = Attention(d=d, h=h)

        self.layer_norm2 = nn.LayerNorm(d)

        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(self.layer_norm1(x), mask=None)
        x = x + attn_out

        ff_out = self.ff2(self.relu(self.ff1(self.layer_norm2(x))))
        x = x + ff_out
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d: int,
        n_layers: int,
        h: int,
        max_seq_len: int = 3,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d = d

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d)
        self.pos_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=d)

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d=d, h=h) for _ in range(n_layers)]
        )
        self.linear = nn.Linear(d, self.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[-1]
        pos_ids = torch.arange(seq_len, device=input_ids.device)

        token_emb = self.embedding(input_ids)
        pos_emb = self.pos_embedding(pos_ids).unsqueeze(0)
        x = token_emb + pos_emb

        for decoder in self.decoder_blocks:
            x = decoder(x)

        logits = self.linear(x)
        return logits[:, -1, :]
