import math
from typing import Optional, cast

import torch
from torch import nn


def assert_dimension(x: torch.Tensor, dim: tuple[int, ...]) -> None:
    assert x.shape == dim


# def softmax(x: torch.Tensor) -> torch.Tensor:
#     exp_x = torch.exp(x)
#     sum_x = torch.sum(exp_x, dim=-1, keepdim=True)
#     return exp_x / sum_x


# def relu(x: torch.Tensor) -> torch.Tensor:
#     x[x < 0] = 0
#     return x


# def xavier_init(fan_in: int, fan_out: int):
#     std = math.sqrt(2.0 / (fan_in + fan_out))
#     return torch.randn(fan_in, fan_out) * std


class Attention(nn.Module):
    # TODO Causal mask
    # TODO Test

    def __init__(self, d: int, h: int) -> None:
        super().__init__()

        self.d = d
        self.h = h
        assert self.d % self.h == 0
        self.d_head = self.d // self.h

        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)

        # self.W_q = xavier_init(d, d_k)
        # self.W_k = xavier_init(d, d_k)
        # self.W_v = xavier_init(d, d_v)

        # projection
        # self.W_proj = xavier_init(d_v, d)
        # self.b_proj = torch.zeros(d)

        self.W_proj = nn.Linear(d, d)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        assert_dimension(x, (batch_size, seq_len, self.d))

        # q = torch.matmul(x, self.W_q)       # batch_size * seq_len * d_k
        # k = torch.matmul(x, self.W_k)       # batch_size * seq_len * d_k
        # v = torch.matmul(x, self.W_v)       # batch_size * seq_len * d_v

        q = self.W_q(x)  # [B, T, d]
        k = self.W_k(x)  # [B, T, d]
        v = self.W_v(x)  # [B, T, d]

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
            elif mask.dim() == 4:
                pass
            else:
                raise ValueError(f"Invalid mask shape: {tuple(mask.shape)}")
            attn_mask = attn_mask & mask

        weights = weights.masked_fill(~attn_mask, torch.finfo(weights.dtype).min)

        weights_softmax = torch.softmax(weights, dim=-1)
        attn = torch.matmul(weights_softmax, v)
        assert_dimension(attn, (batch_size, self.h, seq_len, self.d_head))

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d)
        out = self.W_proj(attn)

        return out


def positional_encoding(max_seq_len: int, d: int, N: int = 100000) -> torch.Tensor:
    pe = torch.zeros(max_seq_len, d)
    for p in range(max_seq_len):
        pe[p, ::2] = torch.sin(p / (N ** (torch.Tensor(range(d // 2)) * 2 / d)))
        pe[p, 1::2] = torch.cos(p / (N ** (torch.Tensor(range(d // 2)) * 2 / d)))
    return pe


# class LayerNorm(nn.Module):
#
#     def __init__(self, *, d: int, eps: float = 1e-8) -> None:
#         self.d = d
#         self.eps = eps
#         self.gamma = torch.ones(d) /d
#         self.beta = torch.zeros(d)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         mean = x.mean(dim=-1, keepdim=True)
#         var = x.var(dim=-1, correction=0, keepdim=True)
#         std = torch.sqrt(var + self.eps)
#
#         out = torch.mul((x - mean) / std, self.gamma) + self.beta
#         return out


# class Embedding:
#
#     def __init__(self, *, vocab_size: int, d: int) -> None:
#         self.vocab_size = vocab_size
#         self.d = d
#         self.weights = torch.randn(vocab_size, d) / math.sqrt(d)
#
#     def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
#         emb = self.weights[input_ids]
#         return emb


class DecoderBlock(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d)
        self.attention = Attention(d=d, h=h)

        # TODO residual connection

        self.layer_norm2 = nn.LayerNorm(d)
        self.linear = nn.Linear(d, d)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Transformer block with residual connections.
        attn_out = self.attention(self.layer_norm1(x), mask=None)
        x = x + attn_out

        ff_out = self.relu(self.linear(self.layer_norm2(x)))
        x = x + ff_out

        return x


class Transformer(nn.Module):
    def __init__(self, *, vocab_size: int, d: int, n_layers: int, h: int) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d = d

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d)
        pe = positional_encoding(max_seq_len=100, d=d, N=10)
        self.pe: torch.Tensor
        self.register_buffer("pe", pe)

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d=d, h=h) for _ in range(n_layers)]
        )
        self.linear = nn.Linear(d, self.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[-1]

        x1 = self.embedding.forward(input_ids)
        pe = cast(torch.Tensor, self.pe)[:seq_len, :].clone()
        x2 = x1 + pe

        xs = [x2]
        for i, decoder in enumerate(self.decoder_blocks):
            new_x = decoder(xs[i])
            xs.append(new_x)

        x3 = xs[-1]
        x4 = self.linear(x3)

        # x5 = torch.softmax(x4, dim=-1)    # 不应该加 softmax, pytorch.cross_entropy 已经包含 softmax

        x6 = x4[:, -1, :]  # the output of last token

        return x6


# TODO loss function
# TODO check backprop
# TODO optimize
