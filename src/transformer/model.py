import math
from typing import Optional

import torch


def assert_dimension(x: torch.Tensor, dim: tuple[int, ...]) -> None:
    assert x.shape == dim


def softmax(x: torch.Tensor) -> torch.Tensor:
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=-1)
    return exp_x / sum_x


class Attention:

    def __init__(self, d: int, d_k: int, d_v: int) -> None:
        self.d = d
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = torch.Tensor(d, d_k)
        self.W_k = torch.Tensor(d, d_k)
        self.W_v = torch.Tensor(d, d_v)

        # projection
        self.W_proj = torch.Tensor(d_v, d)
        self.b_proj = torch.Tensor(d)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        assert_dimension(x, (batch_size, seq_len, self.d))

        q = torch.matmul(x, self.W_q)       # batch_size * seq_len * d_k
        k = torch.matmul(x, self.W_k)       # batch_size * seq_len * d_k
        v = torch.matmul(x, self.W_v)       # batch_size * seq_len * d_v

        weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)   # batch_size * seq_len * seq_len
        weights_softmax = softmax(weights)          # batch_size * seq_len * seq_len
        attn = torch.matmul(weights_softmax, v)     # batch_size * seq_len * d_v

        out = torch.matmul(attn, self.W_proj) + self.b_proj

        return out


def positional_encoding(seq_len: int, d: int, N: int = 100000) -> torch.Tensor:
    pe = torch.zeros(seq_len, d)
    for p in range(seq_len):
        pe[p, ::2] = torch.sin(p / (N ** (torch.Tensor(range(d // 2)) * 2 / d)))
        pe[p, 1::2] = torch.cos(p / (N ** (torch.Tensor(range(d // 2)) * 2 / d)))
    return pe


class LayerNorm:

    def __init__(self, *, d: int, eps: float = 1e-8) -> None:
        self.d = d
        self.eps = eps
        self.gamma = torch.ones(d) /d
        self.beta = torch.zeros(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, correction=0, keepdim=True)
        std = torch.sqrt(var + self.eps)

        out = torch.mul((x - mean) / std, self.gamma) + self.beta
        return out
