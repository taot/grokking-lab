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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        assert_dimension(x, (batch_size, seq_len, self.d))

        q = torch.matmul(x, self.W_q)       # batch_size * seq_len * d_k
        k = torch.matmul(x, self.W_k)    # batch_size * seq_len * d_k
        v = torch.matmul(x, self.W_v)    # batch_size * seq_len * d_v

        weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)   # batch_size * seq_len * seq_len
        weights_softmax = softmax(weights)          # batch_size * seq_len * seq_len
        attn = torch.matmul(weights_softmax, v)     # batch_size * seq_len * d_v

        return attn
