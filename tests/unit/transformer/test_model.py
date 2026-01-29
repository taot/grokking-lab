import math

import torch
from torch import nn
from torch.testing import assert_close

from transformer.model import softmax, Attention, positional_encoding, LayerNorm


def test_softmax() -> None:
    x = torch.Tensor([[1, 2], [3, 4]])
    y = softmax(x)
    print(y)
    assert_close(y, torch.tensor([[0.268941, 0.098938], [1.987223, 0.731059]]))


# def test_attention() -> None:
#     d = 4
#     batch = 1
#     seq_len = 3
#
#     my_mha = Attention(d, d, d)
#     official_mha = nn.MultiheadAttention(d, 1, batch_first=True)
#
#     with torch.no_grad():
#         # 注意 PyTorch MHA 的权重排列方式
#         # my_mha.W_q.weight.copy_(official_mha.in_proj_weight[:d])
#         # my_mha.W_k.weight.copy_(official_mha.in_proj_weight[d:2 * d])
#         # my_mha.W_v.weight.copy_(official_mha.in_proj_weight[2 * d:])
#         # my_mha.W_o.weight.copy_(official_mha.out_proj.weight)
#
#         W = official_mha.in_proj_weight.detach()
#         my_mha.W_q = W[:d].clone()
#         my_mha.W_k = W[d:2 * d].clone()
#         my_mha.W_v = W[2 * d:].clone()
#
#         official_mha.out_proj.weight
#
#     # 测试
#     x = torch.randn(batch, seq_len, d)
#
#     official_out, official_attn = official_mha(x, x, x)
#     my_out = my_mha.forward(x, None)
#
#     print(f"输出差异: {(official_out - my_out).abs().max():.2e}")  # 应该 < 1e-5
#     # print(f"Attention 差异: {(official_attn - my_attn).abs().max():.2e}")
#     assert_close(my_out, official_out)


def plain_positional_encoding(p: int, j: int, d: int, N: int) -> float:
    if j % 2 == 0:
        i = j // 2
        v = math.sin(p / math.pow(N, 2 * i / d))
    else:
        i = (j - 1) // 2
        v = math.cos(p / math.pow(N, 2 * i / d))
    return v


def test_positional_encoding() -> None:
    pe = positional_encoding(seq_len=5, d=4, N=8)

    expected = torch.Tensor([
            [ 0.000000,  1.000000,  0.000000,  1.000000],
            [ 0.841471,  0.540302,  0.346234,  0.938148],
            [ 0.909297, -0.416147,  0.649637,  0.760245],
            [ 0.141120, -0.989992,  0.872678,  0.488296],
            [-0.756802, -0.653644,  0.987766,  0.155944]
    ])

    assert_close(pe, expected)


def test_layer_norm() -> None:
    layer_norm = LayerNorm(d=3)
    layer_norm.beta = torch.Tensor([0.1, 0.2, 0.3])

    x = torch.Tensor([
        [1, 2, 3],
        [1, 4, 9],
    ])
    out = layer_norm.forward(x)

    expected = torch.Tensor([
        [-0.308248, 0.200000, 0.708248],
        [-0.270389, 0.132657, 0.737733]
    ])
    assert_close(out, expected)
