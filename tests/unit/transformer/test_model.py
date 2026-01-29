import math

import pytest
import torch
from torch import nn
from torch.testing import assert_close

from transformer.model import softmax, Attention, positional_encoding, LayerNorm, Embedding, relu, Transformer, xavier_init


def test_softmax() -> None:
    x = torch.Tensor([[1, 2], [3, 5]])
    y = softmax(x)
    print("\ny:")
    print(y)
    expected = torch.Tensor([
        [0.268941, 0.731059],
        [0.119203, 0.880797]
    ])
    assert_close(y, expected)


def test_softmax_batch() -> None:
    x = torch.Tensor([
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    ])

    y = softmax(x)

    expected = torch.Tensor([
        [
            [0.250000, 0.250000, 0.250000, 0.250000],
            [0.250000, 0.250000, 0.250000, 0.250000],
            [0.250000, 0.250000, 0.250000, 0.250000]
        ],
        [
            [0.250000, 0.250000, 0.250000, 0.250000],
            [0.250000, 0.250000, 0.250000, 0.250000],
            [0.250000, 0.250000, 0.250000, 0.250000]
        ]
    ])

    assert_close(y, expected)


def test_relu() -> None:
    x = torch.Tensor([[1, -1, -2], [2, 0, 1]])
    y = relu(x)
    expected = torch.Tensor([
        [1., 0., 0.],
        [2., 0., 1.]
    ])

    assert_close(y, expected)


def plain_positional_encoding(p: int, j: int, d: int, N: int) -> float:
    if j % 2 == 0:
        i = j // 2
        v = math.sin(p / math.pow(N, 2 * i / d))
    else:
        i = (j - 1) // 2
        v = math.cos(p / math.pow(N, 2 * i / d))
    return v


def test_positional_encoding() -> None:
    pe = positional_encoding(max_seq_len=5, d=4, N=8)

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


@pytest.mark.parametrize(
    "input_ids, expected",
    [
        [
            torch.LongTensor([2, 0, 1]),
            torch.Tensor([
                [5., 6.],
                [1., 2.],
                [3., 4.]
            ])
        ],
        [
            torch.LongTensor([
                [0, 1, 2],
                [1, 2, 0],
            ]),
            torch.Tensor([
                [
                    [1., 2.],
                    [3., 4.],
                    [5., 6.]
                ],
                [
                    [3., 4.],
                    [5., 6.],
                    [1., 2.]
                ]
            ])
        ]
    ]
)
def test_embedding(input_ids: torch.Tensor, expected: torch.Tensor) -> None:
    embedding = Embedding(vocab_size=3, d=2)
    embedding.weights = torch.Tensor([
        [1, 2],
        [3, 4],
        [5, 6],
    ])

    emb = embedding.forward(input_ids)
    assert_close(emb, expected)


def test_transformer() -> None:
    transformer = Transformer(vocab_size=3, d=4, n_layers=1)

    x = torch.LongTensor([
        [2, 0, 1]
    ])

    out = transformer.forward(x)

    print(out)
