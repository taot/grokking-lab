import torch
from torch import nn
from torch.testing import assert_close

from transformer.model import softmax, Attention


def test_softmax() -> None:
    x = torch.Tensor([[1, 2], [3, 4]])
    y = softmax(x)
    print(y)
    assert_close(y, torch.tensor([[0.268941, 0.098938], [1.987223, 0.731059]]))


def test_attention() -> None:
    d = 4
    batch = 1
    seq_len = 3

    my_mha = Attention(d, d, d)
    official_mha = nn.MultiheadAttention(d, 1, batch_first=True)

    with torch.no_grad():
        # 注意 PyTorch MHA 的权重排列方式
        # my_mha.W_q.weight.copy_(official_mha.in_proj_weight[:d])
        # my_mha.W_k.weight.copy_(official_mha.in_proj_weight[d:2 * d])
        # my_mha.W_v.weight.copy_(official_mha.in_proj_weight[2 * d:])
        # my_mha.W_o.weight.copy_(official_mha.out_proj.weight)

        W = official_mha.in_proj_weight.detach()
        my_mha.W_q = W[:d].clone()
        my_mha.W_k = W[d:2 * d].clone()
        my_mha.W_v = W[2 * d:].clone()

    # 测试
    x = torch.randn(batch, seq_len, d)

    official_out, official_attn = official_mha(x, x, x)
    my_out = my_mha.forward(x, None)

    print(f"输出差异: {(official_out - my_out).abs().max():.2e}")  # 应该 < 1e-5
    # print(f"Attention 差异: {(official_attn - my_attn).abs().max():.2e}")
