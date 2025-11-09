import torch
from transformer.layers.attention import scaled_dot_product_attention, MultiHeadAttention

def test_sdpa_shapes_and_simplemask():
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 5, 4)
    v = torch.randn(2, 5, 6)
    mask = torch.ones(2, 3, 5)
    mask[:, :, -2:] = 0
    out, attn = scaled_dot_product_attention(q, k, v, mask=mask)
    assert out.shape == (2, 3, 6)
    assert attn.shape == (2, 3, 5)
    rowsum = attn.sum(-1)
    assert torch.allclose(rowsum[mask.sum(-1) > 0], torch.ones_like(rowsum[mask.sum(-1) > 0]), atol=1e-5)

def test_mha_project_shapes_and_heads():
    B, L, D, H = 2, 7, 64, 8
    x = torch.randn(B, L, D)
    mha = MultiHeadAttention(d_model=D, num_heads=H)
    y = mha(x, x, x)
    assert y.shape == (B, L, D)

def test_mha_raises_if_bad_heads():
    import pytest
    with pytest.raises(AssertionError):
        _ = MultiHeadAttention(d_model=63, num_heads=8)
