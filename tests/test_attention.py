import torch
import pytest
from transformer.layers.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
)


def test_sdpa_shapes_and_simplemask():
    """Basic shape and masking test for scaled_dot_product_attention."""
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 5, 4)
    v = torch.randn(2, 5, 6)
    mask = torch.ones(2, 3, 5)
    mask[:, :, -2:] = 0
    out, attn = scaled_dot_product_attention(q, k, v, mask=mask)
    assert out.shape == (2, 3, 6)
    assert attn.shape == (2, 3, 5)
    rowsum = attn.sum(-1)
    assert torch.allclose(
        rowsum[mask.sum(-1) > 0], torch.ones_like(rowsum[mask.sum(-1) > 0]), atol=1e-5
    )


def test_sdpa_uniform_scores_returns_mean_of_values():
    """Test uniform attention behavior."""
    B, Lq, Lk, dk, dv = 2, 3, 5, 4, 6
    q = torch.zeros(B, Lq, dk)
    k = torch.zeros(B, Lk, dk)
    v = torch.stack(
        [torch.arange(Lk).float().unsqueeze(1).repeat(1, dv) for _ in range(B)], dim=0
    )
    out, attn = scaled_dot_product_attention(q, k, v)
    expected = v.mean(dim=1).unsqueeze(1).repeat(1, Lq, 1)
    assert torch.allclose(out, expected, atol=1e-5)
    assert torch.allclose(attn, torch.full_like(attn, 1.0 / Lk), atol=1e-6)


def test_sdpa_boolean_mask_equivalence():
    """
    Boolean mask (True=keep, False=mask) should behave like 1/0 mask for this implementation.
    """
    B, Lq, Lk, dk, dv = 1, 2, 4, 3, 5
    q = torch.randn(B, Lq, dk)
    k = torch.randn(B, Lk, dk)
    v = torch.randn(B, Lk, dv)

    bin_mask = torch.tensor([[[1, 1, 0, 0]]], dtype=torch.float32).repeat(B, Lq, 1)
    bool_mask = bin_mask.bool()

    out_bin, attn_bin = scaled_dot_product_attention(q, k, v, mask=bin_mask)
    out_bool, attn_bool = scaled_dot_product_attention(q, k, v, mask=bool_mask)

    assert torch.allclose(out_bin, out_bool, atol=1e-6)
    assert torch.allclose(attn_bin, attn_bool, atol=1e-6)
    assert torch.all(attn_bin[..., 2:] == 0)


def test_sdpa_backward_gradients_flow():
    """
    Ensure gradients flow through q,k,v.
    """
    B, Lq, Lk, dk, dv = 2, 3, 4, 5, 6
    q = torch.randn(B, Lq, dk, requires_grad=True)
    k = torch.randn(B, Lk, dk, requires_grad=True)
    v = torch.randn(B, Lk, dv, requires_grad=True)
    out, _ = scaled_dot_product_attention(q, k, v)
    loss = out.pow(2).mean()
    loss.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_mha_project_shapes_and_heads():
    """Verify MultiHeadAttention output dimensions and internal consistency."""
    B, L, D, H = 2, 7, 64, 8
    x = torch.randn(B, L, D)
    mha = MultiHeadAttention(d_model=D, num_heads=H)
    y = mha(x, x, x)
    assert y.shape == (B, L, D)


def test_mha_raises_if_bad_heads():
    """
    MultiHeadAttention should raise an AssertionError if d_model is not
    divisible by num_heads, since each head must have an equal share of D.
    """
    with pytest.raises(AssertionError):
        _ = MultiHeadAttention(d_model=63, num_heads=8)
