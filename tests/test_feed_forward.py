import torch
from transformer.layers import PositionWiseFeedForward


def test_ffn_shape_and_respects_batch_seq():
    """Output shape matches input and operates position-wise."""
    B, L, D, DF = 2, 5, 32, 64
    x = torch.randn(B, L, D)
    ffn = PositionWiseFeedForward(d_model=D, d_ff=DF, dropout=0.0)
    y = ffn(x)
    assert y.shape == (B, L, D)
    perm = torch.randperm(L)
    y_perm = ffn(x[:, perm, :])
    assert torch.allclose(y[:, perm, :], y_perm, atol=1e-6)


def test_ffn_respects_dropout_rate():
    """Dropout should zero out roughly the expected proportion of activations."""
    torch.manual_seed(0)
    B, L, D, DF = 2, 8, 16, 64
    dropout = 0.5
    ffn = PositionWiseFeedForward(d_model=D, d_ff=DF, dropout=dropout)
    x = torch.randn(B, L, D)
    ffn.train()
    y = ffn(x)
    assert y.shape == (B, L, D)
    zero_ratio = (y.abs() < 1e-8).float().mean().item()
    assert 0.1 < zero_ratio < 0.9


def test_ffn_gradients_flow():
    """Ensure gradients propagate through FFN."""
    B, L, D, DF = 2, 6, 16, 32
    x = torch.randn(B, L, D, requires_grad=True)
    ffn = PositionWiseFeedForward(d_model=D, d_ff=DF, dropout=0.0)
    y = ffn(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_ffn_deterministic_in_eval_mode():
    """Dropout must be disabled in eval mode (deterministic output)."""
    torch.manual_seed(0)
    B, L, D, DF = 2, 5, 16, 32
    x = torch.randn(B, L, D)
    ffn = PositionWiseFeedForward(d_model=D, d_ff=DF, dropout=0.5)
    ffn.eval()
    y1 = ffn(x)
    y2 = ffn(x)
    assert torch.allclose(y1, y2, atol=1e-7)


def test_ffn_extreme_values_stability():
    """Ensure FFN doesn’t produce NaN/inf for large inputs."""
    B, L, D, DF = 2, 4, 32, 64
    x = torch.randn(B, L, D) * 1e3
    ffn = PositionWiseFeedForward(d_model=D, d_ff=DF, dropout=0.0)
    y = ffn(x)
    assert torch.isfinite(y).all()
