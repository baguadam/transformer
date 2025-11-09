import torch
from transformer.layers.embeddings import calculate_positional_encoding


def test_positional_encoding_shape_and_range():
    """Basic shape and numerical sanity test for positional encoding."""
    L, D = 20, 32
    pe = calculate_positional_encoding(seq_len=L, dim=D)
    assert pe.shape == (L, D)
    assert pe.abs().max() <= 1.0 + 1e-6


def test_positional_encoding_repro_small():
    """Deterministic test for small positional encodings."""
    L, D = 4, 6
    pe = calculate_positional_encoding(seq_len=L, dim=D)
    assert torch.allclose(pe[0, 0::2], torch.zeros_like(pe[0, 0::2]))
    assert torch.allclose(pe[0, 1::2], torch.ones_like(pe[0, 1::2]))
