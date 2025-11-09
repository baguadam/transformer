import torch
from transformer.layers import PositionWiseFeedForward

def test_ffn_shape_and_respects_batch_seq():
    B, L, D, DF = 2, 5, 32, 64
    x = torch.randn(B, L, D)
    ffn = PositionWiseFeedForward(d_model=D, d_ff=DF, dropout=0.0)
    y = ffn(x)
    assert y.shape == (B, L, D)