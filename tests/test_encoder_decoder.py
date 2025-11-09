import torch
from transformer.models.encoder import Encoder, EncoderLayer
from transformer.models.decoder import Decoder, DecoderLayer

def test_encoder_layer_shapes_with_mask():
    B, S, D, H = 2, 11, 64, 8
    x = torch.randn(B, S, D)
    layer = EncoderLayer(d_model=D, d_ff=128, num_heads=H, dropout=0.0)
    src_mask = torch.ones(B, 1, S, S)
    y = layer(x, src_mask=src_mask)
    assert y.shape == (B, S, D)

def test_decoder_layer_shapes_with_masks():
    B, S, T, D, H = 2, 11, 7, 64, 8
    x = torch.randn(B, T, D)
    mem = torch.randn(B, S, D)
    layer = DecoderLayer(d_model=D, d_ff=128, num_heads=H, dropout=0.0)
    tgt_mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)
    mem_mask = torch.ones(B, 1, T, S)
    y = layer(x, mem, tgt_mask=tgt_mask, memory_mask=mem_mask)
    assert y.shape == (B, T, D)

def test_encoder_decoder_stacks():
    B, S, T, D, H = 2, 12, 9, 128, 8
    enc = Encoder(num_layers=2, d_model=D, d_ff=256, num_heads=H, dropout=0.0)
    dec = Decoder(num_layers=2, d_model=D, d_ff=256, num_heads=H, dropout=0.0)
    src = torch.randn(B, S, D)
    mem = enc(src)
    tgt = torch.randn(B, T, D)
    out = dec(tgt, mem)
    assert mem.shape == (B, S, D)
    assert out.shape == (B, T, D)