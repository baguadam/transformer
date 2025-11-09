import torch
from transformer.models.encoder import Encoder, EncoderLayer
from transformer.models.decoder import Decoder, DecoderLayer


def test_encoder_layer_shapes_with_mask():
    """Ensure a single EncoderLayer returns the correct output shape when given a mask."""
    B, S, D, H = 2, 11, 64, 8
    x = torch.randn(B, S, D)
    layer = EncoderLayer(d_model=D, d_ff=128, num_heads=H, dropout=0.0)
    src_mask = torch.ones(B, 1, S, S)
    y = layer(x, src_mask=src_mask)
    assert y.shape == (B, S, D)


def test_decoder_layer_shapes_with_masks():
    """Verify that a single DecoderLayer outputs the correct tensor shape with masks."""
    B, S, T, D, H = 2, 11, 7, 64, 8
    x = torch.randn(B, T, D)
    mem = torch.randn(B, S, D)
    layer = DecoderLayer(d_model=D, d_ff=128, num_heads=H, dropout=0.0)
    tgt_mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)
    mem_mask = torch.ones(B, 1, T, S)
    y = layer(x, mem, tgt_mask=tgt_mask, memory_mask=mem_mask)
    assert y.shape == (B, T, D)


def test_encoder_decoder_stacks():
    """Confirm that Encoder and Decoder stacks integrate correctly."""
    B, S, T, D, H = 2, 12, 9, 128, 8
    enc = Encoder(num_layers=2, d_model=D, d_ff=256, num_heads=H, dropout=0.0)
    dec = Decoder(num_layers=2, d_model=D, d_ff=256, num_heads=H, dropout=0.0)
    src = torch.randn(B, S, D)
    mem = enc(src)
    tgt = torch.randn(B, T, D)
    out = dec(tgt, mem)
    assert mem.shape == (B, S, D)
    assert out.shape == (B, T, D)


def test_encoder_padding_mask_invariance():
    """Check that the encoder ignores padded tokens when masked."""
    B, S, D, H = 1, 8, 32, 4
    x = torch.randn(B, S, D)
    layer = EncoderLayer(d_model=D, d_ff=64, num_heads=H, dropout=0.0)
    src_mask = torch.ones(B, 1, 1, S)
    src_mask[..., -3:] = 0

    y1 = layer(x, src_mask=src_mask)

    x2 = x.clone()
    x2[..., -3:, :] = torch.randn_like(x2[..., -3:, :]) * 10.0
    y2 = layer(x2, src_mask=src_mask)
    keep_q = slice(0, S - 3)

    assert torch.allclose(y1[:, keep_q, :], y2[:, keep_q, :], atol=1e-6)


def test_encoder_accepts_broadcast_and_square_masks():
    """Ensure Encoder supports both broadcasted and square mask shapes."""
    B, S, D, H = 2, 10, 64, 8
    x = torch.randn(B, S, D)
    enc = Encoder(num_layers=2, d_model=D, d_ff=128, num_heads=H, dropout=0.0)

    pad_mask = torch.ones(B, 1, 1, S)  # broadcast
    y1 = enc(x, src_mask=pad_mask)

    square_mask = torch.ones(B, 1, S, S)
    square_mask[..., -2:] = 0
    y2 = enc(x, src_mask=square_mask)

    assert y1.shape == (B, S, D)
    assert y2.shape == (B, S, D)


def test_decoder_causal_independence_from_future_tokens():
    """Test that the decoder's causal mask prevents future leakage."""
    B, T, S, D, H = 1, 8, 6, 32, 4
    layer = DecoderLayer(d_model=D, d_ff=64, num_heads=H, dropout=0.0)
    x = torch.randn(B, T, D)
    mem = torch.randn(B, S, D)

    causal = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)

    y1 = layer(x, mem, tgt_mask=causal, memory_mask=None)
    t = T // 2
    x2 = x.clone()
    x2[:, t + 1 :, :] = torch.randn_like(x2[:, t + 1 :, :]) * 10.0

    y2 = layer(x2, mem, tgt_mask=causal, memory_mask=None)

    assert torch.allclose(y1[:, : t + 1, :], y2[:, : t + 1, :], atol=1e-6)


def test_decoder_masks_shapes_broadcast():
    """Verify that the decoder supports combined (causal x padding) masks and
    cross-attention memory masks with correct broadcasting."""
    B, T, S, D, H = 2, 7, 9, 48, 6
    layer = DecoderLayer(d_model=D, d_ff=96, num_heads=H, dropout=0.0)
    x = torch.randn(B, T, D)
    mem = torch.randn(B, S, D)

    causal = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)
    pad_t = torch.ones(B, 1, 1, T)
    pad_t[..., -2:] = 0
    tgt_mask = pad_t * causal

    mem_mask = torch.ones(B, 1, T, S)
    mem_mask[..., -3:] = 0

    y = layer(x, mem, tgt_mask=tgt_mask, memory_mask=mem_mask)
    assert y.shape == (B, T, D)


def test_grad_flow_encoder_and_decoder_layer():
    """Confirm that gradients propagate correctly through encoder and decoder layers."""
    B, S, T, D, H = 2, 6, 5, 32, 4
    enc_layer = EncoderLayer(d_model=D, d_ff=64, num_heads=H, dropout=0.0)
    dec_layer = DecoderLayer(d_model=D, d_ff=64, num_heads=H, dropout=0.0)

    x = torch.randn(B, S, D, requires_grad=True)
    mem = enc_layer(x)  # (B,S,D)

    y_in = torch.randn(B, T, D, requires_grad=True)
    causal = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)
    out = dec_layer(y_in, mem, tgt_mask=causal)

    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert y_in.grad is not None and torch.isfinite(y_in.grad).all()


def test_encoder_stack_padding_mask_invariance():
    """Verify that multi-layer encoders ignore masked (padded) positions."""
    B, S, D, H = 1, 10, 64, 8
    enc = Encoder(num_layers=3, d_model=D, d_ff=128, num_heads=H, dropout=0.0)
    x = torch.randn(B, S, D)

    mask = torch.ones(B, 1, 1, S)
    mask[..., -4:] = 0
    y1 = enc(x, src_mask=mask)

    x2 = x.clone()
    x2[..., -4:, :] = torch.randn_like(x2[..., -4:, :]) * 10
    y2 = enc(x2, src_mask=mask)

    keep_q = slice(0, S - 4)
    assert torch.allclose(y1[:, keep_q, :], y2[:, keep_q, :], atol=1e-6)
