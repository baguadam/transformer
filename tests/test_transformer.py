import torch
from transformer.models.transformer import Transformer


def test_transformer_forward_shapes():
    """Basic forward pass and output shape check."""
    B, Ls, Lt = 2, 12, 10
    Vsrc = Vtgt = 100
    model = Transformer(
        src_vocab=Vsrc,
        tgt_vocab=Vtgt,
        num_layers=2,
        d_model=64,
        d_ff=128,
        num_heads=8,
        pad_id=0,
        max_len=128,
    )
    src = torch.randint(1, Vsrc, (B, Ls))
    tgt = torch.randint(1, Vtgt, (B, Lt))
    logits = model(src, tgt)
    assert logits.shape == (B, Lt, Vtgt)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_pad_ignores_loss_positions():
    """Ensure padding tokens do not blow up loss."""
    B, Ls, Lt = 2, 6, 5
    V = 50
    pad = 0
    model = Transformer(
        src_vocab=V,
        tgt_vocab=V,
        num_layers=1,
        d_model=32,
        d_ff=64,
        num_heads=4,
        pad_id=pad,
        max_len=64,
    )
    src = torch.randint(1, V, (B, Ls))
    tgt = torch.randint(1, V, (B, Lt))
    tgt[:, -2:] = pad
    logits = model(src, tgt)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad)
    loss = loss_fn(logits.reshape(-1, V), tgt.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_transformer_mask_causal_property():
    """Ensure causal mask prevents future leakage in target attention."""
    B, Ls, Lt = 1, 6, 6
    V = 32
    model = Transformer(
        src_vocab=V,
        tgt_vocab=V,
        num_layers=1,
        d_model=32,
        d_ff=64,
        num_heads=4,
        pad_id=0,
        max_len=64,
        dropout=0.0,
    )
    src = torch.randint(1, V, (B, Ls))
    tgt = torch.randint(1, V, (B, Lt))
    out1 = model(src, tgt)

    tgt2 = tgt.clone()
    tgt2[:, -1] = torch.randint(1, V, (1,))
    out2 = model(src, tgt2)

    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)


def test_transformer_eval_determinism():
    """In eval mode, repeated runs should be deterministic."""
    B, Ls, Lt = 2, 7, 6
    V = 20
    model = Transformer(
        src_vocab=V,
        tgt_vocab=V,
        num_layers=1,
        d_model=16,
        d_ff=32,
        num_heads=4,
        pad_id=0,
        max_len=32,
    )
    model.eval()
    src = torch.randint(1, V, (B, Ls))
    tgt = torch.randint(1, V, (B, Lt))
    y1 = model(src, tgt)
    y2 = model(src, tgt)
    assert torch.allclose(y1, y2, atol=1e-7)
