import torch
from transformer.models.transformer import Transformer

def test_transformer_forward_shapes():
    B, Ls, Lt = 2, 12, 10
    Vsrc = Vtgt = 100
    model = Transformer(
        src_vocab=Vsrc, tgt_vocab=Vtgt,
        num_layers=2, d_model=64, d_ff=128, num_heads=8, pad_id=0, max_len=128
    )
    src = torch.randint(1, Vsrc, (B, Ls))
    tgt = torch.randint(1, Vtgt, (B, Lt))
    logits = model(src, tgt)
    assert logits.shape == (B, Lt, Vtgt)

def test_pad_ignores_loss_positions():
    B, Ls, Lt = 2, 6, 5
    V = 50
    pad = 0
    model = Transformer(
        src_vocab=V, tgt_vocab=V,
        num_layers=1, d_model=32, d_ff=64, num_heads=4, pad_id=pad, max_len=64
    )
    src = torch.randint(1, V, (B, Ls))
    tgt = torch.randint(1, V, (B, Lt))
    tgt[:, -2:] = pad
    logits = model(src, tgt)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad)
    loss = loss_fn(logits.reshape(-1, V), tgt.reshape(-1))
    assert torch.isfinite(loss)
