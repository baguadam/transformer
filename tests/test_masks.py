import torch
from transformer.utils.masks import make_padding_mask, make_causal_mask

def test_padding_mask_shape_and_values():
    ids = torch.tensor([[1,2,0,0],[3,4,5,0]])
    mask = make_padding_mask(ids, pad_id=0)
    assert mask.shape == (2,1,1,4)
    expect = torch.tensor([[[[1,1,0,0]]], [[[1,1,1,0]]]], dtype=mask.dtype)
    assert torch.equal(mask, expect)

def test_causal_mask_is_lower_triangular():
    m = make_causal_mask(5)
    assert m.shape == (1,1,5,5)
    upper = torch.triu(torch.ones(5,5), diagonal=1)
    assert (m.squeeze() * upper).sum() == 0
