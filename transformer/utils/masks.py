import torch
from torch import Tensor

def make_padding_mask(seq: Tensor, pad_id: int) -> Tensor:
    """
    Create a 1/0 padding mask.

    Args:
        seq (Tensor): (batch, L) token ids
        pad_id (int): Padding token id

    Returns:
        mask (Tensor): (batch, 1, 1, L) mask with 1 where tokens are valid, 0 where padding
    """
    mask = (seq != pad_id).unsqueeze(1).unsqueeze(2)
    return mask.to(seq.dtype)


def make_causal_mask(size: int, device=None, dtype=torch.float32) -> Tensor:
    """
    Create a lower-triangular 1/0 causal mask for self-attention.

    Args:
        size (int): Sequence length (L)
        device: Torch device
        dtype: Mask dtype (float recommended)

    Returns:
        tri (Tensor): (1, 1, L, L) mask with 1 on/under diagonal, 0 above (future positions)
    """
    tri = torch.tril(torch.ones(size, size, device=device, dtype=dtype))
    return tri.unsqueeze(0).unsqueeze(0)
