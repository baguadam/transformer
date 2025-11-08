import torch
from torch import Tensor

def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """
    Calculates the scaled dot-product attention based on "Attention Is All You Need" paper.
    
    Args:
        q: (Tensor) Query of shape (..., L_q, d_k)
        k: (Tensor) Key of shape (..., L_k, d_k)
        v: (Tensor) Value of shape (..., L_v, d_k)
        mark: (optional Tensor) Optional mask parameter, broadcastable to (..., L_q, d_k)

    Returns: 
        out: (Tensor) Attention output of shape (..., L_q, d_k)
        attn: (Tensor) Attention weights of shape (..., L_q, L_k)
    """
    k_dim = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k_dim ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out, attn