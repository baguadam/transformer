import torch
from torch import Tensor

def calculate_positional_encoding(seq_len: int, dim: int, N: int=10000) -> Tensor:
    """
    Calculates the sinusoidal encoding of the given sequence.

    Args:
        seq_len: (int) Length of the sequence.
        dim: (int) Dimension of the model.
        N: (int) Denomater base for frequencies. 

    Returns:
        p: (Tensor) The positional encoding suitable to add embeddings.
    """
    P = torch.zeros(seq_len, dim)
    for pos in range(seq_len):
        for i in range(dim):
            denom = torch.pow(torch.tensor(N, dtype=torch.float32), 2 * (i // 2) / dim)
            P[pos, i] = torch.sin(pos / denom) if i % 2 == 0 else torch.cos(pos / denom)
    return P