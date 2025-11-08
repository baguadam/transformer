import torch
from torch import Tensor
from torch import nn

def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """
    Calculates the scaled dot-product attention based on "Attention Is All You Need" paper.
    
    Args:
        q (Tensor): Query of shape (..., L_q, d_k)
        k (Tensor): Key of shape   (..., L_k, d_k)
        v (Tensor): Value of shape (..., L_k, d_v)
        mask (optional Tensor): Optional mask parameter, broadcastable to (..., L_q, d_k)

    Returns: 
        out (Tensor): Attention output of shape (..., L_q, d_k)
        attn (Tensor): Attention weights of shape (..., L_q, L_k)
    """
    k_dim = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k_dim ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out, attn


class MultiHeadAttention(nn.Module):
    """
    Class representing the Multi-Head Attention layer in the architecture.
    It can be used for both self-attention and cross-attention

    Attributes:
        num_heads (int): Number of attention heads.
        d_model (int): Model's dimension.
    """
    def __init__(self, num_heads: int  = 6, d_model: int = 512):
        """
        Initializes a MultiHeadAttention object.

        Args:
            num_heads (int): Number of attention heads.
            d_model (int): Model's dimension
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out      = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        (batch, seq, d_model) -> (batch, heads, seq, d_k)
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: Tensor) -> Tensor:
        """
        (batch, heads, seq, d_k) -> (batch, seq, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            q, k, v (Tensor): Query, Key and Value shape of (batch_size, seq_len, d_model)
            mask (optional Tensor): Broadcastable mask to (batch_size, num_heads, L_q, L_k)

        Returns:
            out_proj (Tensor): The combined attention scores of shape (batch_size, seq_len, d_model)
        """
        q = self._split_heads(self.q_linear(q))
        k = self._split_heads(self.k_linear(k))
        v = self._split_heads(self.v_linear(v))

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        context_vector, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        out_proj = self.out(self._combine_heads(context_vector))
        return out_proj
    
if __name__ == "__main__":
    q = 10 * torch.rand((3, 4))
    k = 10 * torch.rand((3, 4))
    v = 10 * torch.rand((3, 4))
    context_vector, attn_weights = scaled_dot_product_attention(q, k, v)
    print("SDPA out shape:", context_vector.shape) # expect torch.Size([3, 4])
    print("SDPA attn shape:", attn_weights.shape)  # expect torch.Size([3, 4])

    num_heads = 8
    d_model = 512
    batch_size = 2
    seq_len = 10
    x = torch.rand(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
    out = mha(x, x, x)
    print("MHA out shape:", out.shape) # expect torch.Size([2, 10, 512])