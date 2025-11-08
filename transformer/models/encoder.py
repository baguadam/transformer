from torch import nn
from transformer.layers import MultiHeadAttention, PositionWiseFeedForward

class Encoder(nn.Module):
    """
    Implements a single Transformer Ecndoer block as described in "Attention Is All You Need" paper
    """
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        """
        Creates an instance of Encoder.

        Args:
            d_model (int): Model's dimension.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model)
        self.pwffn = PositionWiseFeedForward(d_model=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            norm2_out (Tensor): Output tensor of same shape as input.
        """
        attn_out = self.mha(x, x, x)
        attn_out = self.dropout(attn_out)
        norm1_out = self.norm1(x + attn_out)
        ffn_out = self.pwffn(norm1_out)
        ffn_out = self.dropout(ffn_out)
        norm2_out = self.norm2(norm1_out + ffn_out)
        return norm2_out