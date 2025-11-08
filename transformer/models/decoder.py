from torch import nn
from transformer.layers import MultiHeadAttention, PositionWiseFeedForward

class Decoder(nn.Module):
    """
    Implements a single Transformer Decoder block as described in "Attention Is All You Need" paper 
    """
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        """
        Creates an instance of Decoder.

        Args:
            d_model (int): Model's dimension.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model)
        self.ca = MultiHeadAttention(d_model=d_model)
        self.pwffn = PositionWiseFeedForward(d_model=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out):
        """
        Forward pass.

        Args:
            x (Tensor): Decoder input of shape (batch_size, target_seq_len, d_model)
            encoder_out (Tensor): Encoder output of shape (batch_size, source_seq_len, d_model)

        Returns:
            norm3_out (Tensor): Output tensor of shape (batch_size, target_seq_len, d_model)
        """
        attn_out = self.mha(x, x, x)
        attn_out = self.dropout(attn_out)
        norm1_out = self.norm1(x + attn_out)

        cross_attn_out = self.ca(norm1_out, encoder_out, encoder_out)
        cross_attn_out = self.dropout(cross_attn_out)
        norm2_out = self.norm2(norm1_out + cross_attn_out)

        ffn_out = self.pwffn(norm2_out)
        ffn_out = self.dropout(ffn_out)
        norm3_out = self.norm3(norm2_out + ffn_out)
        return norm3_out
