from torch import nn, Tensor
from transformer.layers import MultiHeadAttention, PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """
    Implements a single Transformer Encoder layer as described in "Attention Is All You Need" paper.

    Sequence of sublayers:
        1) Multi-Head Self-Attention
        2) Add & Norm
        3) Point-Wise Feed-Forward Network
        4) Add & Norm
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Creates an instance of EncoderLayer.

        Args:
            d_model (int): Model's dimension.
            d_ff (int): Hidden size of the feed-forward network
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.pwffn = PositionWiseFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            src_mask (optional Tensor): Optional mask broadcastable to (batch, heads|1, src_len, src_len)

        Returns:
            norm2_out (Tensor): Output tensor of same shape as input.
        """
        attn_out = self.mha(x, x, x, mask=src_mask)
        attn_out = self.dropout(attn_out)
        norm1_out = self.norm1(x + attn_out)

        ffn_out = self.pwffn(norm1_out)
        ffn_out = self.dropout(ffn_out)
        norm2_out = self.norm2(norm1_out + ffn_out)
        return norm2_out


class Encoder(nn.Module):
    """
    Implements the Encoder stack with the given number of layers.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int = 512,
        d_ff: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Creates an instance of Encoder.

        Args:
            num_layers (int): Number of layers.
            d_model (int): Model's dimension.
            d_ff (int): Hidden size of the feed-forward network for each layer.
            num_heads (int): Attention heads per layer.
            dropout (float): Dropout's probability.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            src_mask (Optional tensor): Mask, broadcastable to (batch, heads|1, src_len, src_len)

        Returns:
            x (Tensor): Output tensor of same shape as input.
        """
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return x
