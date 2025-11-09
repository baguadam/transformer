from torch import nn, Tensor
from transformer.layers import MultiHeadAttention, PositionWiseFeedForward

class DecoderLayer(nn.Module):
    """
    Implements a single Transformer Decoder block as described in "Attention Is All You Need" paper. 

    Sequence of sublayers:
        1) Masked Multi-Head Self-Attention
        2) Add & Norm
        3) Cross (Encoder-Decoder) Attention
        4) Add & Norm
        5) Position-wise Feed-Forward
        6) Add & Norm
    """
    def __init__(self, d_model: int = 512, d_ff: int = 2048, num_heads: int = 8, dropout: float = 0.1):
        """
        Creates an instance of DecoderLayer.

        Args:
            d_model (int): Model's dimension.
            d_ff (int): Hidden size of the feed-forward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability after attention and FFN.
        """
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ca = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.pwffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, encoder_out: Tensor, tgt_mask: Tensor | None = None, memory_mask: Tensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Decoder input of shape (batch_size, tgt_len, d_model)
            encoder_out (Tensor): Encoder output of shape (batch_size, source_seq_len, d_model)
            tgt_mask (optional Tensor): Causal/padding mask for target side, broadcastable to (batch, heads|1, tgt_len, tgt_len)
            memory_mask (optional Tensor): Padding mask for encoder memory, broadcastable to (batch, heads|1, tgt_len, src_len)

        Returns:
            norm3_out (Tensor): Output tensor of shape (batch_size, tgt_len, d_model)
        """
        attn_out = self.mha(x, x, x, mask=tgt_mask)
        attn_out = self.dropout(attn_out)
        norm1_out = self.norm1(x + attn_out)

        cross_attn_out = self.ca(norm1_out, encoder_out, encoder_out, mask=memory_mask)
        cross_attn_out = self.dropout(cross_attn_out)
        norm2_out = self.norm2(norm1_out + cross_attn_out)

        ffn_out = self.pwffn(norm2_out)
        ffn_out = self.dropout(ffn_out)
        norm3_out = self.norm3(norm2_out + ffn_out)
        return norm3_out


class Decoder(nn.Module):
    """
    Implements the Decoder stack with the given number of layers. 
    """
    def __init__(self, num_layers: int, d_model: int = 512, d_ff: int = 2048, num_heads: int = 8, dropout: float = 0.1):
        """
        Creates an instance of Decoder.

        Args:
            num_layers (int): Number of layers.
            d_model (int): Model's dimension.
            d_ff (int): Feed-forward hidden size for each layer.
            num_heads (int): Attention heads per layer.
            dropout (float): Dropout probability in each layer.
        """
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x: Tensor, enc_out: Tensor, tgt_mask: Tensor | None = None, memory_mask: Tensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (batch_size, tgt_len, d_model)
            enc_out (Tensor): Encoder's output tensor of shape (batch_size, tgt_len, d_model)
            tgt_mask (optional Mask): Causal/padding mask for target side, broadcastable to (batch, heads|1, tgt_len, tgt_len)
            memory_mask (optional Mask): Padding mask for encoder memory, broadcastable to (batch, heads|1, tgt_len, src_len)

        Returns:
            x (Tensor): Output tensor of shape (batch_size, tgt_len, d_model) 
        """
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return x
