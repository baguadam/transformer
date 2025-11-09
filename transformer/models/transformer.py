import math
import torch
from torch import nn, Tensor

from transformer.models import Encoder, Decoder
from transformer.layers.embeddings import calculate_positional_encoding
from transformer.utils.masks import make_causal_mask, make_padding_mask


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer Architecture

    Components:
      - src/tgt embeddings + sinusoidal positional encodings
      - encoder & decoder stacks
      - final projection to target vocab
    """

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        num_layers: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_len: int = 5000,
    ):
        """
        Creates an instance of Transformer.

        Args:
            src_vocab (int): Source vocab size.
            tgt_vocab (int): Target vocab size.
            num_layers (int): Number of encoder/decoder layers.
            d_model (int): Model's dimension.
            d_ff (int): FFN's hidden dimension.
            num_heads (int): Attention heads.
            dropout (float): Dropout's probability.
            pad_id (int): Padding token id.
            max_len (int): Maximum sequence length for positional encodings.
        """
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.proj = nn.Linear(d_model, tgt_vocab)

        pe = calculate_positional_encoding(seq_len=max_len, dim=d_model)
        pe = pe.to(dtype=self.src_emb.weight.dtype)
        self.register_buffer("pe", pe, persistent=False)

    def _add_positional_encoding(self, x: Tensor) -> Tensor:
        """
        Add sinusoidal positional encoding to embeddings.

        Args:
            x (Tensor): (B, L, D)

        Returns:
            L (Tensor): (B, L, D)
        """
        L = x.size(1)
        return x + self.pe[:L, :]

    def forward(self, src_ids: Tensor, tgt_ids: Tensor) -> Tensor:
        """
        Teacher-forced forward pass.

        Args:
            src_ids (Tensor): (B, Ls) source token ids.
            tgt_ids (Tensor): (B, Lt) target token ids (shifted right)

        Returns:
            logits (Tensor): (B, Lt, tgt_vocab) unnormalized scores.
        """
        device = src_ids.device
        dtype = torch.float32

        src_pad_mask = make_padding_mask(src_ids, self.pad_id)
        tgt_pad_mask = make_padding_mask(tgt_ids, self.pad_id)
        causal = make_causal_mask(tgt_ids.size(1), device=device, dtype=dtype)
        tgt_mask = tgt_pad_mask * causal
        memory_mask = src_pad_mask

        src = self.src_emb(src_ids) * math.sqrt(self.d_model)
        src = self.dropout(self._add_positional_encoding(src))

        tgt = self.tgt_emb(tgt_ids) * math.sqrt(self.d_model)
        tgt = self.dropout(self._add_positional_encoding(tgt))

        memory = self.encoder(src, src_mask=memory_mask)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        logits = self.proj(out)
        return logits
