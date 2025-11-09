import torch
from torch import nn, Tensor


class PositionWiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network layer.

    Attributes:
        ffn (nn.Sequential): The sequential block of two linear projection and a ReLU()
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        """
        Creates an instance of PositionWiseFeedForward.

        Args:
            d_model (int): Model's dimension.
            d_ff (int): Feed-Forward Network's dimension.
            dropout (float): Dropout value.
        """
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            x (Tensor): Processed tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.ffn(x)
        return x


if __name__ == "__main__":
    x = torch.rand(2, 10, 512)
    ffn = PositionWiseFeedForward()
    y = ffn(x)
    print("FFN output shape:", y.shape)  # expect torch.Size([2, 10, 512])
