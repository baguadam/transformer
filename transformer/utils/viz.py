from torch import Tensor
import matplotlib.pyplot as plt

def visualize_encoding(P: Tensor):
    """
    Plot a Position (y) × Dimension (x) heatmap of a positional encoding.

    Args:
        P (Tensor): Encoding of shape [seq_len, d_model]
    """
    plt.figure(figsize=(10, 6))
    cax = plt.matshow(P, cmap='viridis')
    plt.title('Positional Encoding')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.colorbar(cax)
    plt.show()
