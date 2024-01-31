import torch
import torch.nn as nn


class ResidualFeedForward(nn.Module):
    def __init__(self, embed_dim: int, dropout: float) -> None:
        """
        Initializes the ResidualFeedForward module.

        Args:
            embed_dim (int): The embedding dimension size.
        """
        super(ResidualFeedForward, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),

        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualFeedForward module.

        Args:
            x (torch.Tensor): Input tensor to the module.

        Returns:
            torch.Tensor: Output tensor after processing.
        """
        x = x + self.sequential(self.norm(x))
        return x
