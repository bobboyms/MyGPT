import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoder import Encoder


class ModelLLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, dropout: float, num_layers: int) -> None:
        """
        Initializes the ModelLLM module.

        Args:
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            num_layers (int): The number of layers in the encoder.
        """
        super(ModelLLM, self).__init__()

        self.encoder = Encoder(vocab_size, embed_dim,
                               num_heads, dropout, num_layers)

        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ModelLLM module.

        Args:
            x (torch.Tensor): The input tensor.
            padding_mask (torch.Tensor): The padding mask tensor.

        Returns:
            torch.Tensor: The output tensor after processing.
        """
        logits = self.encoder(x, padding_mask)
        logits = self.linear(logits)
        return logits
