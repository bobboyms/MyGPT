import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# class CausalSelfAttention(nn.Module):

#     def __init__(self, embed_dim, heads, dropout):
#         super().__init__()
#         assert embed_dim % heads == 0
#         self.c_attn = nn.Linear(
#             embed_dim, 3 * embed_dim, bias=False)
#         self.c_proj = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.attn_dropout = nn.Dropout(dropout)
#         self.resid_dropout = nn.Dropout(dropout)
#         self.n_head = heads
#         self.n_embd = embed_dim
#         self.dropout = dropout

#     def forward(self, x, mask):
#         B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

#         # calculate query, key, values for all heads in batch and move head forward to be the batch dim
#         q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
#         k = k.view(B, T, self.n_head, C //
#                    self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#         q = q.view(B, T, self.n_head, C //
#                    self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#         v = v.view(B, T, self.n_head, C //
#                    self.n_head).transpose(1, 2)  # (B, nh, T, hs)

#         y = F.scaled_dot_product_attention(
#             q, k, v, attn_mask=None,
#             dropout_p=self.dropout if self.training else 0,
#             is_causal=True)

#         # re-assemble all head outputs side by side
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         # output projection
#         y = self.resid_dropout(self.c_proj(y))
#         return y

# Esse código "CausalSelfAttention" foi inspirado no código do Andrej karpathy
# https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, heads: int, dropout: float):
        super().__init__()
        if embed_dim % heads != 0:
            raise ValueError("embed_dim deve ser divisível por heads")
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = heads
        self.n_embd = embed_dim
        self.dropout = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        x = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.size()

        # calcular query, key, values para todos os heads em batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = map(self._split_heads, [q, k, v])

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class ResidualAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        """
        Initializes the ResidualAttention module.

        Args:
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        """
        super(ResidualAttention, self).__init__()
        # self.attention = nn.MultiheadAttention(
        #     embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.attention = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualAttention module.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The padding mask tensor.

        Returns:
            torch.Tensor: The output tensor after processing.
        """
        x = x + self.attention(self.norm(x), mask)
        return x
