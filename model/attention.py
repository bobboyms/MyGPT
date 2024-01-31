import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# class CausalSelfAttention(nn.Module):
#     def __init__(self, embed_dim, heads, dropout):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.heads = heads
#         self.head_dim = embed_dim // heads
#         self.dropout = nn.Dropout(dropout)

#         assert (
#             self.head_dim * heads == embed_dim
#         ), "Embedding size needs to be divisible by heads"

#         self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
#         self.c_proj = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.attn_dropout = nn.Dropout(dropout)
#         self.resid_dropout = nn.Dropout(dropout)

#     def forward(self, x, mask=None):
#         print(x.shape)
#         B, T, C = x.size()

#         # Projeção linear e divisão em q, k, v
#         q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
#         q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
#         k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
#         v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

#         # Cálculo da atenção
#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         if mask is not None:
#             att = att.masked_fill(mask == 0, float('-inf'))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_dropout(att)
#         y = att @ v

#         # Reagrupar as cabeças e aplicar a projeção de saída
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         y = self.resid_dropout(self.c_proj(y))

#         return y

class CausalSelfAttention(nn.Module):

    def __init__(self, embed_dim, heads, dropout):
        super().__init__()
        assert embed_dim % heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            embed_dim, 3 * embed_dim, bias=False)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = heads
        self.n_embd = embed_dim
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')

    def forward(self, x, mask):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # print("self.dropout", self.dropout)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
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
