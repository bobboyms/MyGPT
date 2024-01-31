import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implementa o módulo de positional encoding para uso em modelos de processamento de linguagem natural,
    como o Transformer. O positional encoding adiciona informações sobre a posição relativa dos tokens na sequência.

    Parâmetros:
    - embed_dim (int): Dimensão do espaço de embedding. Cada palavra ou token no modelo é representado
      como um vetor nesse espaço de dimensão `embed_dim`.
    - dropout (float, opcional): Probabilidade de dropout aplicada à saída do módulo. O padrão é 0.1.
    - max_len (int, opcional): Comprimento máximo da sequência que o modelo pode processar. Isso determina
      o tamanho da matriz de positional encoding pré-calculada. O padrão é 5000.

    A matriz de positional encoding é calculada uma única vez e armazenada como um buffer do módulo.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Cria um tensor para armazenar as posições dos tokens
        position = torch.arange(max_len).unsqueeze(1)

        # Calcula o termo divisor para as funções seno e cosseno
        div_term = torch.exp(torch.arange(0, embed_dim, 2)
                             * (-math.log(10000.0) / embed_dim))

        # Inicializa a matriz de positional encoding com zeros
        pe = torch.zeros(max_len, 1, embed_dim)

        # Preenche a matriz com valores seno e cosseno alternados
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Registra a matriz como um buffer constante do módulo
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica positional encoding à entrada e realiza dropout.

        Parâmetros:
        - x (torch.Tensor): Um tensor de entrada com shape [seq_len, batch_size, embedding_dim].

        Retorna:
        - Tensor: O tensor de entrada com positional encoding aplicado e dropout realizado.
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """
        Initialize the PositionalEmbedding module.

        Args:
        vocab_size (int): The size of the vocabulary.
        embed_dim (int): The embedding dimension size for the positional encoding.
        output_dim (int): The output dimension size of the embedding.
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEmbedding module.

        Args:
        x (torch.Tensor): The input tensor of token indices.

        Returns:
        torch.Tensor: The tensor containing combined word and positional embeddings.
        """

        x = self.word_embedding(x) * math.sqrt(self.embed_dim)
        x = self.positional_encoding(x)
        return x
