import torch
from torch import Tensor
import torch.nn.functional as F


import torch
import torch.nn as nn


def perplexity(loss):
    """
    Calculates perplexity from a cross-entropy loss.

    Parameters:
        loss (float): The cross-entropy loss.

    Returns:
        float: The calculated perplexity.
    """
    return torch.exp(loss).item()


def accuracy(logits: Tensor, labels: Tensor) -> float:
    """
    Calcula a acurácia da previsão da próxima palavra.

    Parâmetros:
        logits (Tensor): Os logits do modelo. Tensor de dimensão [batch_size, vocab_size].
        labels (Tensor): Os rótulos verdadeiros. Tensor de dimensão [batch_size].

    Retorna:
        float: A acurácia das previsões.
    """
    # Aplicar softmax para converter logits em probabilidades
    probabilities = torch.softmax(logits, dim=1)

    # Encontrar a previsão (índice da maior probabilidade)
    _, predicted_indices = torch.max(probabilities, 1)

    # Comparar com os rótulos verdadeiros e calcular a acurácia
    # Converte para float para cálculo da média
    correct_predictions = (predicted_indices == labels).float()
    accuracy = correct_predictions.sum() / len(labels)

    return accuracy.item()
