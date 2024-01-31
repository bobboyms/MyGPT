import torch
from torch.utils.data import Dataset
import pandas as pd


class NextWordPredictionDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, header=None)
        self.data = df.values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Separa a sequência de entrada e a próxima palavra
        # Todos menos o último elemento
        sequence_minus_last = self.data[idx][:-1]
        # Todos menos o primeiro elemento
        sequence_minus_first = self.data[idx][1:]
        return (torch.tensor(sequence_minus_last, dtype=torch.long),
                torch.tensor(sequence_minus_first, dtype=torch.long))
