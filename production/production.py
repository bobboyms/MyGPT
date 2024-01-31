import json
import torch
import torch.nn as nn
from model.model import ModelLLM
# from model.device import get_device
# from tokenizers import Tokenizer


def load_json_file(file_path: str):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Arquivo JSON não encontrado: {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Erro ao decodificar JSON: {file_path}")
        raise


class Production(nn.Module):
    def __init__(self, config, device, state_dict):
        super().__init__()
        self.model = ModelLLM(config["vocab_size"], config["embed_dim"],
                              config["num_heads"], config["dropout"], config["num_layers"])
        self.device = device
        self.model.load_state_dict(torch.load(state_dict, map_location=device))

        self.model.eval()

    def get_word_id(self, x, temperature=1.0):

        logits = self.model(x, None)

        last_logits = logits[:, -1, :]
        scaled_logits = last_logits / temperature
        probabilities = torch.softmax(scaled_logits, dim=-1)

        num_samples = 1
        sampled_indices = torch.multinomial(
            probabilities, num_samples, replacement=True)

        return sampled_indices.item()

    def forward(self, x):

        word_id = self.get_word_id(x.to(device=self.device))

        return word_id


config = load_json_file("production/config.json")
production = Production(config=config, device=torch.device('cpu'),
                        state_dict="production/production_model.pth")
