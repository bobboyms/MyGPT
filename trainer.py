import json
import torch
from model.device import get_device
from model.train import Trainer
from model.model import ModelLLM
from model.dataset import NextWordPredictionDataset
from tokenizers import Tokenizer


if __name__ == "__main__":

    tokenizer_file = "dataset/portuguese/tokenizer.json"

    tokenizer = Tokenizer.from_file(tokenizer_file)
    vocab = tokenizer.get_vocab()

    vocab_size = len(vocab)
    print("Tamanho do Vocabulário:", vocab_size)

    train_dataset = NextWordPredictionDataset(
        'dataset/portuguese/little_train.csv')
    test_dataset = NextWordPredictionDataset(
        'dataset/portuguese/little_test.csv')

    embed_dim = 240
    num_heads = 12
    dropout = 0.3
    num_layers = 6

    data = {
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dropout": dropout,
    }

    print("***** CONFIGS ****")
    print(data)

    # Escrever os dados em um arquivo JSON
    with open('production/config.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    model = ModelLLM(vocab_size, embed_dim, num_heads, dropout, num_layers)

    device = get_device()
    print(f"Device: {device}")
    batch_size = 20
    print(f"Batch Size: {batch_size}")
    trainer = Trainer(model, train_dataset, test_dataset, device, batch_size)
    trainer.train()
