import json
import torch
from model.device import get_device
from model.train import Trainer
from model.model import ModelLLM
from model.dataset import NextWordPredictionDataset
from tokenizers import Tokenizer


if __name__ == "__main__":

    tokenizer = Tokenizer.from_file("dataset/portuguese/tokenizer.json")
    vocab = tokenizer.get_vocab()

    vocab_size = len(vocab)
    print("Tamanho do Vocabulário:", vocab_size)

    train_dataset = NextWordPredictionDataset(
        'dataset/portuguese/train.csv')
    test_dataset = NextWordPredictionDataset(
        'dataset/portuguese/test.csv')

    embed_dim = 240
    num_heads = 6
    dropout = 0.2
    num_layers = 12

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
    batch_size = 170
    print(f"Batch Size: {batch_size}")
    trainer = Trainer(model, train_dataset, test_dataset, device, batch_size)
    trainer.train(num_epochs=200)

