import pandas as pd
import numpy as np
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def copy_lines(arquivo_origem, arquivo_destino, num_linhas=1000):
    with open(arquivo_origem, 'r') as origem:
        linhas = [next(origem) for _ in range(num_linhas)]

    with open(arquivo_destino, 'w') as destino:
        destino.writelines(linhas)


def create_dataset_for_next_word(sequence, window_size):
    return [sequence[i:i + window_size] for i in range(len(sequence) - window_size)]


raw_dataset = "tinyshakespeare.raw"

# Inicializa um tokenizador BPE vazio
tokenizer = Tokenizer(BPE())

# Utiliza o pré-tokenizador Whitespace para dividir o texto em palavras
tokenizer.pre_tokenizer = Whitespace()

# Configura o treinador
trainer = BpeTrainer(vocab_size=60000, special_tokens=[
                     "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# Lista de arquivos a serem utilizados no treinamento
# Substitua pelo caminho do seu arquivo de dados
files = ["raw_text.txt", "test_raw_text.txt"]

# Treina o tokenizador
tokenizer.train(files, trainer)

# Salva o tokenizador treinado para uso futuro
# Escolha um caminho apropriado
tokenizer.save("tokenizer.json")


with open("raw_text.txt", 'r') as f:
    train_data = f.read()

with open("test_raw_text.txt", 'r') as f:
    test_data = f.read()

train_output = tokenizer.encode(train_data)
test_output = tokenizer.encode(test_data)


window_size = 65
train_dataset = create_dataset_for_next_word(train_output.ids, window_size)
test_dataset = create_dataset_for_next_word(test_output.ids, window_size)

train_df = pd.DataFrame(np.array(train_dataset))
test_df = pd.DataFrame(np.array(test_dataset))

train_df.to_csv("train.csv", index=False, header=False)
test_df.to_csv("test.csv", index=False, header=False)

copy_lines("train.csv", "little_train.csv", num_linhas=10000)
copy_lines("test.csv", "little_test.csv", num_linhas=3000)
