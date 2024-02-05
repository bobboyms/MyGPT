import torch
import sys
from tokenizers import Tokenizer
from production.production import production
from model.device import get_device

# Carregar o tokenizador treinado
tokenizer = Tokenizer.from_file("dataset/portuguese/tokenizer.json")

context_size = 64

# Texto de entrada
text = sys.argv[1]

# Convertendo o texto em IDs de palavras
word_ids = tokenizer.encode(text).ids

# Convertendo para tensor e ajustando a dimensão
word_ids_tensor = torch.tensor([word_ids], dtype=torch.long)
for i in range(40):
    # Atualizando o tamanho
    size = word_ids_tensor.size(1)  # Obter o tamanho da segunda dimensão

    # Limitando o tamanho do contexto
    if size > context_size:
        word_ids_tensor = word_ids_tensor[:, -context_size:]

    # Chamando a função de produção
    word_id = production(word_ids_tensor)

    # Concatenando o novo word_id
    word_ids_tensor = torch.cat(
        (word_ids_tensor, torch.tensor([[word_id]], dtype=torch.long)), dim=1)

# Decodificando os IDs de palavras para texto
generated_text = tokenizer.decode(word_ids_tensor[0].tolist())

print(generated_text)
