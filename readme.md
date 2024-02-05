# MyLLM

Esta é uma implementação da arquitetura do ChatGPT desenvolvida do zero por mim. O objetivo é prever a próxima palavra, ou seja, você começa com uma palavra ou frase, e o modelo completará o texto. Este é o princípio de treinamento do GPT.

Siga os passos abaixo para executar o GPT em seu computador. É importante notar que, sem uma GPU, você não conseguirá treinar o modelo, pois arquiteturas como o GPT são complexas, com muitas camadas e parâmetros. Este modelo, por exemplo, possui 12 cabeças de atenção. Caso não tenha uma GPU, minha sugestão é alugar uma instância EC2 com GPU na AWS e treinar por um dia. Após o treinamento, o modelo pode ser executado usando a GPU de seu computador.

Para a implementação, utilizei Python3 e a biblioteca Pytorch 2.0.

**Referências para a implementação:**

- https://www.mikecaptain.com/resources/pdf/GPT-1.pdf
- https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf
- https://arxiv.org/pdf/1706.03762.pdf
- https://github.com/karpathy/nanoGPT

Clone o repositório, faça modificações no código, estude profundamente e, mais importante, aprenda como os modelos de LLM funcionam por trás das cortinas para se destacar profissionalmente.

## Informações Importantes

O dataset utilizado é uma versão reduzida do que normalmente se usa em LLMs, pois indivíduos comuns não têm acesso a supercomputadores. É possível que ocorra overfitting durante o treinamento, mesmo usando técnicas de warmup para o Learning Rate, devido à falta de representatividade adequada do conjunto de teste em relação ao conjunto de treinamento. Isso pode resultar em palavras no conjunto de teste que não estão presentes no treinamento, levando o modelo a "alucinar", embora ainda seja capaz de gerar textos coerentes.

## Bibliotecas Necessárias

Instale as seguintes bibliotecas para executar o código:

```bash
pip install torch torchvision torchaudio
pip install datasets
pip install tokenizers
```

## Preparando o Dataset

Antes de iniciar o treinamento, prepare o dataset para treinar o tokenizador e criar os conjuntos de treinamento e teste. Utilizei o tokenizador da Hugging Face com o algoritmo de tokenização BPE. O dataset inclui frases em português extraídas da Wikipedia.

Prepare o dataset assim:

```bash
cd dataset/portuguese/
python3 process.py
```

## Iniciando o Treinamento

O treinamento está configurado para ocorrer durante 5000 épocas, mas não é necessário completar todas. Após cada época, o arquivo `model.pth`, que contém o modelo treinado para uso em produção, é salvo na pasta 'production'.

```bash
python3 trainer.py
```

## Gerando Palavras

Após o treinamento, você pode gerar palavras com o modelo. Execute o comando abaixo, passando uma palavra ou frase inicial, e o modelo gerará mais 40 palavras a partir dela.

```bash
python3 sample.py "Marte é um planeta"
```