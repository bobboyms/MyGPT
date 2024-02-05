import math
import logging
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from model.metrics import perplexity
from torch.utils.data import Dataset
from typing import Tuple
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

torch.manual_seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    filename='training.log',
                    filemode='a')


class CustomLRScheduler(_LRScheduler):
    """
    Um agendador de taxa de aprendizagem que ajusta a taxa de aprendizagem seguindo um aquecimento linear inicial,
    seguido por um decaimento cosseno até uma taxa mínima especificada.

    Parâmetros:
        optimizer (Optimizer): O otimizador para o qual a taxa de aprendizagem será ajustada.
        warmup_iters (int): O número de iterações para o aquecimento linear.
        lr_decay_iters (int): O número total de iterações após as quais a taxa de aprendizagem decai até min_lr.
        learning_rate (float): A taxa de aprendizagem inicial/máxima.
        min_lr (float): A taxa de aprendizagem mínima após o decaimento.
        last_epoch (int): A última época indexada pelo agendador. O padrão é -1.

    Retorna:
        None
    """

    def __init__(self, optimizer: Optimizer, warmup_iters: int, lr_decay_iters: int,
                 learning_rate: float, min_lr: float, last_epoch: int = -1) -> None:
        self.warmup_iters: int = warmup_iters
        self.lr_decay_iters: int = lr_decay_iters
        self.learning_rate: float = learning_rate
        self.min_lr: float = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """
        Calcula e retorna a nova taxa de aprendizagem para cada grupo de parâmetros do otimizador,
        com base na estratégia de aquecimento e decaimento definida.

        Esse código foi inspirado no código de Andrej karpathy
        https://github.com/karpathy/nanoGPT/blob/master/train.py#L228

        Retorna:
            list[float]: Uma lista contendo a nova taxa de aprendizagem para cada grupo de parâmetros.
        """
        if self._step_count < self.warmup_iters:
            lr = self.learning_rate * self._step_count / self.warmup_iters
        elif self._step_count > self.lr_decay_iters:
            lr = self.min_lr
        else:
            decay_ratio = (self._step_count - self.warmup_iters) / \
                (self.lr_decay_iters - self.warmup_iters)
            assert 0 <= decay_ratio <= 1, "Decay ratio should be between 0 and 1."
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.learning_rate - self.min_lr)

        return [lr for _ in self.optimizer.param_groups]


class Trainer:
    def __init__(self, model: nn.Module, train_dataset: Dataset, test_dataset: Dataset, device: torch.device, batch_size: int) -> None:
        """
        Inicializa o objeto Trainer.

        Args:
            model (nn.Module): Modelo de rede neural para treinamento.
            train_dataset (Dataset): Conjunto de dados de treinamento.
            test_dataset (Dataset): Conjunto de dados de teste.
            device (torch.device): Dispositivo onde o modelo será treinado (CPU ou GPU).
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.mps_device = device

        # Parâmetros
        self.n_epochs_stop = 30

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=True)

        # Inicialização do modelo
        self.model = self.initialize_weights(model)

        # Configuração do Otimizador e Scheduler
        self.optimizer = AdamW(self.model.parameters(),
                               lr=3.6e-05, weight_decay=1e-3)  # 3.6e-05

        iterations = self.train_dataset.__len__() // batch_size
        warmup_iters = 2000  # 5 * iterations
        lr_decay_iters = 100000  # iterations * batch_size
        self.scheduler = CustomLRScheduler(
            self.optimizer, warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters, learning_rate=6e-4, min_lr=3.6e-6)

        self.loss_fn = nn.CrossEntropyLoss()
        self.epoch_print = 1000

    def initialize_weights(self, model: nn.Module) -> nn.Module:
        """
        Inicializa os pesos de um modelo Transformer.

        Parâmetros:
            model (torch.nn.Module): Modelo Transformer a ser inicializado.

        Retorna:
            torch.nn.Module: Modelo com pesos inicializados.
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Inicializa os pesos da camada linear com uma distribuição normal
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                # Inicializa os bias da camada linear com zero, se houver
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Inicializa os pesos da camada de embedding com uma distribuição normal
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                # Inicializa os pesos da LayerNorm com 1 e bias com 0
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        return model.to(device=self.mps_device)

    def calc_loss(self, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(inputs, None)

        logits_flattened = logits.view(-1, logits.size(-1))
        labels_flattened = labels.view(-1)
        loss = self.loss_fn(logits_flattened, labels_flattened)

        return loss, perplexity(loss)

    def train_epoch(self) -> Tuple[float, float]:
        """
        Realiza o treinamento de uma época.

        Returns:
            Tuple[float, float]: Média de perda e acurácia no conjunto de treinamento.
        """

        epoch = 0
        total_loss = 0
        total_perplexity = 0
        self.model.train()
        for batch in self.train_loader:
            inputs, labels = batch
            inputs = inputs.to(device=self.mps_device)
            labels = labels.to(device=self.mps_device)

            self.optimizer.zero_grad()

            loss, perplexity = self.calc_loss(inputs, labels)
            total_loss += loss.item()
            total_perplexity += perplexity

            loss.backward()

            # Aplica gradient clipping para evitar o problema de explosão de gradientes
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            if epoch % self.epoch_print == 0:
                torch.save(self.model.state_dict(), 'production/model.pth')
                # print(
                #     f'Train Epoch: {epoch + 1}/{len(self.train_loader)}, Loss: {loss.item()}, Perplexity: {perplexity}')
            epoch += 1

        avg_loss = total_loss / len(self.train_loader)
        avg_perplexity = total_perplexity / len(self.train_loader)
        return avg_loss, avg_perplexity

    def test_epoch(self) -> Tuple[float, float]:
        """
        Avalia o modelo em uma época no conjunto de teste.

        Returns:
        Tuple[float, float]: Média de perda e acurácia no conjunto de teste.
        """
        epoch = 0
        total_loss = 0
        total_perplexity = 0
        self.model.eval()
        for batch in self.test_loader:
            with torch.no_grad():
                inputs, labels = batch
                inputs = inputs.to(device=self.mps_device)
                labels = labels.to(device=self.mps_device, dtype=torch.long)

                loss, perplexity = self.calc_loss(inputs, labels)
                total_loss += loss.item()
                total_perplexity += perplexity

                if epoch % self.epoch_print == 0:
                    print(
                        f'Test Epoch: {epoch + 1}/{len(self.train_loader)}, Loss: {loss.item()}, Perplexity: {perplexity}')

                epoch += 1
        avg_loss = total_loss / len(self.train_loader)
        avg_perplexity = total_perplexity / len(self.train_loader)
        return avg_loss, avg_perplexity

    def train(self, num_epochs: int = 5000) -> None:
        """
        Executa o processo de treinamento, alternando entre treino e teste, 
        e aplicando parada antecipada se necessário.
        """
        best_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(num_epochs):
            train_loss, train_perplexity = self.train_epoch()
            test_loss, test_perplexity = self.test_epoch()

            self.scheduler.step()

            msg = f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Train Perplexity: {train_perplexity}, Test Loss: {test_loss}, Test Perplexity: {test_perplexity}'
            logging.info(msg)
            print(msg)

            for group in self.optimizer.param_groups:
                print("Current learning rate:", group['lr'])

            # Verificação de parada antecipada
            # if test_loss < best_loss:
            #     best_loss = test_loss
            #     epochs_no_improve = 0
            # else:
            #     epochs_no_improve += 1

            # if epochs_no_improve == self.n_epochs_stop:
            #     msg = f'Parada antecipada na época {epoch + 1}'
            #     print(msg)
            #     logging.info(msg)
            #     break
