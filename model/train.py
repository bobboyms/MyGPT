import logging
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.metrics import perplexity
from torch.utils.data import Dataset
from typing import Tuple
from torch.optim import AdamW


# Define a semente de aleatoriedade
torch.manual_seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    filename='training.log',
                    filemode='a')


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
        # optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-3)
        self.optimizer = AdamW(self.model.parameters(),
                               lr=0.0001, weight_decay=1e-3)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.1, patience=3, verbose=True)
        self.loss_fn = nn.CrossEntropyLoss()
        self.epoch_print = 1000

    def initialize_weights(self, model: nn.Module) -> nn.Module:
        """
        Inicializa os pesos do modelo usando Kaiming Normal Initialization.

        Args:
        model (nn.Module): Modelo de rede neural.

        Returns:
        nn.Module: Modelo com pesos inicializados.
        """
        # for layer in model.modules():
        #     if isinstance(layer, (nn.Linear, nn.Conv2d)):
        #         init.kaiming_normal_(
        #             layer.weight, mode='fan_in', nonlinearity='relu')
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Embedding):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)

        return model.to(device=self.mps_device)

    def calc_loss(self, inputs, labels):
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

    def train(self, num_epochs: int = 500) -> None:
        """
        Executa o processo de treinamento, alternando entre treino e teste, e aplicando parada antecipada se necessário.
        """
        best_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(num_epochs):
            train_loss, train_perplexity = self.train_epoch()
            test_loss, test_perplexity = self.test_epoch()

            self.scheduler.step(train_loss)

            msg = f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Train Perplexity: {train_perplexity}, Test Loss: {test_loss}, Test Perplexity: {test_perplexity}'
            logging.info(msg)
            print(msg)

            # print(msg)

            # if epoch % 10 == 0 or epoch == 0:
            #     print(
            #         f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Train Acc: {train_accuracy}, Test Loss: {test_loss}, Test Acc: {test_accuracy}')

            # Verificação de parada antecipada
            if test_loss < best_loss:
                best_loss = test_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == self.n_epochs_stop:
                msg = f'Parada antecipada na época {epoch + 1}'
                print(msg)
                logging.info(msg)
                break
