import copy
from time import time
from typing import Any
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import Verbosity, ProgressBar, EarlyStop, Callback


class _History:
    def __init__(self, metrics: list[str] = ['loss', 'accuracy']) -> None:
        self.__metrics = metrics

        self.__history = {metric: [] for metric in self.__metrics}

    def __str__(self, prefix: str = '', precision: int = 3) -> str:
        results = []

        for metric in self.__metrics:
            results.append(f'{prefix}_{metric}: {self.__history[metric][-1]:.{precision}f}')

        return ', '.join(results)

    def __getitem__(self, idx: int) -> dict[str, int | float]:
        results = {}

        for metric in self.__metrics:
            results[metric] = self.__history[metric][idx]

        return results

    def reset(self) -> None:
        for key in self.__history.keys():
            self.__history[key].clear()

    def log(self, key: str, value: Any) -> None:
        self.__history[key].append(value)

    def summary(self) -> dict[str, int | float]:
        for metric in self.__metrics:
            self.__history[metric].append(sum(self.__history[metric]) / len(self.__history[metric]))

        return self.__getitem__(-1)


class _BaseRunner(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.__device = torch.device('cuda' if cuda.is_available() else 'cpu')

    @property
    def device(self) -> torch.device:
        return self.__device

    @device.setter
    def device(self, value: str) -> None:
        self.__device = torch.device(value)

    @property
    @abstractmethod
    def weights(self) -> dict[str, Any]:
        raise NotImplementedError('weights not implemented')


class Runner(_BaseRunner):
    def __init__(self, net: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module | nn.modules.loss._Loss, categorical: bool = True, verbose: Verbosity = Verbosity.PROGRESS, elapsed_time: bool = True) -> None:
        super(Runner, self).__init__()

        self.__net = net
        self.__optimizer = optimizer
        self.__criterion = criterion

        self.__categorical = categorical

        self.__verbose = verbose
        self.__elapsed_time = elapsed_time

        self.__history = _History(metrics=['loss'] + (['accuracy'] if self.__categorical else []))

        self.__net = self.__net.to(self.device)

    def __forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)

        output = self.__net.forward(x)
        y_hat = torch.argmax(output, dim=-1) if self.__categorical else output

        return output, y_hat

    def __metrics(self, output: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
        output = output.to(self.device)
        y_hat = y_hat.to(self.device)
        y = y.to(self.device)

        try:
            metrics = self.metrics(output, y_hat, y)

        except NotImplementedError:
            metrics = []

            running_loss = self.__criterion.forward(output, y)
            self.__history.log('loss', running_loss.item())
            metrics.append(running_loss)

            if self.__categorical:
                running_accuracy = torch.sum(y_hat == y) / torch.numel(y)
                self.__history.log('accuracy', running_accuracy.item())
                metrics.append(running_accuracy)

        return metrics

    def metrics(self, output: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError('metrics not implemented')

    def train(self, epochs: int, train_loader: DataLoader, valid_loader: DataLoader | None = None, earlystop: EarlyStop | None = None, callbacks: list[Callback] | None = None) -> None:
        epoch_length = len(str(epochs))

        for epoch in range(epochs):
            self.__net.train()

            start = time()

            for i, (x, y) in enumerate(train_loader):
                output, y_hat = self.__forward(x)
                running_loss, *_ = self.__metrics(output, y_hat, y)

                self.__optimizer.zero_grad()
                running_loss.backward()
                self.__optimizer.step()

                if self.__verbose in [Verbosity.PROGRESS]:
                    prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
                    postfix = self.__history.__str__(prefix='train')
                    ProgressBar.show(prefix, postfix, i, len(train_loader))

            stop = time()

            metrics = self.__history.summary()

            if self.__verbose in [Verbosity.FINAL, Verbosity.PROGRESS]:
                prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
                postfix = self.__history.__str__(prefix='train') + (f' ({(stop - start):.3f} secs)' if self.__elapsed_time else '')
                ProgressBar.show(prefix, postfix, len(train_loader), len(train_loader), show_progress=(self.__verbose == Verbosity.PROGRESS), freeze=(valid_loader is not None), newline=(valid_loader is None))

            self.__history.reset()

            if valid_loader:
                metrics = self.validate(valid_loader)

            if earlystop:
                earlystop.step(metrics)

                if earlystop.stop():
                    print(f'Stop by Early Stopping check with metric `{earlystop.monitor}` (best: {earlystop.best}, patience: {earlystop.patience})')

                    break

            exclude = ['_', 'self', 'callbacks', 'callback', 'exclude']

            if callbacks is not None:
                for callback in callbacks:
                    if (epoch + 1) % callback.interval == 0:
                        callback(**{key: value for key, value in locals().items() if key not in exclude} | {'runner': self})

    @torch.no_grad()
    def validate(self, valid_loader: DataLoader) -> dict[str, int | float]:
        self.__net.eval()

        start = time()

        for i, (x, y) in enumerate(valid_loader):
            output, y_hat = self.__forward(x)
            _ = self.__metrics(output, y_hat, y)

            if self.__verbose in [Verbosity.PROGRESS]:
                prefix = ''
                postfix = self.__history.__str__(prefix='valid')
                ProgressBar.show(prefix, postfix, i, len(valid_loader), show_progress=False)

        stop = time()

        metrics = self.__history.summary()

        if self.__verbose in [Verbosity.FINAL, Verbosity.PROGRESS]:
            prefix = ''
            postfix = self.__history.__str__(prefix='valid') + (f' ({(stop - start):.3f} secs)' if self.__elapsed_time else '')
            ProgressBar.show(prefix, postfix, len(valid_loader), len(valid_loader), show_progress=False, newline=True)

        self.__history.reset()

        return metrics

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> dict[str, int | float]:
        self.__net.eval()

        start = time()

        for i, (x, y) in enumerate(test_loader):
            output, y_hat = self.__forward(x)
            _ = self.__metrics(output, y_hat, y)

            if self.__verbose in [Verbosity.PROGRESS]:
                prefix = 'Test'
                postfix = self.__history.__str__(prefix='test')
                ProgressBar.show(prefix, postfix, i, len(test_loader))

        stop = time()

        metrics = self.__history.summary()

        if self.__verbose in [Verbosity.FINAL, Verbosity.PROGRESS]:
            prefix = 'Test'
            postfix = self.__history.__str__(prefix='test') + (f' ({(stop - start):.3f} secs)' if self.__elapsed_time else '')
            ProgressBar.show(prefix, postfix, len(test_loader), len(test_loader), show_progress=(self.__verbose == Verbosity.PROGRESS), newline=True)

        self.__history.reset()

        return metrics

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> tuple[torch.Tensor, ...]:
        self.__net.eval()

        start = time()

        ys = []
        y_hats = []

        for i, (x, y) in enumerate(data_loader):
            _, y_hat = self.__forward(x)

            ys.append(y)
            y_hats.append(y_hat)

            if self.__verbose in [Verbosity.PROGRESS]:
                prefix = 'Evaluate'
                postfix = ''
                ProgressBar.show(prefix, postfix, i, len(data_loader))

        y = torch.concat(ys, dim=0).cpu()
        y_hat = torch.concat(y_hats, dim=0).cpu()

        stop = time()

        if self.__verbose in [Verbosity.FINAL, Verbosity.PROGRESS]:
            prefix = 'Evaluate'
            postfix = f'elapsed time: {(stop - start):.3f}' if self.__elapsed_time else ''
            ProgressBar.show(prefix, postfix, len(data_loader), len(data_loader), show_progress=(self.__verbose == Verbosity.PROGRESS), newline=True)

        self.__history.reset()

        return y, y_hat

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.__net.eval()

        start = time()

        _, y_hat = self.__forward(x)
        y_hat = y_hat.cpu()

        stop = time()

        if self.__verbose in [Verbosity.FINAL, Verbosity.PROGRESS]:
            prefix = 'Predict'
            postfix = f'elapsed time: {(stop - start):.3f}' if self.__elapsed_time else ''
            ProgressBar.show(prefix, postfix, 1, 1, show_progress=(self.__verbose == Verbosity.PROGRESS), newline=True)

        return y_hat

    @property
    @torch.no_grad()
    def weights(self) -> dict[str, Any]:
        return {'net': copy.deepcopy(self.__net).cpu()}

    @property
    @torch.no_grad()
    def optimizer(self) -> optim.Optimizer:
        return self.__optimizer

    @optimizer.setter
    @torch.no_grad()
    def optimizer(self, value: optim.Optimizer) -> None:
        self.__optimizer = value

    @property
    @torch.no_grad()
    def criterion(self) -> nn.Module | nn.modules.loss._Loss:
        return self.__criterion

    @criterion.setter
    @torch.no_grad()
    def criterion(self, value: nn.Module | nn.modules.loss._Loss) -> None:
        self.__criterion = value
