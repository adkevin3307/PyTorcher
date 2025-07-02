import copy
from time import time
from typing import Any
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import ProgressBar, Callback


class _History:
    def __init__(self, metrics: list[str] = ['loss', 'accuracy'], additional_metrics: list[str] | None = None) -> None:
        self.__metrics = metrics
        self.__additional_metrics = [] if additional_metrics is None else additional_metrics

        self.__history = {'count': [], 'loss': [], 'correct': [], 'accuracy': []}

        for key in self.__additional_metrics:
            self.__history[key] = []

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

        if len(self.__history['count']) == len(self.__history['correct']) and len(self.__history['count']) > len(self.__history['accuracy']):
            self.__history['accuracy'].append(self.__history['correct'][-1] / self.__history['count'][-1])

    def summary(self) -> dict[str, int | float]:
        _count = sum(self.__history['count'])

        if _count == 0:
            _count = 1

        _loss = sum(self.__history['loss']) / len(self.__history['loss'])
        _correct = sum(self.__history['correct'])
        _accuracy = _correct / _count

        self.__history['count'].append(_count)
        self.__history['loss'].append(_loss)
        self.__history['correct'].append(_correct)
        self.__history['accuracy'].append(_accuracy)

        for key in self.__additional_metrics:
            _value = sum(self.__history[key]) / len(self.__history[key])

            self.__history[key].append(_value)

        return self[-1]


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


class EarlyStop:
    def __init__(self, monitor: str, patience: int, greater_is_better: bool = False) -> None:
        self.__count = 0
        self.__best = None

        self.__monitor = monitor
        self.__patience = patience
        self.__greater_is_better = greater_is_better

    def __better(self, value: Any) -> bool:
        if self.__greater_is_better:
            return value > self.__best

        return value < self.__best

    @property
    def best(self) -> Any:
        return self.__best

    @property
    def monitor(self) -> str:
        return self.__monitor

    @property
    def patience(self) -> int:
        return self.__patience

    def step(self, metrics: dict[str, Any]) -> None:
        if self.__monitor not in metrics:
            return

        if self.__best is None or self.__better(metrics[self.__monitor]):
            self.__count = 0
            self.__best = metrics[self.__monitor]

            return

        self.__count += 1

    def stop(self) -> bool:
        return self.__count > self.__patience


class Runner(_BaseRunner):
    def __init__(self, net: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module | nn.modules.loss._Loss, additional_metrics: list[str] | None = None) -> None:
        super(Runner, self).__init__()

        self.__history = _History(metrics=['loss'] + ([] if additional_metrics is None else additional_metrics), additional_metrics=additional_metrics)

        self.__net = net
        self.__optimizer = optimizer
        self.__criterion = criterion

        self.__net = self.__net.to(self.__device)

    def __step(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.__device)
        y = y.to(self.__device)

        output = self.__net.forward(x)
        y_hat = output

        running_loss = self.__criterion.forward(output, y)

        self.__history.log('count', torch.numel(y))
        self.__history.log('correct', torch.sum(y_hat == y).item())

        self.__history.log('loss', running_loss.item())

        return (y_hat, running_loss)

    def train(self, epochs: int, train_loader: DataLoader, valid_loader: DataLoader | None = None, earlystop: EarlyStop | None = None, callbacks: list[Callback] | None = None) -> None:
        epoch_length = len(str(epochs))

        for epoch in range(epochs):
            self.__net.train()

            start = time()

            for i, (x, y) in enumerate(train_loader):
                _, running_loss = self.__step(x, y)

                self.__optimizer.zero_grad()
                running_loss.backward()
                self.__optimizer.step()

                prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
                postfix = self.__history.__str__(prefix='train')
                ProgressBar.show(prefix, postfix, i, len(train_loader))

            stop = time()

            metrics = self.__history.summary()

            prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
            postfix = self.__history.__str__(prefix='train') + f' ({(stop - start):.3f} secs)'
            ProgressBar.show(prefix, postfix, len(train_loader), len(train_loader), freeze=(valid_loader is not None), newline=(valid_loader is None))

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
                        callback(**{key: value for key, value in locals().items() if key not in exclude})

    @torch.no_grad()
    def validate(self, valid_loader: DataLoader) -> dict[str, int | float]:
        self.__net.eval()

        start = time()

        for i, (x, y) in enumerate(valid_loader):
            _ = self.__step(x, y)

            prefix = ''
            postfix = self.__history.__str__(prefix='valid')
            ProgressBar.show(prefix, postfix, i, len(valid_loader), show_progress=False)

        stop = time()

        metrics = self.__history.summary()

        prefix = ''
        postfix = self.__history.__str__(prefix='valid') + f' ({(stop - start):.3f} secs)'
        ProgressBar.show(prefix, postfix, len(valid_loader), len(valid_loader), show_progress=False, newline=True)

        self.__history.reset()

        return metrics

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> dict[str, int | float]:
        self.__net.eval()

        for i, (x, y) in enumerate(test_loader):
            _ = self.__step(x, y)

            prefix = 'Test'
            postfix = self.__history.__str__(prefix='test')
            ProgressBar.show(prefix, postfix, i, len(test_loader))

        metrics = self.__history.summary()

        prefix = 'Test'
        postfix = self.__history.__str__(prefix='test')
        ProgressBar.show(prefix, postfix, len(test_loader), len(test_loader), newline=True)

        self.__history.reset()

        return metrics

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> tuple[torch.Tensor, ...]:
        self.__net.eval()

        start = time()

        ys = []
        y_hats = []

        for i, (x, y) in enumerate(data_loader):
            output, _ = self.__step(x, y)
            y_hat = output

            ys.append(y)
            y_hats.append(y_hat)

            ProgressBar.show('Evaluate', '', i, len(data_loader))

        y = torch.concat(ys, dim=0).cpu()
        y_hat = torch.concat(y_hats, dim=0).cpu()

        stop = time()

        ProgressBar.show('Evaluate', f'elapsed time: {(stop - start):.3f}', len(data_loader), len(data_loader), newline=True)

        self.__history.reset()

        return y, y_hat

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.__net.eval()

        x = x.to(self.__device)
        output = self.__net.forward(x)
        y_hat = output.cpu()

        return y_hat

    @property
    @torch.no_grad()
    def weights(self) -> dict[str, Any]:
        return {'net': copy.deepcopy(self.__net).cpu()}
