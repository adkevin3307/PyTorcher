from abc import ABC, abstractmethod
from time import time
from typing import Any, Optional, Sequence, Dict, Tuple, Union, TypeVar
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from utils import ProgressBar


class _History:
    def __init__(self, metrics: Sequence[str] = ['loss', 'accuracy'], additional_keys: Sequence[str] = []) -> None:
        self.metrics = metrics
        self.additional_keys = additional_keys

        self._history = {
            'count': [],
            'loss': [],
            'correct': [],
            'accuracy': []
        }

        for key in self.additional_keys:
            self._history[key] = []

    def __str__(self) -> str:
        results = []

        for metric in self.metrics:
            results.append(f'{metric}: {self._history[metric][-1]:.3f}')

        return ', '.join(results)

    def __getitem__(self, idx: int) -> Dict[str, Union[int, float]]:
        results = {}

        for metric in self.metrics:
            results[metric] = self._history[metric][idx]

        return results

    def reset(self) -> None:
        for key in self._history.keys():
            self._history[key].clear()

    def log(self, key: str, value: Any) -> None:
        self._history[key].append(value)

        if len(self._history['count']) == len(self._history['correct']) and len(self._history['count']) > len(self._history['accuracy']):
            self._history['accuracy'].append(self._history['correct'][-1] / self._history['count'][-1])

    def summary(self) -> None:
        _count = sum(self._history['count'])
        if _count == 0:
            _count = 1

        _loss = sum(self._history['loss']) / len(self._history['loss'])
        _correct = sum(self._history['correct'])
        _accuracy = _correct / _count

        self._history['count'].append(_count)
        self._history['loss'].append(_loss)
        self._history['correct'].append(_correct)
        self._history['accuracy'].append(_accuracy)

        for key in self.additional_keys:
            _value = sum(self._history[key]) / len(self._history[key])
            self._history[key].append(_value)


class _BaseRunner(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.device = torch.device('cuda' if cuda.is_available() else 'cpu')

    @property
    @abstractmethod
    def weights(self) -> None:
        raise NotImplementedError('weights not implemented')


NetBase = TypeVar('NetBase', bound=nn.Module)
CriterionBase = TypeVar('CriterionBase', bound=nn.modules.loss._Loss)
class ExampleRunner(_BaseRunner):
    def __init__(self, net: NetBase, optimizer: optim.Optimizer, criterion: CriterionBase, log_dir: str = 'log') -> None:
        super(ExampleRunner, self).__init__()

        self.history = _History(metrics=['loss', 'accuracy'])
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = {'train': 0, 'test': 0, 'epoch': 0}

        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def _step(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        y = y.to(self.device)

        output = self.net.forward(x)
        y_hat = torch.argmax(output, dim=-1)

        running_loss = self.criterion.forward(output, y)

        self.history.log('count', torch.numel(y))
        self.history.log('loss', running_loss.item())
        self.history.log('correct', torch.sum(y_hat == y).item())

        return (y_hat, running_loss)

    def _record(self, tag: str, info: Optional[dict] = None, updatable: bool = True) -> None:
        _log = self.history[-1]

        for key, value in _log.items():
            self.writer.add_scalar(f'{key} / {tag}', value, global_step=self.global_step[tag])

        if info is not None:
            for key, value in info.items():
               self.writer.add_scalar(f'{key} / {tag}', value, global_step=self.global_step[tag])

        if updatable:
            self.global_step[tag] += 1

    def train(self, epochs: int, train_loader: DataLoader, valid_loader: Optional[DataLoader] = None, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None) -> None:
        epoch_length = len(str(epochs))

        for epoch in range(epochs):
            self.net.train()

            for i, (x, y) in enumerate(train_loader):
                _, running_loss = self._step(x, y)

                self.optimizer.zero_grad()
                running_loss.backward()
                self.optimizer.step()

                prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
                postfix = str(self.history)
                ProgressBar.show(prefix, postfix, i, len(train_loader))

                self._record('train')

            self._record('epoch', info={'LR': self.optimizer.param_groups[0]['lr']})
            self.history.summary()

            prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
            postfix = str(self.history)
            ProgressBar.show(prefix, postfix, len(train_loader), len(train_loader), newline=True)

            self.history.reset()

            if valid_loader:
                self.test(valid_loader)

            if scheduler:
                scheduler.step()

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> None:
        self.net.eval()

        for i, (x, y) in enumerate(test_loader):
            _ = self._step(x, y)

            prefix = 'Test'
            postfix = str(self.history)
            ProgressBar.show(prefix, postfix, i, len(test_loader))

            self._record('test')

        self.history.summary()

        prefix = 'Test'
        postfix = str(self.history)
        ProgressBar.show(prefix, postfix, len(test_loader), len(test_loader), newline=True)

        self.history.reset()

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> torch.Tensor:
        self.net.eval()

        start = time()

        y_hat = []

        for i, (x, ) in enumerate(data_loader):
            output, _ = self._step(x, torch.zeros(1))
            y_hat.append(output)

            ProgressBar.show('Evaluate', '', i, len(data_loader))

        y_hat = torch.concat(y_hat, dim=0).cpu()

        stop = time()

        ProgressBar.show('Evaluate', f'elapsed time: {(stop - start):.3f}', len(data_loader), len(data_loader), newline=True)

        return y_hat

    @property
    @torch.no_grad()
    def weights(self) -> Dict[str, Any]:
        return {self.net.__class__.__name__: self.net}
