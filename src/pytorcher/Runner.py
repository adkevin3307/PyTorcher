import copy
from tqdm import tqdm
from typing import Any
from functools import partial
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .utils import Verbosity, Callback


class _History:
    def __init__(self, metrics: list[str] = ['loss', 'accuracy']) -> None:
        self.__history = {metric: [] for metric in metrics}

    def __str__(self) -> str:
        return self.convert(self.summary(), prefix='', precision=3)

    def __getitem__(self, idx: int) -> dict[str, int | float]:
        metrics = {}

        for key, value in self.__history.items():
            metrics[key] = value[idx]

        return metrics

    def convert(self, metrics: dict[str, int | float], prefix: str = '', precision: int = 3) -> str:
        return ', '.join([f'{f"{prefix}_" if len(prefix) > 0 else ""}{key}: {value:.{precision}f}' for key, value in metrics.items()])

    def reset(self) -> None:
        for key in self.__history.keys():
            self.__history[key].clear()

    def log(self, key: str, value: Any) -> None:
        self.__history[key].append(value)

    def summary(self) -> dict[str, int | float]:
        metrics = {}

        for key, value in self.__history.items():
            metrics[key] = sum(value) / len(value)

        return metrics


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
    def __init__(
        self,
        net: nn.Module,
        optimizer: optim.Optimizer | None = None,
        criterion: nn.Module | nn.modules.loss._Loss | None = None,
        categorical: bool = True,
        verbose: Verbosity = Verbosity.DETAIL,
        elapsed_time: bool = True,
        checkpoints: str = 'checkpoints',
    ) -> None:

        super(Runner, self).__init__()

        self.__net = net
        self.__optimizer = optimizer
        self.__criterion = criterion

        self.__categorical = categorical

        self.__verbose = verbose

        self.__metrics = ['loss'] + (['accuracy'] if self.__categorical else [])
        self.__history = _History(metrics=self.__metrics)

        self.__writer = SummaryWriter(log_dir=checkpoints)
        self.__record_count = {'train': 0, 'validate': 0, 'test': 0}

        self.__progress_bar = partial(tqdm, ascii=' =', bar_format='{desc}, [{bar:20}] {percentage:6.2f}%{postfix}' + (' ({elapsed_s:.3f} secs)' if elapsed_time else ''))

        self.__net = self.__net.to(self.device)

    def __forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)

        output = self.__net.forward(x)
        y_hat = torch.argmax(output, dim=-1) if self.__categorical else output

        return output, y_hat

    def __step(self, output: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = output.to(self.device)
        y_hat = y_hat.to(self.device)
        y = y.to(self.device)

        try:
            result = self.step(output, y_hat, y)

        except Exception:
            assert self.__criterion is not None

            result = {}

            running_loss = self.__criterion.forward(output, y)
            result['loss'] = running_loss

            if self.__categorical:
                running_accuracy = torch.sum(y_hat == y) / torch.numel(y)
                result['accuracy'] = running_accuracy

        for key, value in result.items():
            self.__history.log(key, value.item())

        return tuple(result.values())

    def step(self, output: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError('step not implemented')

    def train(self, epochs: int, data_loader: DataLoader, valid_loader: DataLoader | None = None, callbacks: list[Callback] | None = None) -> None:
        assert self.__optimizer is not None

        epoch_length = len(str(epochs))
        epoch_progress_bar = self.__progress_bar(total=epochs, desc=f'Epochs: {0:>{epoch_length}} / {epochs}', position=0, disable=(self.__verbose != Verbosity.GENERAL))

        for epoch in range(epochs):
            self.__net.train()

            batch_progress_bar = self.__progress_bar(total=len(data_loader), desc=f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}', position=0, disable=(self.__verbose != Verbosity.DETAIL))

            for x, y in data_loader:
                output, y_hat = self.__forward(x)
                running_loss, *_ = self.__step(output, y_hat, y)

                self.__optimizer.zero_grad()
                running_loss.backward()
                self.__optimizer.step()

                batch_progress_bar.set_postfix_str(self.__history.convert(self.__history[-1]))
                batch_progress_bar.update()

            metrics = self.__history.summary()
            self.__history.reset()

            for metric, value in metrics.items():
                self.__writer.add_scalar(f'{metric}/train', value, self.__record_count['train'])

            self.__record_count['train'] += 1

            postfix = self.__history.convert(metrics, prefix='train')

            batch_progress_bar.set_postfix_str(postfix)
            batch_progress_bar.refresh()

            epoch_progress_bar.set_description_str(f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}')
            epoch_progress_bar.set_postfix_str(postfix if epoch_progress_bar.disable or epoch_progress_bar.postfix is None else (postfix + epoch_progress_bar.postfix[len(postfix) :]))
            epoch_progress_bar.update()

            if valid_loader is not None:
                metrics = self.validate(valid_loader)

                postfix = [postfix, self.__history.convert(metrics, prefix='valid')]
                postfix = ', '.join([element for element in postfix if len(element) > 0])

                batch_progress_bar.set_postfix_str(postfix)
                batch_progress_bar.refresh()

                epoch_progress_bar.set_postfix_str(postfix)
                epoch_progress_bar.refresh()

            exclude = ['_', 'self', 'callbacks', 'callback', 'exclude']

            if callbacks is not None:
                stop_iteration = False

                for callback in callbacks:
                    if (callback.interval > 0 and (epoch + 1) % callback.interval == 0) or (epoch + 1 == epochs and callback.last):
                        try:
                            retval = callback(**{key: value for key, value in locals().items() if key not in exclude} | {'runner': self})

                            if callback.name is not None and retval is not None:
                                locals()[callback.name] = retval

                        except StopIteration as e:
                            print(e)
                            stop_iteration = True

                            break

                if stop_iteration:
                    break

        epoch_progress_bar.set_description_str(f'Epochs: {epochs} / {epochs}')
        epoch_progress_bar.set_postfix_str()
        epoch_progress_bar.refresh()

    @torch.no_grad()
    def validate(self, data_loader: DataLoader) -> dict[str, int | float]:
        self.__net.eval()

        progress_bar = self.__progress_bar(total=len(data_loader), desc='Validate', position=1, leave=False, disable=(self.__verbose == Verbosity.NONE))

        for x, y in data_loader:
            output, y_hat = self.__forward(x)
            _ = self.__step(output, y_hat, y)

            progress_bar.set_postfix_str(self.__history.convert(self.__history[-1]))
            progress_bar.update()

        metrics = self.__history.summary()
        self.__history.reset()

        for metric, value in metrics.items():
            self.__writer.add_scalar(f'{metric}/validate', value, self.__record_count['validate'])

        self.__record_count['validate'] += 1

        progress_bar.set_postfix_str(self.__history.convert(metrics))
        progress_bar.refresh()

        return metrics

    @torch.no_grad()
    def test(self, data_loader: DataLoader) -> dict[str, int | float]:
        self.__net.eval()

        progress_bar = self.__progress_bar(total=len(data_loader), desc='Test', position=0, leave=True, disable=(self.__verbose == Verbosity.NONE))

        for x, y in data_loader:
            output, y_hat = self.__forward(x)
            _ = self.__step(output, y_hat, y)

            progress_bar.set_postfix_str(self.__history.convert(self.__history[-1]))
            progress_bar.update()

        metrics = self.__history.summary()
        self.__history.reset()

        for metric, value in metrics.items():
            self.__writer.add_scalar(f'{metric}/test', value, self.__record_count['test'])

        self.__record_count['test'] += 1

        progress_bar.set_postfix_str(self.__history.convert(metrics))
        progress_bar.refresh()

        return metrics

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> tuple[torch.Tensor, ...]:
        self.__net.eval()

        progress_bar = self.__progress_bar(total=len(data_loader), desc='Evaluate', position=0, leave=True, disable=(self.__verbose == Verbosity.NONE))

        ys = []
        y_hats = []

        for x, y in data_loader:
            _, y_hat = self.__forward(x)

            ys.append(y)
            y_hats.append(y_hat)

            progress_bar.update()

        y = torch.concat(ys, dim=0).cpu()
        y_hat = torch.concat(y_hats, dim=0).cpu()

        return y, y_hat

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.__net.eval()

        progress_bar = self.__progress_bar(total=1, desc='Predict', position=0, leave=True, disable=(self.__verbose == Verbosity.NONE))

        _, y_hat = self.__forward(x)
        y_hat = y_hat.cpu()

        progress_bar.update()

        return y_hat

    @property
    @torch.no_grad()
    def weights(self) -> dict[str, Any]:
        return {'net': copy.deepcopy(self.__net).cpu()}

    @property
    @torch.no_grad()
    def optimizer(self) -> optim.Optimizer | None:
        return self.__optimizer

    @optimizer.setter
    @torch.no_grad()
    def optimizer(self, value: optim.Optimizer | None) -> None:
        self.__optimizer = value

    @property
    @torch.no_grad()
    def criterion(self) -> nn.Module | nn.modules.loss._Loss | None:
        return self.__criterion

    @criterion.setter
    @torch.no_grad()
    def criterion(self, value: nn.Module | nn.modules.loss._Loss | None) -> None:
        self.__criterion = value
