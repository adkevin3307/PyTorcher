import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import EarlyStop


def validate_callback(data_loader: DataLoader, *args, **kwargs) -> dict[str, int | float]:
    runner = kwargs['runner']

    metrics = runner.validate(data_loader)

    return metrics


def scheduler_callback(scheduler: optim.lr_scheduler.LRScheduler, *args, **kwargs) -> None:
    scheduler.step()


def checkpoint_callback(tag: str = '', checkpoints: str = 'checkpoints', *args, **kwargs) -> None:
    runner = kwargs['runner']
    epoch = kwargs['epoch']

    folder = os.path.join(checkpoints, tag)
    os.makedirs(folder, exist_ok=True)

    torch.save(runner.weights['net'].state_dict(), os.path.join(folder, 'final' if epoch is None else f'{epoch + 1}' + '.pt'))


def early_stop_callback(early_stop: EarlyStop, *args, **kwargs) -> None:
    metrics = kwargs['metrics']

    early_stop.step(metrics)

    if early_stop.stop():
        raise StopIteration(f'Stop by Early Stopping check with metric `{early_stop.monitor}` (best: {early_stop.best:.3f}, patience: {early_stop.patience})')
