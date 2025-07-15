import os
import enum
import random
import numpy as np
from typing import Any, Callable

import torch
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader


def fix_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def load_data(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = -1) -> DataLoader:
    if num_workers == -1:
        num_workers = len(os.sched_getaffinity(0))

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader


class Verbosity(enum.Enum):
    NONE = 0
    DETAIL = 1
    GENERAL = 2


class ProgressBar:
    __length = 0

    @staticmethod
    def show(prefix: str, postfix: str, current: int, total: int, show_progress: bool = True, newline: bool = False) -> str:
        progress = 1 if current == total else ((current + 1) / total)

        current_progress = progress * 100
        progress_bar = '=' * int(progress * 20)

        message = []

        if len(prefix) > 0:
            message.append(f'{prefix}')

        if show_progress:
            message.append(f'[{progress_bar:<20}] {current_progress:6.2f}%')

        if len(postfix) > 0:
            message.append(f'{postfix}')

        message = ', '.join(message)

        print(f'\r{" " * ProgressBar.__length}', end='', flush=True)
        print(f'\r{message}', end='', flush=True)

        if newline:
            print()

            ProgressBar.__length = 0
        else:
            ProgressBar.__length = len(message) + 1

        return message


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


class Callback:
    def __init__(self, function: Callable, interval: int, name: str | None = None, *args, **kwargs) -> None:
        self.__function = function
        self.__interval = interval
        self.__name = name

        self.__args = args
        self.__kwargs = kwargs

    def __call__(self, *args, **kwargs) -> Any:
        return self.__function(*(self.__args + args), **(self.__kwargs | kwargs))

    @property
    def interval(self) -> int:
        return self.__interval

    @property
    def name(self) -> str | None:
        return self.__name
