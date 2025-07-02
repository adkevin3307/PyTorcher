import os
import random
import logging
import numpy as np
from typing import Any, Callable

import torch
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger('PyTorch-Runner')


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


class ProgressBar:
    __buffer = ''
    __last_length = 0

    @staticmethod
    def show(prefix: str, postfix: str, current: int, total: int, show_progress: bool = True, newline: bool = False, freeze: bool = False) -> None:
        progress = (current + 1) / total
        if current == total:
            progress = 1

        current_progress = progress * 100
        progress_bar = '=' * int(progress * 20)

        message = ''

        if len(prefix) > 0:
            message += f'{prefix}, '

        if show_progress:
            message += f'[{progress_bar:<20}]'

            if not newline:
                message += f' {current_progress:6.2f}%'

        if len(postfix) > 0:
            message += f', {postfix}'

        message = ProgressBar.__buffer + message

        print(f'\r{" " * ProgressBar.__last_length}', end='', flush=True)
        print(f'\r{message}', end='', flush=True)

        if freeze:
            ProgressBar.__buffer = message

        if newline:
            print()

            logger.info(message)

            ProgressBar.__buffer = ''
            ProgressBar.__last_length = 0
        else:
            ProgressBar.__last_length = len(message) + 1


class Callback:
    def __init__(self, function: Callable, interval: int, *args, **kwargs) -> None:
        self.__function = function
        self.__interval = interval

        self.__args = args
        self.__kwargs = kwargs

    def __call__(self, *args, **kwargs) -> Any:
        return self.__function(*(self.__args + args), **(self.__kwargs | kwargs))

    @property
    def interval(self) -> int:
        return self.__interval
