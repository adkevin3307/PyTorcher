import pandas as pd
from typing import Callable, Any

import torch
from torch.utils.data import Dataset


class Normalize:
    def __init__(self, mean: pd.Series, std: pd.Series, epsilon: float = 1e-8) -> None:
        self.__mean = mean
        self.__std = std
        self.__epsilon = epsilon

        assert set(self.__mean.index) == set(self.__std.index)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        _df = df.copy()

        columns = _df.columns[_df.columns.isin(self.__mean.index.to_list())].to_list()
        _df.loc[:, columns] = (_df[columns] - self.__mean[columns]) / (self.__std[columns] + self.__epsilon)

        return _df


class Denormalize:
    def __init__(self, mean: pd.Series, std: pd.Series, epsilon: float = 1e-8) -> None:
        self.__mean = mean
        self.__std = std
        self.__epsilon = epsilon

        assert set(self.__mean.index) == set(self.__std.index)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        _df = df.copy()

        columns = _df.columns[_df.columns.isin(self.__mean.index.to_list())].to_list()
        _df.loc[:, columns] = _df[columns] * (self.__std[columns] + self.__epsilon) + self.__mean[columns]

        return _df


class ToTensor:
    def __init__(self, exclude_columns: list[str] = ['time']) -> None:
        self.__exclude_columns = exclude_columns

    def __call__(self, df: pd.DataFrame) -> torch.Tensor:
        columns = df.columns[~df.columns.isin(self.__exclude_columns)]

        return torch.tensor(df[columns].values, dtype=torch.float)


class Dummy:
    def __init__(self) -> None:
        pass

    def __call__(self, value: Any) -> Any:
        return value


class CustomDataset(Dataset):
    def __init__(self, transforms: tuple[Callable, Callable] | None = None) -> None:
        self.__transforms = transforms

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx = self.__len__() + idx

        if self.__transforms is not None:
            x_transform, y_transform = self.__transforms

        raise NotImplementedError
