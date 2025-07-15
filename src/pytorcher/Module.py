import torch
import torch.nn as nn
import torch.optim as optim


class WeightedLoss(nn.Module):
    def __init__(self, criterion: nn.Module | nn.modules.loss._Loss, reduction: str = 'mean', max_weight: float | None = None, epsilon: float = 1e-8) -> None:
        super(WeightedLoss, self).__init__()

        self.__max_weight = max_weight
        self.__epsilon = epsilon

        self.__criterion = criterion(reduction='none')

        match reduction:
            case 'mean':
                self.__reduction = torch.mean

            case 'sum':
                self.__reduction = torch.sum

            case _:
                raise NotImplementedError(f'reduction {reduction} not implemented')

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _targets = torch.abs(targets)
            _targets[_targets < self.__epsilon] = self.__epsilon

            weight = torch.sum(_targets, dim=1, keepdim=True) / _targets

            if self.__max_weight is not None:
                weight = torch.clamp(weight, max=self.__max_weight)

        loss = self.__criterion(outputs, targets)
        loss = self.__reduction(weight * loss)

        return loss


class GradualWarmupScheduler(optim.lr_scheduler.LRScheduler):
    def __init__(self, scheduler: optim.lr_scheduler.LRScheduler, init_lr: float, warmup_epoch: int, updatable: bool = True) -> None:
        self.__scheduler = scheduler
        self.__init_lr = init_lr
        self.__warmup_epoch = warmup_epoch
        self.__updatable = updatable

        self.last_epoch = -1
        self.base_lrs = [group['initial_lr'] for group in self.__scheduler.optimizer.param_groups]

        super(GradualWarmupScheduler, self).__init__(self.__scheduler.optimizer)

    def get_lr(self) -> list[float]:
        if self.last_epoch > self.__warmup_epoch:
            return self.__scheduler.get_last_lr()

        if self.__updatable:
            return [self.__init_lr + (base_lr - self.__init_lr) * (self.last_epoch / self.__warmup_epoch) for base_lr in self.base_lrs]

        return [self.__init_lr]

    def step(self, epoch: int | None = None) -> None:
        if self.last_epoch > self.__warmup_epoch:
            self.__scheduler.step(None if epoch is None else (epoch - self.__warmup_epoch))
            self._last_lr = self.__scheduler.get_last_lr()

        else:
            super(GradualWarmupScheduler, self).step(epoch)
