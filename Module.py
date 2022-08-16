from typing import Optional, Sequence, Any
import torch
import torch.nn as nn
import torch.optim as optim


class WeightedLoss(nn.Module):
    def __init__(self, criterion: Any, reduction: str = 'mean', epsilon: float = 1e-8, max_weight: Optional[float] = None, disable_weighted: bool = False) -> None:
        super(WeightedLoss, self).__init__()

        self.epsilon = epsilon
        self.max_weight = max_weight
        self.disable_weighted = disable_weighted

        if self.disable_weighted:
            self.criterion = criterion(reduction=reduction)
        else:
            self.criterion = criterion(reduction='none')

        if reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'sum':
            self.reduction = torch.sum
        else:
            raise NotImplementedError('reduction not implemented')

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.disable_weighted:
            loss = self.criterion(outputs, targets)
        else:
            with torch.no_grad():
                _targets = torch.abs(targets)
                _targets[_targets < self.epsilon] = self.epsilon

                weight = torch.sum(_targets, dim=1, keepdim=True) / _targets

                if self.max_weight:
                    weight = torch.clamp(weight, max=self.max_weight)

            loss = self.criterion(outputs, targets)
            loss = self.reduction(weight * loss)

        return loss


class GradualWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, init_lr: float, warmup_epoch: int, scheduler: optim.lr_scheduler._LRScheduler, updatable: bool = True, max_warmup_epoch: int = 500) -> None:
        self.last_epoch = -1
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]

        self.init_lr = init_lr
        self.warmup_epoch = min(warmup_epoch, max_warmup_epoch)
        self.scheduler = scheduler
        self.updatable = updatable

        super(GradualWarmupScheduler, self).__init__(optimizer)


    def get_lr(self) -> Sequence[float]:
        if self.last_epoch > self.warmup_epoch:
            return self.scheduler.get_last_lr()

        if self.updatable:
            return [self.init_lr + (base_lr - self.init_lr) * (self.last_epoch / self.warmup_epoch) for base_lr in self.base_lrs]

        return [self.init_lr]

    def step(self, epoch: Optional[int] = None) -> None:
        if self.last_epoch > self.warmup_epoch:
            if epoch is None:
                self.scheduler.step(None)
            else:
                self.scheduler.step(epoch - self.warmup_epoch)

            self._last_lr = self.scheduler.get_last_lr()
        else:
            super(GradualWarmupScheduler, self).step(epoch)
