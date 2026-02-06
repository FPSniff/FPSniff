"""Learning-rate warmup scheduler.

A lightweight implementation of a gradual warmup scheduler that wraps another scheduler.
This avoids external dependencies and keeps the repository self-contained.
"""

from __future__ import annotations

from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    """Linearly warm up the learning rate, then delegate to an after-scheduler.

    Args:
        optimizer: Torch optimizer.
        multiplier: Target lr multiplier after warmup. If 1.0, warmup reaches the base lr.
        warm_epoch: Number of warmup epochs.
        after_scheduler: Scheduler to use after warmup (e.g., CosineAnnealingLR).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        multiplier: float,
        warm_epoch: int,
        after_scheduler: Optional[_LRScheduler] = None,
        last_epoch: int = -1,
    ) -> None:
        if warm_epoch < 1:
            raise ValueError("warm_epoch must be >= 1")
        if multiplier < 1.0:
            raise ValueError("multiplier must be >= 1.0")

        self.multiplier = float(multiplier)
        self.warm_epoch = int(warm_epoch)
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.finished:
            return self.after_scheduler.get_last_lr() if self.after_scheduler is not None else [group["lr"] for group in self.optimizer.param_groups]

        # Epoch is 0-indexed in PyTorch schedulers.
        cur_epoch = self.last_epoch + 1
        if cur_epoch <= self.warm_epoch:
            # Linear warmup from base_lr to base_lr * multiplier
            scale = 1.0 + (self.multiplier - 1.0) * (cur_epoch / float(self.warm_epoch))
            return [base_lr * scale for base_lr in self.base_lrs]

        # Warmup done
        self.finished = True
        if self.after_scheduler is not None:
            # Set the starting point for after_scheduler.
            self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
            return self.after_scheduler.get_lr()
        return [base_lr * self.multiplier for base_lr in self.base_lrs]

    def step(self, epoch: Optional[int] = None):
        if self.finished and self.after_scheduler is not None:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warm_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
            return

        super().step(epoch)
        if self.finished and self.after_scheduler is not None:
            self._last_lr = self.after_scheduler.get_last_lr()
