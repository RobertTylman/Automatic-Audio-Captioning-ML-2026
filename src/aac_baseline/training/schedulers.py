import warnings

import torch


class InverseLRHalfMix(torch.optim.lr_scheduler._LRScheduler):
    """
    Inverse decay with exponential warmup and a 50/50 residual base-LR mix.

    Closed form:
        base_lr * (1 - warmup ** (step + 1)) *
        (0.5 * (1 + step / inv_gamma) ** (-power) + 0.5)

    This decays like inverse LR early on, but asymptotes to 0.5 * base_lr
    instead of decaying toward zero.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        inv_gamma: float = 1.0,
        power: float = 1.0,
        warmup: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if inv_gamma <= 0:
            raise ValueError("Invalid value for inv_gamma")
        if power < 0:
            raise ValueError("Invalid value for power")
        if not 0.0 <= warmup < 1.0:
            raise ValueError("Invalid value for warmup")

        self.inv_gamma = inv_gamma
        self.power = power
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                stacklevel=2,
            )
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self) -> list[float]:
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        inverse_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        mixed_mult = 0.5 * inverse_mult + 0.5
        return [base_lr * warmup * mixed_mult for base_lr in self.base_lrs]


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict | None,
):
    if not scheduler_config:
        return None

    scheduler_name = scheduler_config.get("name")
    if scheduler_name in {"inverse_lr_half_mix", "InverseLRHalfMix"}:
        return InverseLRHalfMix(
            optimizer=optimizer,
            inv_gamma=scheduler_config.get("inv_gamma", 1.0),
            power=scheduler_config.get("power", 1.0),
            warmup=scheduler_config.get("warmup", 0.0),
        )

    raise ValueError(f"Unsupported lr scheduler: {scheduler_name}")