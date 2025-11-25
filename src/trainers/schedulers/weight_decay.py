"""Weight decay scheduler that automatically applies to optimizer."""

import numpy as np
import torch


class WeightDecayScheduler:
    """Cosine scheduler for weight decay that automatically applies to optimizer."""

    def __init__(self, optimizer: torch.optim.Optimizer, base_wd: float, final_wd: float, total_steps: int) -> None:
        """Initialize the WeightDecayScheduler.

        Args:
            optimizer: PyTorch optimizer
            base_wd: Starting weight decay value
            final_wd: Ending weight decay value
            total_steps: Total number of training steps
        """
        self.optimizer = optimizer
        self.base_wd = base_wd
        self.final_wd = final_wd
        self.total_steps = total_steps
        self.current_step = 0

    def step(self) -> None:
        """Compute current weight decay and apply to all param groups."""
        wd = self._get_wd()

        # Only apply to first param group (regularized parameters)
        if len(self.optimizer.param_groups) > 0:
            self.optimizer.param_groups[0]["weight_decay"] = wd

        self.current_step += 1

    def _get_wd(self) -> float:
        """Compute weight decay based on cosine schedule."""
        if self.current_step >= self.total_steps:
            return self.final_wd

        progress = self.current_step / self.total_steps
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return self.final_wd + (self.base_wd - self.final_wd) * cosine_decay

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.current_step = state_dict["current_step"]
