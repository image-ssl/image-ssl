"""Momentum scheduler to return momentum values based on cosine schedule."""

import numpy as np


class MomentumScheduler:
    """Cosine scheduler for momentum that returns the scheduled value."""

    def __init__(self, base_momentum: float, final_momentum: float, total_steps: int) -> None:
        """MomentumScheduler initializer.

        Args:
            base_momentum: Starting momentum value
            final_momentum: Ending momentum value
            total_steps: Total number of training steps
        """
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum
        self.total_steps = total_steps
        self.current_step = 0

    def step(self) -> float:
        """Get current momentum value and advance step counter."""
        momentum = self.get_value()
        self.current_step += 1
        return momentum

    def get_value(self) -> float:
        """Compute momentum based on cosine schedule."""
        if self.current_step >= self.total_steps:
            return self.final_momentum

        progress = self.current_step / self.total_steps
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))

        return self.final_momentum + (self.base_momentum - self.final_momentum) * cosine_decay

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.current_step = state_dict["current_step"]
