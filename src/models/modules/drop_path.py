"""Implementation of DropPath for stochastic depth regularization."""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    When applied in main path of residual blocks, this is equivalent to
    stochastically dropping the entire residual block during training.

    Reference: Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
    """

    def __init__(self, drop_prob: float) -> None:
        """Initialize DropPath module.

        Args:
            drop_prob (float): Probability of dropping a path (0.0 to 1.0).
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DropPath to input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor (same shape as input).
        """
        # If drop_prob is 0 or not training, return input unchanged
        if self.drop_prob == 0.0 or not self.training:
            return x

        # Calculate keep probability
        keep_prob = 1.0 - self.drop_prob

        # Create random tensor with shape (B, 1, 1, ...) to broadcast
        # across all dimensions except batch
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)

        # Binarize: convert to 0 or 1
        random_tensor.floor_()

        # Scale by keep_prob to maintain expected value during training
        return x.div(keep_prob) * random_tensor
