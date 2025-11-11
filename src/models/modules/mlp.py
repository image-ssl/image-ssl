"""Implementation of MLP (Feedforward) block for Vision Transformer."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP (Feedforward) block with GeLU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float,
    ) -> None:
        """Initialize the MLP module.

        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden features.
            out_features (int): Number of output features.
            drop (float): Dropout rate.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)
