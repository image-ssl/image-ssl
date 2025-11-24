"""Implementation of DINO head for DinoV1 self-supervised learning."""

import torch
import torch.nn as nn


class DINOHead(nn.Module):
    """DINO Head as described in the DinoV1 paper."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: int,
        norm_last_layer: bool,
        num_layers: int,
        hidden_dim: int,
        bottleneck_dim: int,
    ) -> None:
        """Initialize the DINO Head.

        Args:
            in_dim (int): Dimension of the input features.
            out_dim (int): Dimension of the output features.
            use_bn (bool): Whether to use batch normalization.
            norm_last_layer (bool): Whether to normalize the last layer.
            num_layers (int): Number of layers in the MLP.
            hidden_dim (int): Dimension of the hidden layers.
            bottleneck_dim (int): Dimension of the bottleneck layer.

        """
        super().__init__()
        num_layers = max(num_layers, 1)
        # Build the MLP
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        # Initialize weights
        self.apply(self._init_weights)
        # Wrap the last layer with weight normalization
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        # Freeze the scale of the last layer
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize the weights of linear layers with truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DINO head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim).
        """
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        return self.last_layer(x)
