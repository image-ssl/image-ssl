"""Implementation of Vision Transformer (ViT) with Pre-training Heads for Self-Supervised Learning."""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .vit import VisionTransformer


class VisionTransformerWithPretrainingHeadsOutput:
    """Output class for VisionTransformerWithPretrainingHeads."""

    def __init__(self, **kwargs: dict[str, torch.Tensor]) -> None:
        """Initialize the output with pre-training head outputs.

        Args:
            **kwargs (dict[str, torch.Tensor]): Outputs from different pre-training heads.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key: str) -> torch.Tensor:
        """Allow dictionary-style access: output['simclr'].

        Args:
            key (str): The key corresponding to the desired output.

        Returns:
            torch.Tensor: The output tensor corresponding to the key.
        """
        return getattr(self, key)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        """Allow dictionary-style assignment: output['simclr'] = value.

        Args:
            key (str): The key corresponding to the output to set.
            value (torch.Tensor): The tensor to set for the given key.

        Returns:
            None
        """
        setattr(self, key, value)


class VisionTransformerWithPretrainingHeads(nn.Module, PyTorchModelHubMixin):
    """Vision Transformer with Pre-training Heads for self-supervised learning."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        qkv_bias: bool,
        dropout_hidden: float,
        dropout_attention: float,
        dropout_path: float,
        pretrain_objectives: list[str],
    ) -> None:
        """Initialize the Vision Transformer with Pre-training Heads.

        Args:
            image_size (int): Size of input image (assumed square).
            patch_size (int): Size of each patch (assumed square).
            in_channels (int): Number of input channels (3 for RGB).
            hidden_size (int): Dimension of the encoder layers (embedding dimension).
            num_hidden_layers (int): Number of transformer blocks (depth).
            num_attention_heads (int): Number of attention heads.
            intermediate_size (int): Dimension of the feedforward layers.
            qkv_bias (bool): Whether to use bias in QKV projection.
            dropout_hidden (float): Dropout rate for MLP.
            dropout_attention (float): Dropout rate for attention.
            dropout_path (float): Dropout rate for stochastic depth.
            pretrain_objectives (list[str]): List of pre-training objectives to include.
        """
        super().__init__()

        self.encoder = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            qkv_bias=qkv_bias,
            dropout_hidden=dropout_hidden,
            dropout_attention=dropout_attention,
            dropout_path=dropout_path,
        )

        # Initialize pre-training heads based on specified objectives
        self.heads = nn.ModuleDict()
        for objective in pretrain_objectives:
            if objective == "simclr":
                self.heads[objective] = self._init_simclr_proj_head(hidden_size)
            else:
                raise NotImplementedError(f"Unsupported pre-training objective: {objective}")

    def _init_simclr_proj_head(
        self, in_features: int, projection_size: int = 128, hidden_size: int = 2048
    ) -> nn.Sequential:
        """Initialize the SimCLR projection head.

        Args:
            in_features (int): Input feature dimension.
            projection_size (int): Output projection dimension.
            hidden_size (int): Hidden layer dimension in the projection head.

        Returns:
            nn.Sequential: The SimCLR projection head.
        """
        return nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
            nn.BatchNorm1d(projection_size),
        )

    def forward(self, x: torch.Tensor) -> VisionTransformerWithPretrainingHeadsOutput:
        """Forward pass through the Vision Transformer and pre-training heads."""
        x = self.encoder(x)
        outputs = {"encoder": x}
        for name, head in self.heads.items():
            outputs[name] = head(x.cls)
        return VisionTransformerWithPretrainingHeadsOutput(**outputs)
