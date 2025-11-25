"""Implementation of Vision Transformer (ViT) with Pre-training Heads for Self-Supervised Learning."""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from src.models.modules.dino import DINOHead

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
        dino_out_dim: int,
        dino_use_bn: bool,
        dino_norm_last_layer: bool,
        dino_num_layers: int,
        dino_hidden_dim: int,
        dino_bottleneck_dim: int,
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
            dino_out_dim (int): Output dimension for DINO head.
            dino_use_bn (bool): Whether to use batch norm in DINO head.
            dino_norm_last_layer (bool): Whether to normalize last layer in DINO head.
            dino_num_layers (int): Number of layers in DINO head.
            dino_hidden_dim (int): Hidden dimension in DINO head.
            dino_bottleneck_dim (int): Bottleneck dimension in DINO head.
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

        # Initialize pre-training head
        self.heads = nn.ModuleDict()
        self.heads["dino"] = DINOHead(
            in_dim=hidden_size,
            out_dim=dino_out_dim,
            use_bn=dino_use_bn,
            norm_last_layer=dino_norm_last_layer,
            num_layers=dino_num_layers,
            hidden_dim=dino_hidden_dim,
            bottleneck_dim=dino_bottleneck_dim,
        )

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> VisionTransformerWithPretrainingHeadsOutput:
        """Forward pass through the Vision Transformer and pre-training heads."""
        # Handle case where input is a list of tensors (multi-crop)
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        # Encode all crops
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            # Forward pass through the encoder
            _out = self.encoder(torch.cat(x[start_idx:end_idx])).cls
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        # TODO: Wrap this around ViTWithPretrainingHeadsOutput for multiple heads.
        return self.heads["dino"](output)
