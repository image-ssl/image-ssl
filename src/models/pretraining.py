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
            if objective == "dino":
                # TODO: Add as args
                # TODO: Check DinoHead param sizes for encoder sizes
                self.heads[objective] = DINOHead(
                    in_dim=hidden_size,
                    out_dim=65536,  # typical DINO out_dim
                    use_bn=False,  # match original defaults unless you want BN
                    norm_last_layer=True,
                    num_layers=3,
                    hidden_dim=2048,
                    bottleneck_dim=256,
                )
            else:
                raise NotImplementedError(f"Unsupported pre-training objective: {objective}")

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> VisionTransformerWithPretrainingHeadsOutput:
        """Forward pass through the Vision Transformer and pre-training heads."""
        # cls = self.encoder(x)  # [B, hidden_size], CLS embedding
        # outputs = {"encoder": cls}
        # for name, head in self.heads.items():
        #     outputs[name] = head(cls)  # <-- pass CLS directly
        # return VisionTransformerWithPretrainingHeadsOutput(**outputs)
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.encoder(torch.cat(x[start_idx:end_idx])).cls
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.heads["dino"](output)
