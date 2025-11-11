"""Implementation of Vision Transformer (ViT)."""

# https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/vit#transformers.ViTConfig

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .modules.patch import PatchEmbedding
from .modules.transformer import TransformerBlock


class VisionTransformerOutput:
    """Output class for VisionTransformer."""

    def __init__(self, last_hidden_state: torch.Tensor, cls: torch.Tensor) -> None:
        """Initialize the output with the last hidden state.

        Args:
            last_hidden_state (torch.Tensor): The final hidden states from the transformer.
            cls (torch.Tensor): The CLS token representation.
        """
        self.last_hidden_state = last_hidden_state
        self.cls = cls

    def __getitem__(self, key: str) -> torch.Tensor:
        """Allow dictionary-style access: output['last_hidden_state'].

        Args:
            key (str): The key corresponding to the desired output.

        Returns:
            torch.Tensor: The output tensor corresponding to the key.
        """
        return getattr(self, key)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        """Allow dictionary-style assignment: output['last_hidden_state'] = value.

        Args:
            key (str): The key corresponding to the output to set.
            value (torch.Tensor): The tensor to set for the given key.

        Returns:
            None
        """
        setattr(self, key, value)


class VisionTransformer(nn.Module, PyTorchModelHubMixin):
    """Vision Transformer Base Model."""

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
    ) -> None:
        """Initialize the Vision Transformer.

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
        """
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token - learnable parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Position embeddings - learnable parameters; +1 for CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))

        # Position embedding dropout
        self.pos_drop = nn.Dropout(p=dropout_hidden)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_attention_heads,
                    qkv_bias=qkv_bias,
                    hidden_drop=dropout_hidden,
                    attn_drop=dropout_attention,
                    path_drop=dropout_path,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for the model."""
        # Initialize position embeddings and cls token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize patch embedding projection
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=0.02)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

        # Initialize all linear layers
        self.apply(self._init_linear_weights)

    def _init_linear_weights(self, m: nn.Module) -> None:
        """Initialize linear layer weights.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: CLS token output of shape (B, hidden_size).
        """
        B = x.shape[0]  # noqa: N806

        # Patch embedding: (B, C, H, W) -> (B, num_patches, hidden_size)
        x = self.patch_embed(x)

        # Add CLS token: (B, num_patches, hidden_size) -> (B, num_patches + 1, hidden_size)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_size)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        # Return CLS token only
        return VisionTransformerOutput(last_hidden_state=x, cls=x[:, 0])  # (B, hidden_size)
