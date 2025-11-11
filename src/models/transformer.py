"""Implementation of Transformer Block for Vision Transformer."""

import torch
import torch.nn as nn

from .attention import Attention
from .drop_path import DropPath
from .mlp import MLP


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""

    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        num_heads: int,
        qkv_bias: bool,
        hidden_drop: float,
        attn_drop: float,
        path_drop: float,
    ) -> None:
        """Initialize the Transformer Block.

        Args:
            embed_dim (int): Embedding dimension.
            intermediate_size (int): Dimension of the MLP hidden layer.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to use bias in QKV projection.
            hidden_drop (float): Dropout rate for MLP.
            attn_drop (float): Dropout rate for attention.
            path_drop (float): DropPath rate for stochastic depth.
        """
        super().__init__()

        # Layer normalization before attention (Pre-LayerNorm architecture)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Multi-head self-attention
        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop_prob=attn_drop,
            proj_drop_prob=hidden_drop,
        )

        # DropPath for attention branch
        self.drop_path_attn = DropPath(path_drop) if path_drop > 0.0 else nn.Identity()

        # Layer normalization before MLP
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP block
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=intermediate_size,
            out_features=embed_dim,
            drop=hidden_drop,
        )

        # DropPath for MLP branch
        self.drop_path_mlp = DropPath(path_drop) if path_drop > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Transformer Block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, embed_dim).
        """
        # Attention block with residual connection
        x = x + self.drop_path_attn(self.attn(self.norm1(x)))
        # MLP block with residual connection
        return x + self.drop_path_mlp(self.mlp(self.norm2(x)))
