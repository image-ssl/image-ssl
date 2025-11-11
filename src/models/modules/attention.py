"""Implementation of Multi-Head Self-Attention for Vision Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class Attention(nn.Module):
    """Multi-Head Self-Attention module."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool,
        attn_drop_prob: float,
        proj_drop_prob: float,
    ) -> None:
        """Initialize the Attention module.

        Args:
            embed_dim (int): Dimension of input embeddings.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to include bias in QKV projection.
            attn_drop_prob (float): Dropout rate for attention weights.
            proj_drop_prob (float): Dropout rate for output projection.
        """
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5  # 1/sqrt(head_dim) for scaled dot-product

        # Single linear layer for Q, K, V projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)

        # Dropout for attention weights
        self.attn_drop_prob = attn_drop_prob

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, embed_dim)
                             where B is batch size, N is sequence length

        Returns:
            torch.Tensor: Output tensor of shape (B, N, embed_dim)
        """
        B, N, C = x.shape  # noqa: N806

        # Generate Q, K, V
        # qkv: (B, N, 3 * embed_dim) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, N, head_dim)

        # Scaled dot-product attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop_prob if self.training else 0.0,
            is_causal=False,
        )

        # Reshape back: (B, num_heads, N, head_dim) -> (B, N, embed_dim)
        x = x.transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)
        return self.proj_drop(x)
