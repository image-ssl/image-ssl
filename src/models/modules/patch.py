"""Implementation of a PatchEmbedding model for image patches."""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Patch Embedding module for images."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        """Initialize the Patch Embedding module.

        Args:
            image_size (int): Size of the input image (assumed square).
            patch_size (int): Size of each patch (assumed square).
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            embed_dim (int): Dimension of the embedding space.

        Returns:
            PatchEmbedding: An instance of the PatchEmbedding module.
        """
        super().__init__()

        assert image_size % patch_size == 0, f"Image size ({image_size}) must be divisible by patch size ({patch_size})"

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = (self.grid_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Patch Embedding module.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, in_channels, image_size, image_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, embed_dim).
        """
        x = self.proj(x)
        x = x.flatten(2)
        return x.transpose(1, 2)
