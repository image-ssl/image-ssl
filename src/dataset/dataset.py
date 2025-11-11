"""Implementation of a TransformDataset class to apply transforms for self-supervised learning."""

import torch
from datasets import Dataset

from .transform import ImageTransform


class ImageTransformDataset(torch.utils.data.Dataset):
    """Wrapper class to apply transformations for self-supervised learning."""

    def __init__(self, dataset: Dataset, transform: ImageTransform | None) -> None:
        """Initialize the TransformDataset.

        Args:
            dataset (Dataset): The underlying HuggingFace dataset.
            transform (ImageTransform | None): The transform to apply to each image.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        """Get an item from the dataset and apply the transform.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict[str, torch.Tensor | tuple[torch.Tensor, ...]]: Transformed image(s) per objective.
        """
        img = self.dataset[idx]["image"]
        if self.transform is not None:
            # could be single image or tuple of views
            img = self.transform(img)
        return img
