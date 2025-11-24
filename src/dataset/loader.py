"""DataLoader creation for SSL pretraining."""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split

from .dataset import ImageTransformDataset
from .transform import ImageTransform


def create_pretrain_dataloaders(
    val_split: float | None,
    batch_size: int,
    image_size: int,
    num_local_crops: int = 6,
    local_crop_size: int = 36,
    global_crops_scale: tuple[float, float] = (0.4, 1.0),
    local_crops_scale: tuple[float, float] = (0.05, 0.4),
    seed: int = 42,
) -> tuple[DataLoader, DataLoader | None]:
    """Create train and optional validation DataLoaders for SSL pretraining.

    Args:
        val_split (float | None): Fraction of dataset to use as validation (0.0-1.0). None = no val set.
        batch_size (int): Batch size for DataLoader.
        image_size (int): Size to which images are resized/cropped.
        num_local_crops (int): Number of local crops for multi-crop (DINO). Default=6.
        local_crop_size (int): Size of local crops for multi-crop (DINO). Default=36.
        global_crops_scale (tuple[float, float]): Scale range for global crops. Default (0.4, 1.0).
        local_crops_scale (tuple[float, float]): Scale range for local crops. Default (0.05, 0.4).
        seed (int): Random seed for dataset splitting. Default=42.

    Returns:
        tuple[train_loader, val_loader | None]: DataLoaders for train and val (val_loader=None if val_split is None).
    """
    # Load Hugging Face dataset
    hf_dataset = load_dataset("tsbpp/fall2025_deeplearning", split="train")
    transform = ImageTransform(
        image_size=image_size,
        num_local_crops=num_local_crops,
        local_crop_size=local_crop_size,
        global_crops_scale=global_crops_scale,
        local_crops_scale=local_crops_scale,
    )
    dataset = ImageTransformDataset(hf_dataset, transform=transform)

    # Split into train/val if needed
    if val_split is not None and 0.0 < val_split < 1.0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            drop_last=False,
        )
    else:
        train_dataset = dataset
        val_loader = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader
