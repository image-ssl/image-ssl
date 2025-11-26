"""Compute dataset statistics such as mean and standard deviation for normalization."""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm


def compute_dataloader_statistics(dataloader: DataLoader, device: str = "cuda") -> tuple[list[float], list[float]]:
    """Compute mean and std.

    Args:
        dataloader (DataLoader): DataLoader for the dataset.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        tuple[list[float], list[float]]: Mean and std for each channel.
    """
    sum_pixels = torch.zeros(3, device=device)
    sum_squared_pixels = torch.zeros(3, device=device)
    total_pixels = 0

    for batch in tqdm(dataloader, desc="Computing statistics"):
        images = batch["image"].to(device)
        batch_size, channels, height, width = images.shape
        n_pixels = batch_size * height * width
        images = images.view(channels, -1)
        sum_pixels += images.sum(dim=1)
        sum_squared_pixels += (images**2).sum(dim=1)
        total_pixels += n_pixels

    mean = sum_pixels / total_pixels
    variance = (sum_squared_pixels / total_pixels) - (mean**2)
    std = torch.sqrt(variance)
    return mean.cpu().tolist(), std.cpu().tolist()


def _collate_fn(batch: list) -> dict:
    """Custom collate function to transform images to tensors.

    Args:
        batch (list): List of dataset items.

    Returns:
        dict: Batch with images as tensors.
    """
    images = torch.stack([_transform(item["image"]) for item in batch])
    return {"image": images}


if __name__ == "__main__":
    dataset = load_dataset("tsbpp/fall2025_deeplearning", split="train")
    sample_size = 10_000
    dataset = dataset.select(range(sample_size))
    _transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts PIL Image to [C, H, W] tensor in [0, 1]
        ]
    )
    dataloader = DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=8, collate_fn=_collate_fn, persistent_workers=True
    )
    mean, std = compute_dataloader_statistics(dataloader, "cuda:0")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
