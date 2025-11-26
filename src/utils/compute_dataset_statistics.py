"""Compute dataset statistics such as mean and standard deviation for normalization."""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F  # noqa: N812
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
        images = batch["image"].to(device, non_blocking=True)
        sum_pixels += images.sum(dim=[0, 2, 3])
        sum_squared_pixels += (images**2).sum(dim=[0, 2, 3])
        total_pixels += images.shape[0] * images.shape[2] * images.shape[3]

    mean = sum_pixels / total_pixels
    variance = (sum_squared_pixels / total_pixels) - (mean**2)
    std = torch.sqrt(variance)
    return mean.cpu().tolist(), std.cpu().tolist()


if __name__ == "__main__":
    dataset = load_dataset("tsbpp/fall2025_deeplearning", split="train")
    sample_size = 10_000
    dataset = dataset.select(range(sample_size))

    def _transform_example(example: dict) -> dict:
        """Transform function to convert images to tensors.

        Args:
            example (dict): Dataset example.

        Returns:
            dict: Transformed example.
        """
        example["image"] = F.to_tensor(example["image"])
        return example

    dataset = dataset.with_transform(_transform_example)

    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    mean, std = compute_dataloader_statistics(dataloader, "cuda:0")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
