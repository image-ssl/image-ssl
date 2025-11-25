import torch

def compute_dataset_statistics(dataloader):
    """Compute mean and std for your dataset."""
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    
    print("Computing mean...")
    for images, _ in dataloader:
        # images shape: [batch, channels, height, width]
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)  # [batch, channels, pixels]
        mean += images.mean(2).sum(0)
        total_images += batch_size
    
    mean /= total_images
    
    print("Computing std...")
    for images, _ in dataloader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        std += ((images - mean.view(3, 1)) ** 2).mean(2).sum(0)
    
    std = torch.sqrt(std / total_images)
    
    return mean.tolist(), std.tolist()

# Usage - compute on raw ToTensor() data (before normalization)
from torch.utils.data import DataLoader

# Create a temporary transform WITHOUT normalization
temp_transform = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),  # Converts to [0, 1]
])

# Apply to your dataset
temp_dataset = YourDataset(transform=temp_transform)
temp_loader = DataLoader(temp_dataset, batch_size=64, shuffle=False)

mean, std = compute_dataset_statistics(temp_loader)
print(f"Mean: {mean}")
print(f"Std: {std}")