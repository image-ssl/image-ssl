"""Image Transformations for Self-Supervised Learning."""

import torch
from PIL import Image
from torchvision import transforms


class ImageTransform:
    """Image Transformation pipeline for self-supervised learning."""

    def __init__(
        self,
        image_size: int,
        transformation_types: list[str],
        num_views: int = 2,
        crop_scale: tuple[float, float] = (0.15, 1.0),
        crop_size: int | None = None,
    ) -> None:
        """Initialize the image transform.

        Args:
            image_size (int): Size to which images are resized/cropped.
            transformation_types (list[str]): Types of transformation pipelines.
            num_views (int): Number of augmented views to create per image. Default=2.
            crop_scale (tuple[float, float]): Scale range for RandomResizedCrop (min, max).
                                             Lower values = smaller crops. Default=(0.15, 1.0).
            crop_size (int | None): Final crop size. If None, uses image_size. Default=None.
        """
        self.image_size = image_size
        self.transformation_types = transformation_types

        self.transforms, self.num_views = dict(), dict()
        for t in transformation_types:
            if t == "simclr":
                self.transforms[t], self.num_views[t] = self._init_simclr_transform(
                    num_views=num_views, crop_scale=crop_scale, crop_size=crop_size
                )
            elif t == "dino":
                # DINO v1: 2 global crops + 6 local crops
                self.transforms[t], self.num_views[t] = self._init_dino_transform()
            else:
                raise NotImplementedError(f"Transformation type '{t}' not supported.")

    def _init_simclr_transform(
        self,
        num_views: int = 2,
        crop_scale: tuple[float, float] = (0.15, 1.0),
        crop_size: int | None = None,
    ) -> tuple[transforms.Compose, int]:
        """Initialize SimCLR transform pipeline.

        Args:
            num_views (int): Number of augmented views to create per image. Default=2.
            crop_scale (tuple[float, float]): Scale range for RandomResizedCrop (min, max).
                                             Default=(0.15, 1.0). Lower values = smaller crops.
            crop_size (int | None): Final crop size. If None, uses self.image_size. Default=None.

        Returns:
            tuple[transforms.Compose, int]: The transform and number of views.
        """
        s = 1.0  # strength of color jitter
        color_jitter = transforms.ColorJitter(
            brightness=0.8 * s,
            contrast=0.8 * s,
            saturation=0.8 * s,
            hue=0.2 * s,
        )
        kernel_size = max(3, int(0.1 * self.image_size) // 2 * 2 + 1)
        final_size = crop_size if crop_size is not None else self.image_size
        transform = transforms.Compose(
            [
                # the SimCLR paper uses (0.08, 1.0) scale but they have larger images (224x224)
                # we use (0.15, 1.0) for smaller images (96x96)
                # crop_scale controls the size range: (0.15, 1.0) means crops are 15%-100% of image area
                transforms.RandomResizedCrop(final_size, scale=crop_scale),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))],
                    p=0.5,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform, num_views

    def _init_dino_transform(self) -> tuple[dict[str, transforms.Compose], int]:
        """Initialize DINO v1 multi-crop transform pipeline.
        
        DINO v1 uses:
        - 2 global crops (larger, full augmentation)
        - 6 local crops (smaller, minimal augmentation, size 36-48 pixels)
        
        Returns:
            tuple[dict[str, transforms.Compose], int]: Dictionary with 'global' and 'local' transforms, and total num_views (8).
        """
        s = 1.0  # strength of color jitter
        
        # Global crop transform (full augmentation)
        global_color_jitter = transforms.ColorJitter(
            brightness=0.4 * s,
            contrast=0.4 * s,
            saturation=0.4 * s,
            hue=0.1 * s,
        )
        global_kernel_size = max(3, int(0.1 * self.image_size) // 2 * 2 + 1)
        
        # Global crops: use image_size, scale (0.14, 1.0) as per DINO paper
        global_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size, scale=(0.14, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([global_color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=global_kernel_size, sigma=(0.1, 2.0))],
                    p=1.0,  # Always apply blur for global crops in DINO
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        
        # Local crop transform (minimal augmentation)
        # Local crops: crop area corresponds to 36-48 pixels in original image
        # For a 96x96 image: 36px = 0.375 linear, 48px = 0.5 linear
        # Scale uses area: (0.375^2, 0.5^2) = (0.14, 0.25) for 36-48 pixel crops
        # All local crops resized to 48 pixels for consistent tensor shapes
        local_transforms = []
        for _ in range(6):
            local_transform = transforms.Compose(
                [
                    # Scale (0.14, 0.25) gives crop areas of 36-48 pixels in original image
                    transforms.RandomResizedCrop(48, scale=(0.14, 0.25)),
                    transforms.RandomHorizontalFlip(),
                    # No color jitter, no grayscale, no blur for local crops (DINO v1)
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            local_transforms.append(local_transform)
        
        transforms_dict = {
            "global": global_transform,
            "local": local_transforms,  # List of 6 transforms with different sizes
        }
        
        # 2 global + 6 local = 8 total views
        num_views = 8
        
        return transforms_dict, num_views

    def __call__(self, x: Image.Image) -> dict[str, torch.Tensor]:
        """Apply the transform to an image.

        Args:
            x (PIL.Image): Input image.

        Returns:
            dict[str, torch.Tensor]: Dictionary of transformed views.
        """
        # Return the specified number of views
        out = dict()
        for t, transform in self.transforms.items():
            num_views = self.num_views[t]
            
            # Handle DINO multi-crop case
            if isinstance(transform, dict):
                # DINO: 2 global crops + 6 local crops
                global_crops = tuple(transform["global"](x) for _ in range(2))
                # Each local crop uses a different transform with size 36-48 pixels
                local_crops = tuple(local_tf(x) for local_tf in transform["local"])
                all_crops = global_crops + local_crops
                out[t] = torch.stack(all_crops, dim=0)  # [8, C, H, W]
            elif num_views == 1:
                out[t] = (transform(x),)
            else:
                out[t] = tuple(transform(x) for _ in range(num_views))
                out[t] = torch.stack(out[t], dim=0)
        return out
