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
    ) -> None:
        """Initialize the image transform.

        Args:
            image_size (int): Size to which images are resized/cropped.
            transformation_types (list[str]): Types of transformation pipelines.
        """
        self.image_size = image_size
        self.transformation_types = transformation_types

        self.transforms, self.num_views = dict(), dict()
        for t in transformation_types:
            if t == "simclr":
                self.transforms[t], self.num_views[t] = self._init_simclr_transform()
            else:
                raise NotImplementedError(f"Transformation type '{t}' not supported.")

    def _init_simclr_transform(self) -> tuple[transforms.Compose, int]:
        """Initialize SimCLR transform pipeline.

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
        transform = transforms.Compose(
            [
                # the SimCLR paper uses (0.08, 1.0) scale but they have larger images (224x224)
                # we use (0.15, 1.0) for smaller images (96x96)
                transforms.RandomResizedCrop(self.image_size, scale=(0.15, 1.0)),
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
        num_views = 2
        return transform, num_views

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
            if num_views == 1:
                out[t] = (transform(x),)
            else:
                out[t] = tuple(transform(x) for _ in range(num_views))
                out[t] = torch.stack(out[t], dim=0)
        return out
