"""Image Transformations for Self-Supervised Learning."""

import random

import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms


class GaussianBlur:
    """Apply Gaussian Blur to the PIL image."""

    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0) -> None:
        """Initialize GaussianBlur.

        Args:
            p (float): Probability of applying the blur.
            radius_min (float): Minimum radius for the blur.
            radius_max (float): Maximum radius for the blur.
        """
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply Gaussian Blur to the image with probability p.

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Blurred image or original image.
        """
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))


class Solarization:
    """Apply Solarization to the PIL image."""

    def __init__(self, p: float) -> None:
        """Initialize Solarization.

        Args:
            p (float): Probability of applying solarization.
        """
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply Solarization to the image with probability p.

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Solarized image or original image.
        """
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


class ImageTransform:
    """Image Transformation pipeline for self-supervised learning."""

    def __init__(
        self,
        image_size: int,
        num_local_crops: int = 6,
        local_crop_size: int = 36,
        global_crops_scale: tuple[float, float] = (0.4, 1.0),
        local_crops_scale: tuple[float, float] = (0.05, 0.4),
    ) -> None:
        """Initialize the image transform.

        Args:
            image_size (int): Size to which images are resized/cropped.
            num_local_crops (int): Number of local crops for multi-crop (DINO). Default=6.
            local_crop_size (int): Size of local crops for multi-crop (DINO). Default=36.
            global_crops_scale (tuple[float, float]): Scale range for global crops. Default (0.4, 1.0).
            local_crops_scale (tuple[float, float]): Scale range for local crops. Default (0.05, 0.4).
        """
        self.image_size = image_size
        self.local_crop_size = local_crop_size
        self.num_local_crops = num_local_crops
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale

        self.transforms, self.num_views = self._init_dino_transform()

    def _init_dino_transform(self) -> tuple[dict[str, transforms.Compose], int]:
        """Initialize DINO v1 multi-crop transform pipeline.

        DINO v1 uses:
        - 2 global crops (larger, full augmentation)
        - 6 local crops (smaller, minimal augmentation, size 36-48 pixels)

        Returns:
            tuple[dict[str, transforms.Compose], int]: Dictionary with transforms and total num_views.
        """
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                # TODO: Find out mean/std for the entire dataset
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # first global crop
        global_transformation_1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size, scale=self.global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(1.0),
                normalize,
            ]
        )
        # second global crop
        global_transformation_2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size, scale=self.global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(0.1),
                Solarization(0.2),
                normalize,
            ]
        )

        # transformation for the local small crops
        local_transformation = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.local_crop_size, scale=self.local_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(p=0.5),
                normalize,
            ]
        )

        transforms_dict = {
            "global_1": global_transformation_1,
            "global_2": global_transformation_2,
            "local": local_transformation,  # List of transforms with different sizes
        }

        # 2 global + n local = 2+n total views
        num_views = 2 + self.num_local_crops
        return transforms_dict, num_views

    def __call__(self, x: Image.Image) -> list[torch.Tensor]:
        """Apply the transform to an image.

        Args:
            x (PIL.Image): Input image.

        Returns:
            list[torch.Tensor]: Dictionary of transformed views.
        """
        # Return the specified number of views
        out = list()
        out.append(self.transforms["global_1"](x))
        out.append(self.transforms["global_2"](x))
        for _ in range(self.num_local_crops):
            out.append(self.transforms["local"](x))
        return out
