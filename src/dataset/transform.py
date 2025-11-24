"""Image Transformations for Self-Supervised Learning."""

import random

import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms


class GaussianBlur:
    """Apply Gaussian Blur to the PIL image."""

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))


class Solarization:
    """Apply Solarization to the PIL image."""

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


class ImageTransform:
    """Image Transformation pipeline for self-supervised learning."""

    def __init__(
        self,
        image_size: int,
        transformation_types: list[str],
        num_local_crops: int = 6,
        local_crop_size: int = 36,
        global_crops_scale: tuple[float, float] = (0.4, 1.0),
        local_crops_scale: tuple[float, float] = (0.05, 0.4),
    ) -> None:
        """Initialize the image transform.

        Args:
            image_size (int): Size to which images are resized/cropped.
            transformation_types (list[str]): Types of transformation pipelines.
            num_local_crops (int): Number of local crops for multi-crop (DINO). Default=6.
            local_crop_size (int): Size of local crops for multi-crop (DINO). Default=36.
            global_crops_scale (tuple[float, float]): Scale range for global crops. Default (0.4, 1.0).
            local_crops_scale (tuple[float, float]): Scale range for local crops. Default (0.05, 0.4).
        """
        self.image_size = image_size
        self.transformation_types = transformation_types
        self.local_crop_size = local_crop_size
        self.num_local_crops = num_local_crops
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale

        self.transforms, self.num_views = dict(), dict()
        for t in transformation_types:
            if t == "dino":
                # DINO v1: 2 global crops + n local crops
                self.transforms[t], self.num_views[t] = self._init_dino_transform()
            else:
                raise NotImplementedError(f"Transformation type '{t}' not supported.")

    def _init_dino_transform(self) -> tuple[dict[str, transforms.Compose], int]:
        """Initialize DINO v1 multi-crop transform pipeline.

        DINO v1 uses:
        - 2 global crops (larger, full augmentation)
        - 6 local crops (smaller, minimal augmentation, size 36-48 pixels)

        Returns:
            tuple[dict[str, transforms.Compose], int]: Dictionary with 'global' and 'local' transforms, and total num_views (8).
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
        out.append(self.transforms["dino"]["global_1"](x))
        out.append(self.transforms["dino"]["global_2"](x))
        for _ in range(self.num_local_crops):
            out.append(self.transforms["dino"]["local"](x))
        return out
