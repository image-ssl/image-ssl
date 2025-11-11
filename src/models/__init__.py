"""Implementations of Vision Transformer models."""

import argparse
import inspect

import torch

from .pretraining import VisionTransformerWithPretrainingHeads
from .vit import VisionTransformer

MODEL_REGISTRY = {
    "base": VisionTransformer,
    "pretraining": VisionTransformerWithPretrainingHeads,
}


def _construct_init_args(cls: type, args: argparse.Namespace) -> dict:
    """Construct a dictionary of initialization arguments for the given class based on the provided argparse.Namespace.

    Args:
        cls (type): The class for which to construct initialization arguments.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: A dictionary of initialization arguments.
    """
    sig = inspect.signature(cls.__init__)
    param_names = set(sig.parameters.keys())
    return {k: v for k, v in vars(args).items() if k in param_names and k != "self"}


def init_model(
    args: argparse.Namespace, device: torch.device, cls: str
) -> VisionTransformer | VisionTransformerWithPretrainingHeads:
    """Initialize the model based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        device (torch.device): Device to load the model onto.
        cls (str): The class of the model to initialize.

    Returns:
        VisionTransformer | VisionTransformerWithPretrainingHeads: Instantiated model.
    """
    cls = MODEL_REGISTRY[cls]
    if args.checkpoint is not None:
        return cls.from_pretrained(args.checkpoint).to(device)
    init_args = _construct_init_args(cls, args)
    return cls(**init_args).to(device)


__all__ = ["VisionTransformer", "VisionTransformerWithPretrainingHeads", "init_model"]
