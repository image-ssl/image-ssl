"""Implementations of Vision Transformer models."""

import argparse
import inspect

import torch

from .vit import VisionTransformer


def init_model(args: argparse.Namespace, device: torch.device) -> VisionTransformer:
    """Initialize the model based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        device (torch.device): Device to load the model onto.

    Returns:
        VisionTransformer: Instantiated Vision Transformer model.
    """
    if args.checkpoint is not None:
        return VisionTransformer.from_pretrained(args.checkpoint).to(device)
    sig = inspect.signature(VisionTransformer.__init__)
    param_names = set(sig.parameters.keys())
    init_args = {k: v for k, v in vars(args).items() if k in param_names and k != "self"}
    return VisionTransformer(**init_args).to(device)


__all__ = ["VisionTransformer", "init_model"]
