"""Trainer modules for training, model management and logging."""

import argparse

import torch.nn as nn
from torch.utils.data import DataLoader

from .pretraining import PreTrainer

TRAINER_REGISTRY = {
    "pretraining": PreTrainer,
}


def init_trainer(model: nn.Module, train_loader: DataLoader, args: argparse.Namespace, cls: str) -> PreTrainer:
    """Initialize the Trainer with the given model and arguments.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The training dataloader.
        args (argparse.Namespace): Parsed command-line arguments.
        cls (str): Trainer class type.

    Returns:
        Trainer: An instance of the Trainer class.
    """
    cls = TRAINER_REGISTRY[cls]
    if args.checkpoint is not None:
        return cls.from_pretrained(args.checkpoint, model=model)
    trainer_kwargs = vars(args).copy()
    trainer_kwargs.pop("model", None)  # TODO: I don't remember why I did this lol, check if needed
    trainer_kwargs.setdefault("num_steps", len(train_loader) * args.num_epochs)
    return cls(model=model, **trainer_kwargs)


__all__ = ["PreTrainer"]
