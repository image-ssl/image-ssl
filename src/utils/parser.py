"""Utility functions for parsing command-line arguments."""

import argparse


def parse_pretrain_args() -> argparse.Namespace:
    """Parse command-line arguments for pre-training ViT models.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Pre-train a ViT model with specified architecture and hyperparameters."
    )

    # Data and dataset parameters
    parser.add_argument(
        "--val-split",
        type=float,
        default=None,
        help="Proportion of data to use for validation. Default=None (use full dataset for training).",
    )

    # Processor and model parameters
    parser.add_argument(
        "--image-size",
        type=int,
        default=96,
        help="Size of the input images. Default=96.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=8,
        help="Size of the image patches. Default=8.",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=3,
        help="Number of input channels (3 for RGB images). Default=3.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=384,
        help="Dimension of the encoder layers. Default=256.",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=12,
        help="Number of hidden layers in the model. Default=6.",
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=6,
        help="Number of attention heads in the model. Default=6.",
    )
    parser.add_argument(
        "--qkv-bias",
        action="store_true",
        help="If set, add bias to the QKV projections.",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=1536,
        help="Dimension of the feedforward layers. Default=3072.",
    )
    parser.add_argument(
        "--dropout-hidden",
        type=float,
        default=0.0,
        help="Dropout probability for the model. Default=0.0.",
    )
    parser.add_argument(
        "--dropout-attention",
        type=float,
        default=0.0,
        help="Dropout probability for attention layers. Default=0.0.",
    )
    parser.add_argument(
        "--dropout-path",
        type=float,
        default=0.0,
        help="Dropout probability for stochastic depth. Default=0.0.",
    )
    parser.add_argument(
        "--dino-out-dim",
        type=int,
        default=65536,
        help="Output dimension for DINO head. Default=65536.",
    )
    parser.add_argument(
        "--dino-use-bn",
        action="store_true",
        help="Whether to use batch norm in DINO head. Default=False.",
    )
    parser.add_argument(
        "--dino-norm-last-layer",
        action="store_true",
        help="Whether to normalize last layer in DINO head. Default=False.",
    )
    parser.add_argument(
        "--dino-num-layers",
        type=int,
        default=3,
        help="Number of layers in DINO head. Default=3.",
    )
    parser.add_argument(
        "--dino-hidden-dim",
        type=int,
        default=2048,
        help="Hidden dimension in DINO head. Default=2048.",
    )
    parser.add_argument(
        "--dino-bottleneck-dim",
        type=int,
        default=256,
        help="Bottleneck dimension in DINO head. Default=256.",
    )
    parser.add_argument(
        "--dino-base-teacher-temp",
        type=float,
        default=0.04,
        help="Base temperature for the DINO teacher. Default=0.04.",
    )
    parser.add_argument(
        "--dino-final-teacher-temp",
        type=float,
        default=0.04,
        help="Final temperature for the DINO teacher. Default=0.04.",
    )
    parser.add_argument(
        "--dino-warmup-epochs",
        type=int,
        default=0,
        help="Number of warmup epochs for DINO teacher temperature. Default=0.",
    )
    parser.add_argument(
        "--num-local-crops",
        type=int,
        default=6,
        help="Number of local crops for multi-crop (DINO). Default=6.",
    )
    parser.add_argument(
        "--local-crop-size",
        type=int,
        default=36,
        help="Size of local crops for multi-crop (DINO). Default=36.",
    )
    parser.add_argument(
        "--global-crops-scale",
        type=float,
        nargs=2,
        default=(0.4, 1.0),
        help="Scale range for global crops (DINO). Default=(0.4, 1.0).",
    )
    parser.add_argument(
        "--local-crops-scale",
        type=float,
        nargs=2,
        default=(0.05, 0.4),
        help="Scale range for local crops (DINO). Default=(0.05, 0.4).",
    )

    # Training hyperparameters
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path or model ID of a pre-trained checkpoint to load."
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training. Default=16.")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of training epochs. Default=20.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for optimizer. Default=3e-4.",
    )
    parser.add_argument(
        "--optimizer-class",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Optimizer class to use. Default='adamw'.",
    )
    parser.add_argument(
        "--base-wd",
        type=float,
        default=0.04,
        help="Weight decay (L2 regularization) factor for the optimizer. Default=0.05.",
    )
    parser.add_argument(
        "--final-wd",
        type=float,
        default=0.4,
        help="Final weight decay value for weight decay scheduler. Default=0.4.",
    )
    parser.add_argument(
        "--base-momentum",
        type=float,
        default=0.996,
        help="Base momentum for the teacher model. Default=0.996.",
    )
    parser.add_argument(
        "--final-momentum",
        type=float,
        default=1.0,
        help="Final momentum for the teacher model. Default=1.0.",
    )
    parser.add_argument(
        "--scheduler-class",
        type=str,
        default="cosine",
        choices=["cosine", "exponential"],
        help="Learning rate scheduler class to use. Default='cosine'.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Ratio of warmup steps to total training steps for the scheduler. Default=0.1.",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--log-interval-steps",
        type=int,
        default=100,
        help="Log training loss every N steps. Default=1000.",
    )
    parser.add_argument(
        "--save-interval-steps",
        type=int,
        default=1000,
        help="Save model checkpoint every N steps. Default=2000.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./saved_models",
        help="Directory to save model checkpoints. Default='./saved_models'.",
    )
    parser.add_argument(
        "--save-latest",
        action="store_true",
        help="If set, overwrite the latest checkpoint instead of saving per step.",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="If set, track and save the best model checkpoint based on training loss.",
    )

    # Weights & Biases integration
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="If set, use Weights & Biases for experiment tracking.",
    )
    parser.add_argument("--wandb-entity", type=str, default="image-ssl", help="Weights & Biases entity name.")
    parser.add_argument("--wandb-project", type=str, default="pretraining", help="Weights & Biases project name.")
    parser.add_argument("--wandb-name", type=str, default=None, help="Weights & Biases run name.")

    # HuggingFace Hub integration
    parser.add_argument(
        "--upload-model-to-hub",
        action="store_true",
        help="If set, upload the model to Hugging Face Hub.",
    )
    parser.add_argument("--repo-id", type=str, default=None, help="Hugging Face Hub repository ID.")

    # System parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility. Default=42.")

    return parser.parse_args()
