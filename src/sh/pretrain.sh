#!/bin/bash

set -e

# Data and dataset parameters
VAL_SPLIT=""                    # Proportion of data for validation (empty = use full dataset for training)
PRETRAIN_OBJECTIVES=(
    "simclr"
)  # SSL objectives: simclr, mae, byol, dino (can specify multiple)

# Processor and model parameters
IMAGE_SIZE=96                   # Size of input images
PATCH_SIZE=8                    # Size of image patches
IN_CHANNELS=3                   # Number of input channels (3 for RGB)
HIDDEN_SIZE=768                 # Dimension of encoder layers
NUM_HIDDEN_LAYERS=12            # Number of hidden layers
NUM_ATTENTION_HEADS=12          # Number of attention heads
QKV_BIAS=true                   # Add bias to QKV projections
INTERMEDIATE_SIZE=3072          # Dimension of feedforward layers
DROPOUT_HIDDEN=0.0              # Dropout probability for model
DROPOUT_ATTENTION=0.0           # Dropout probability for attention
DROPOUT_PATH=0.0                # Dropout probability for stochastic depth

# Training hyperparameters
CHECKPOINT=""                   # Path or model ID of pre-trained checkpoint (empty = train from scratch)
BATCH_SIZE=16                   # Batch size for training
NUM_EPOCHS=20                   # Number of training epochs
LEARNING_RATE=3e-4              # Learning rate for optimizer
OPTIMIZER_CLASS="adamw"         # Choices: adamw, adam, sgd
WEIGHT_DECAY=0.05               # Weight decay (L2 regularization)
SCHEDULER_CLASS="cosine"        # Choices: cosine, exponential
WARMUP_RATIO=0.1                # Ratio of warmup steps to total training steps
SIMCLR_TEMPERATURE=0.5          # Temperature parameter for SimCLR loss

# Logging and checkpointing
LOG_INTERVAL_STEPS=1000         # Log training loss every N steps
SAVE_INTERVAL_STEPS=2000        # Save model checkpoint every N steps
SAVE_DIR="./saved_models"       # Directory to save checkpoints
SAVE_LATEST=false               # Overwrite latest checkpoint instead of saving per step
SAVE_BEST=false                 # Track and save best model based on training loss

# Weights & Biases configuration
USE_WANDB=true                  # Enable W&B experiment tracking
WANDB_ENTITY="image-ssl"        # W&B entity name
WANDB_PROJECT="pretraining"     # W&B project name
WANDB_NAME=""                   # W&B run name (empty = auto-generated)

# Hugging Face Hub Configuration
UPLOAD_MODEL_TO_HUB=true       # Upload model to Hugging Face Hub
REPO_ID="image-ssl/simclr"     # Hugging Face Hub repository ID

# System Configuration
DEVICE="cuda:0"                 # Torch device (cuda:0, cpu, etc.)
SEED=42                         # Random seed for reproducibility

# Start building the command
CMD="uv run src/pretrain.py"

# Add validation split if specified
if [ -n "$VAL_SPLIT" ]; then
    CMD="$CMD --val-split $VAL_SPLIT"
fi

# Add pretraining objectives
if [ ${#PRETRAIN_OBJECTIVES[@]} -gt 0 ]; then
    CMD="$CMD --pretrain-objectives ${PRETRAIN_OBJECTIVES[@]}"
fi

# Add model architecture parameters
CMD="$CMD --image-size $IMAGE_SIZE"
CMD="$CMD --patch-size $PATCH_SIZE"
CMD="$CMD --in-channels $IN_CHANNELS"
CMD="$CMD --hidden-size $HIDDEN_SIZE"
CMD="$CMD --num-hidden-layers $NUM_HIDDEN_LAYERS"
CMD="$CMD --num-attention-heads $NUM_ATTENTION_HEADS"
CMD="$CMD --intermediate-size $INTERMEDIATE_SIZE"
CMD="$CMD --dropout-hidden $DROPOUT_HIDDEN"
CMD="$CMD --dropout-attention $DROPOUT_ATTENTION"
CMD="$CMD --dropout-path $DROPOUT_PATH"

# Add checkpoint if specified
if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

# Add training hyperparameters
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --num-epochs $NUM_EPOCHS"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --optimizer-class $OPTIMIZER_CLASS"
CMD="$CMD --weight-decay $WEIGHT_DECAY"
CMD="$CMD --scheduler-class $SCHEDULER_CLASS"
CMD="$CMD --warmup-ratio $WARMUP_RATIO"
CMD="$CMD --simclr-temperature $SIMCLR_TEMPERATURE"

# Add logging and checkpointing parameters
CMD="$CMD --log-interval-steps $LOG_INTERVAL_STEPS"
CMD="$CMD --save-interval-steps $SAVE_INTERVAL_STEPS"
CMD="$CMD --save-dir $SAVE_DIR"

# Add boolean flags
if [ "$QKV_BIAS" = true ]; then
    CMD="$CMD --qkv-bias"
fi

if [ "$SAVE_LATEST" = true ]; then
    CMD="$CMD --save-latest"
fi

if [ "$SAVE_BEST" = true ]; then
    CMD="$CMD --save-best"
fi

# Add W&B configuration
if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use-wandb"
    CMD="$CMD --wandb-entity $WANDB_ENTITY"
    CMD="$CMD --wandb-project $WANDB_PROJECT"
    if [ -n "$WANDB_NAME" ]; then
        CMD="$CMD --wandb-name $WANDB_NAME"
    fi
fi

# Add Hugging Face Hub configuration
if [ "$UPLOAD_MODEL_TO_HUB" = true ]; then
    CMD="$CMD --upload-model-to-hub"
    if [ -n "$REPO_ID" ]; then
        CMD="$CMD --repo-id $REPO_ID"
    fi
fi

# Add system configuration
CMD="$CMD --device $DEVICE"
CMD="$CMD --seed $SEED"

# Delete previous model checkpoints if SAVE_DIR exists
if [ -d "$SAVE_DIR" ]; then
    rm -rf "${SAVE_DIR}/"*
fi

# Execute the command
eval $CMD
