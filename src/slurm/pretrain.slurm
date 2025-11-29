#!/bin/bash

#SBATCH --job-name=images_ssl
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=rtx8000
## SBATCH --account=csci_ga_2572-2025fa
#SBATCH --open-mode=append
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/scratch/ab12057/image-ssl       # Update with your NYU NetID
#SBATCH --mail-type=all
#SBATCH --mail-user=ab12057@nyu.edu             # Update with your NYU NetID

set -e

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Current working directory: $(pwd)"

nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the environment
source .venv/bin/activate

# Data and dataset parameters
VAL_SPLIT=""                    # Proportion of data for validation (empty = use full dataset for training)

# Processor and model parameters
IMAGE_SIZE=96                   # Size of input images
PATCH_SIZE=8                    # Size of image patches
IN_CHANNELS=3                   # Number of input channels (3 for RGB)
HIDDEN_SIZE=384                 # Dimension of encoder layers
NUM_HIDDEN_LAYERS=12            # Number of hidden layers
NUM_ATTENTION_HEADS=6           # Number of attention heads
QKV_BIAS=true                   # Add bias to QKV projections
INTERMEDIATE_SIZE=1536          # Dimension of feedforward layers
DROPOUT_HIDDEN=0.0              # Dropout probability for model
DROPOUT_ATTENTION=0.0           # Dropout probability for attention
DROPOUT_PATH=0.0                # Dropout probability for stochastic depth

# DINO / multi-crop parameters
DINO_OUT_DIM=65536
DINO_USE_BN=true
DINO_NORM_LAST_LAYER=true
DINO_NUM_LAYERS=3
DINO_HIDDEN_DIM=2048
DINO_BOTTLENECK_DIM=256
DINO_BASE_TEACHER_TEMP=0.04
DINO_FINAL_TEACHER_TEMP=0.04
DINO_WARMUP_EPOCHS=0
NUM_LOCAL_CROPS=6
LOCAL_CROP_SIZE=36
GLOBAL_CROPS_SCALE=(0.4 1.0)
LOCAL_CROPS_SCALE=(0.05 0.4)

# Training hyperparameters
CHECKPOINT=""                   # Path or model ID of pre-trained checkpoint (empty = train from scratch)
BATCH_SIZE=256                  # Batch size for training
NUM_EPOCHS=5                    # Number of training epochs
LEARNING_RATE=3e-4              # Learning rate for optimizer
OPTIMIZER_CLASS="adamw"         # Choices: adamw, adam, sgd
# Weight decay scheduling parameters
BASE_WD=0.04
FINAL_WD=0.4
# Momentum scheduling parameters (used by some losses / teacher models)
BASE_MOMENTUM=0.996
FINAL_MOMENTUM=1.0
# Learning rate scheduling parameters
LR_SCHEDULER_CLASS="cosine"        # Choices: cosine, exponential
WARMUP_RATIO=0.1                # Ratio of warmup steps to total training steps

# Logging and checkpointing
LOG_INTERVAL_STEPS=15         # Log training loss every N steps
SAVE_INTERVAL_STEPS=315        # Save model checkpoint every N steps
SAVE_DIR="./saved_models/vit-s8-L6"       # Directory to save checkpoints
SAVE_LATEST=true               # Overwrite latest checkpoint instead of saving per step
SAVE_BEST=true                 # Track and save best model based on specified loss
LOSS_METRIC_FOR_BEST_MODEL="train"    # Metric to use for best model: train, val

# Weights & Biases configuration
USE_WANDB=true                 # Enable W&B experiment tracking
WANDB_ENTITY="image-ssl"        # W&B entity name
WANDB_PROJECT="pretraining"     # W&B project name
WANDB_NAME="vit-s8-L6"                   # W&B run name (empty = auto-generated)

# Hugging Face Hub Configuration
UPLOAD_MODEL_TO_HUB=true      # Upload model to Hugging Face Hub
REPO_ID="image-ssl/vit-s8-L6"                   # Hugging Face Hub repository ID

# System Configuration
DEVICE="cuda:0"                 # Torch device (cuda:0, cpu, etc.)
SEED=42                         # Random seed for reproducibility

# Start building the command
CMD="uv run src/pretrain.py"

# Add validation split if specified
if [ -n "$VAL_SPLIT" ]; then
    CMD="$CMD --val-split $VAL_SPLIT"
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

# DINO / multi-crop flags (only pass those that are set in this script)
CMD="$CMD --dino-out-dim $DINO_OUT_DIM"
if [ "$DINO_USE_BN" = true ]; then
    CMD="$CMD --dino-use-bn"
fi
if [ "$DINO_NORM_LAST_LAYER" = true ]; then
    CMD="$CMD --dino-norm-last-layer"
fi
CMD="$CMD --dino-num-layers $DINO_NUM_LAYERS"
CMD="$CMD --dino-hidden-dim $DINO_HIDDEN_DIM"
CMD="$CMD --dino-bottleneck-dim $DINO_BOTTLENECK_DIM"
CMD="$CMD --dino-base-teacher-temp $DINO_BASE_TEACHER_TEMP"
CMD="$CMD --dino-final-teacher-temp $DINO_FINAL_TEACHER_TEMP"
CMD="$CMD --dino-warmup-epochs $DINO_WARMUP_EPOCHS"
CMD="$CMD --num-local-crops $NUM_LOCAL_CROPS"
CMD="$CMD --local-crop-size $LOCAL_CROP_SIZE"
CMD="$CMD --global-crops-scale ${GLOBAL_CROPS_SCALE[0]} ${GLOBAL_CROPS_SCALE[1]}"
CMD="$CMD --local-crops-scale ${LOCAL_CROPS_SCALE[0]} ${LOCAL_CROPS_SCALE[1]}"

# Add checkpoint if specified
if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

# Add training hyperparameters
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --num-epochs $NUM_EPOCHS"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --optimizer-class $OPTIMIZER_CLASS"
CMD="$CMD --base-wd $BASE_WD"
CMD="$CMD --final-wd $FINAL_WD"
CMD="$CMD --base-momentum $BASE_MOMENTUM"
CMD="$CMD --final-momentum $FINAL_MOMENTUM"
CMD="$CMD --lr-scheduler-class $LR_SCHEDULER_CLASS"
CMD="$CMD --warmup-ratio $WARMUP_RATIO"

# Add logging and checkpointing parameters
CMD="$CMD --log-interval-steps $LOG_INTERVAL_STEPS"
CMD="$CMD --save-interval-steps $SAVE_INTERVAL_STEPS"
CMD="$CMD --save-dir $SAVE_DIR"
CMD="$CMD --loss-metric-for-best-model $LOSS_METRIC_FOR_BEST_MODEL"

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

# W&B configuration
if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use-wandb"
    CMD="$CMD --wandb-entity $WANDB_ENTITY"
    CMD="$CMD --wandb-project $WANDB_PROJECT"
    if [ -n "$WANDB_NAME" ]; then
        CMD="$CMD --wandb-name $WANDB_NAME"
    fi
fi

# Hugging Face Hub configuration
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
echo "Running: $CMD"
eval $CMD

# Deactivate the environment
deactivate

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
