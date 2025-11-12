#!/bin/bash

# Automatically get NetID from current user
NETID=$(whoami)

#SBATCH --job-name=${NETID}_images_ssl
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
## SBATCH --partition=a100_1
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/scratch/${NETID}/image-ssl
#SBATCH --mail-type=all
#SBATCH --mail-user=${NETID}@nyu.edu


echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Current working directory: $(pwd)"
echo "Running as user: ${NETID}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the environment
source .venv/bin/activate

# Run the training script with pretrain.sh configurations
bash src/sh/pretrain.sh

# Deactivate the environment
deactivate

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
