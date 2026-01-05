# Image-SSL

An end-to-end framework for experimenting with self-supervised visual representation learning based on transformer architectures, designed for the NYU Deep Learning (CSCI-GA 2672) Fall 2025 Final Project.

## Table of contents

- [Project Overview](#project-overview)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
    - [Pretraining](#pretraining)
    - [Linear Probing & KNN Evaluation](#linear-probing--knn-evaluation)
    - [HPC / SLURM Usage](#hpc--slurm-usage)
- [Data](#data)
- [Saved Models & Results](#saved-models--results)
- [Evaluation & Submission](#evaluation--submission)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

Image-SSL is a modular codebase for self-supervised learning (SSL) from images, tailored to the parameter/boundary constraints of the CSCI-GA 2672 Final Project at NYU. It supports:

- **SSL pretraining of ViT** (Vision Transformer) models with DINO loss and multi-crop augmentation.
- Clean separation of pretraining and downstream evaluation (linear probing, KNN).
- Robust data processing pipelines built on top of HuggingFace Datasets and torchvision.
- SLURM job script for streamlined training on academic clusters.
- Reproducible and scalable workflows with integrated logging (Weights & Biases and Hugging Face Hub).

The entire pipeline is designed for reproducibility, research ease, and compliance with course requirements (random init, <100M params, approved data, etc.).

---

## Repository Layout

Key files and folders:

- `src/` — main source (pretraining, dataset, models, trainers, utils scripts):
    - `pretrain.py` — entrypoint for SSL pretraining.
    - `models/` — vision transformer and DINO head implementations:
        - `vit.py`, `pretraining.py`, `modules/` (attention, mlp, patch, dino, transformer, drop_path).
    - `dataset/` — dataset and transform pipelines.
    - `trainers/` — training logic and loss modules.
    - `utils/` — argument parsing, data statistics.
    - `slurm/` — `pretrain.slurm` for training on clusters.
- `eval/` — scripts and configs for evaluation/competition submission:
    - `testset_X/` — KNN/linear probe scripts and dataset prep for each eval split.
- `docs/` — project design docs and NYU project specification.
- `requirements.txt`, `pyproject.toml` — dependencies.

---

## Installation

Recommended: use `uv` for environment management.

1. Install `uv` [from here](https://docs.astral.sh/uv/getting-started/installation/).

2. Create and activate a Python 3.12 virtual environment:

```bash
uv venv --python=3.12
```

3. Install all required dependencies:

```bash
uv sync --all-extras
```

---

## Quickstart

### Pretraining

To launch SSL pretraining on the provided dataset:

```bash
python src/pretrain.py \
    --image-size 96 \
    --batch-size 16 \
    --num-epochs 50 \
    --save-dir ./saved_models/exp_01 \
    --device cuda \
    --use-wandb
```

**Important Arguments:**

- `--image-size`, `--patch-size`: control transformer granularity.
- `--dino-*`: tune DINO head and loss for your experiments.
- Other core options: `--optimizer-class`, `--lr-scheduler-class`, `--learning-rate`, `--val-split`, etc.
- See all options in `src/utils/parser.py`.

**Checkpoints** are saved in `saved_models/`, with `step_XXXX`, `best_model`, and `final_model` folders.

### Linear Probing & KNN Evaluation

Use the scripts in `eval/` to conduct downstream evaluation and produce Kaggle submissions.

**KNN Baseline:**
```bash
python eval/testset_X/create_submission_knn.py \
    --data_dir path/to/kaggle_data \
    --output submission.csv \
    --model_name ./saved_models/best_model \
    --model_class pretraining \
    --resolution 96
```

**Linear Probe Baseline:**
```bash
python eval/testset_X/create_submission_linear.py \
    --data_dir path/to/kaggle_data \
    --output submission.csv \
    --model_name ./saved_models/best_model \
    --epochs 100 \
    --resolution 96
```

### HPC / SLURM Usage

`src/slurm/pretrain.slurm` is ready for NYU's Greene or other clusters. Edit the script's header for:

- Your NetID/email.
- The working directory and cluster environment details.
- SLURM job parameters (GPUs, time, memory, etc).

Run with:

```bash
sbatch src/slurm/pretrain.slurm
```

Outputs are logged to the specified `logs/` directory.

---

## Data

- **Pretraining:** Uses HuggingFace dataset `"tsbpp/fall2025_deeplearning"` (see loader in `src/dataset/loader.py`). No labels are used.
- **Augmentation:** DINO-style multi-crop pipelines (see `src/dataset/transform.py`).
- Dataset structure and splits are managed automatically; validation split optional.
- For evaluation, follow the instructions and data structures in each `eval/testset_X/README.md`:

    Example:

    ```
    data/
      ├── train/
      ├── val/
      ├── test/
      ├── train_labels.csv
      ├── val_labels.csv
      ├── test_images.csv
      └── sample_submission.csv
    ```

---

## Saved Models & Results

- All checkpoints, configs, and logs are saved under the `saved_models/` directory specified via `--save-dir`.
- Each checkpoint contains all model weights and key training metadata (e.g., config.json).
- Optionally, models may be uploaded to HuggingFace Hub directly via argument flags.

---

## Evaluation & Submission

1. **Feature Extraction:** Scripts use your trained backbone to extract features on validation/test sets for linear-probe or KNN classification. Models are loaded and frozen appropriately, in line with the project spec (see evaluation rules).
2. **Submission:** Both KNN and linear probing scripts output a `submission.csv`. Double check the required format and data constraints in the Evaluation section.
3. **Compliance:** The codebase strictly separates pretraining and evaluation to prevent any violation of the challenge rules (no adaptation, random initialization, parameter/size limits).
4. **Reproducibility:** Configurations, seeds, and feature normalizations are logged and controlled for repeatability.

## Acknowledgements

This project is based on the Fall 2025 NYU Deep Learning final project formulation and includes contributions and best practices from the DINO, ViT, and popular SSL research ecosystems. Additional baseline and loader logic reference methods from [facebookresearch/dino](https://github.com/facebookresearch/dino), HuggingFace Transformers, and PyTorch Lightning.
