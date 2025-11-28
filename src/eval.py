"""Kaggle-ready evaluation pipeline for CUB-200 SSL submission generation."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

import models
import utils


MODEL_CLASS_LOOKUP = {
    "base": models.VisionTransformer,
    "pretraining": models.VisionTransformerWithPretrainingHeads,
}


class CUBImageDataset(Dataset):
    """Image dataset backed by the Kaggle-style CUB directory."""

    def __init__(
        self,
        image_dir: Path,
        csv_path: Path,
        has_labels: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.df = pd.read_csv(csv_path)
        self.has_labels = has_labels and "class_id" in self.df.columns
        self.filenames = self.df["filename" if "filename" in self.df.columns else "id"].tolist()
        self.labels = self.df["class_id"].astype(int).tolist() if self.has_labels else None
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str] | tuple[torch.Tensor, str]:
        image_path = self.image_dir / self.filenames[idx]
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        if self.has_labels and self.labels is not None:
            return tensor, self.labels[idx], self.filenames[idx]
        return tensor, self.filenames[idx]


def _outputs_to_tensor(outputs: torch.Tensor | object) -> torch.Tensor:
    """Normalize heterogeneous model outputs to a single tensor representation."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "cls"):
        return getattr(outputs, "cls")
    raise TypeError(
        "Unsupported output type returned by the model. "
        "Ensure --model-class matches the checkpoint you are evaluating."
    )


def _prepare_device(device_str: str) -> torch.device:
    """Resolve a user-provided device string with graceful accelerator fallback."""
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    if device_str.startswith("mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def _load_model(args, device: torch.device) -> torch.nn.Module:
    model_cls = MODEL_CLASS_LOOKUP[args.model_class]
    model = model_cls.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint into {model_cls.__name__} ({total_params / 10**6:.2f}M parameters).")
    return model


def _extract_features_from_loader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
    encoder = model.encoder if hasattr(model, "encoder") else model
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    filenames: list[str] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            if len(batch) == 3:
                images, batch_labels, batch_names = batch
                labels.append(batch_labels)
            else:
                images, batch_names = batch
                batch_labels = None
            images = images.to(device, non_blocking=True)
            outputs = encoder(images)
            batch_features = _outputs_to_tensor(outputs).detach().cpu()
            features.append(batch_features)
            filenames.extend(batch_names)
    feature_tensor = torch.cat(features, dim=0) if features else torch.empty(0)
    label_tensor = torch.cat(labels, dim=0) if labels else None
    return feature_tensor, label_tensor, filenames


def _save_feature_bundle(path: str | None, **payload) -> None:
    if path is None:
        return
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)
    print(f"Saved features to {save_path}")


def _run_knn_classifier(
    feature_bank: torch.Tensor,
    feature_labels: torch.Tensor,
    query_features: torch.Tensor,
    k: int,
    num_classes: int,
    device: torch.device,
) -> list[int]:
    if feature_bank.numel() == 0 or query_features.numel() == 0:
        raise ValueError("Empty features provided to k-NN classifier.")
    k = max(1, min(k, feature_bank.shape[0]))
    bank = torch.nn.functional.normalize(feature_bank.to(device), dim=1)
    queries = torch.nn.functional.normalize(query_features.to(device), dim=1)
    sims = torch.matmul(queries, bank.T)
    top_sims, top_indices = torch.topk(sims, k=k, dim=1)
    labels = feature_labels.to(device)[top_indices]
    votes = torch.zeros(queries.shape[0], num_classes, device=device)
    votes.scatter_add_(1, labels, top_sims)
    preds = votes.argmax(dim=1)
    return preds.cpu().tolist()


def _validate_submission(submission_df: pd.DataFrame, sample_path: Path, num_classes: int) -> None:
    required_columns = ["id", "class_id"]
    if list(submission_df.columns) != required_columns:
        raise ValueError(f"Submission columns must be {required_columns}.")
    if submission_df["class_id"].min() < 0 or submission_df["class_id"].max() >= num_classes:
        raise ValueError("Predicted class ids fall outside the expected range.")
    if sample_path.exists():
        sample_df = pd.read_csv(sample_path)
        if list(sample_df.columns) == required_columns:
            missing = set(sample_df["id"]) - set(submission_df["id"])
            if missing:
                raise ValueError(f"Submission is missing {len(missing)} ids found in sample_submission.csv.")


def _run_cub_submission(args, model: torch.nn.Module, device: torch.device) -> None:
    data_dir = Path(args.cub_data_dir)
    train_csv = data_dir / "train_labels.csv"
    val_csv = data_dir / "val_labels.csv"
    test_csv = data_dir / "test_images.csv"
    sample_path = data_dir / "sample_submission.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            "CUB data directory is missing expected CSV files (train_labels.csv / test_images.csv). "
            "Please run prepare_cub200_for_kaggle.py first."
        )

    print(f"Loading CUB dataset from {data_dir}")
    train_dataset = CUBImageDataset(
        image_dir=data_dir / "train",
        csv_path=train_csv,
        has_labels=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.cub_batch_size,
        shuffle=False,
        num_workers=args.cub_num_workers,
        pin_memory=args.cub_pin_memory,
    )

    feature_bank, feature_labels, _ = _extract_features_from_loader(
        model, train_loader, device, desc="train features"
    )

    if feature_labels is None:
        raise RuntimeError("No labels were loaded for the training split; cannot run k-NN.")

    if args.cub_include_val:
        if not val_csv.exists():
            raise FileNotFoundError("Requested --cub-include-val but val_labels.csv was not found.")
        val_dataset = CUBImageDataset(
            image_dir=data_dir / "val",
            csv_path=val_csv,
            has_labels=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.cub_batch_size,
            shuffle=False,
            num_workers=args.cub_num_workers,
            pin_memory=args.cub_pin_memory,
        )
        val_features, val_labels, _ = _extract_features_from_loader(
            model, val_loader, device, desc="val features"
        )
        feature_bank = torch.cat([feature_bank, val_features], dim=0)
        feature_labels = torch.cat([feature_labels, val_labels], dim=0)
        print(f"Feature bank augmented with validation split ({feature_bank.shape[0]} samples total).")

    test_dataset = CUBImageDataset(
        image_dir=data_dir / "test",
        csv_path=test_csv,
        has_labels=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.cub_batch_size,
        shuffle=False,
        num_workers=args.cub_num_workers,
        pin_memory=args.cub_pin_memory,
    )
    test_features, _, test_filenames = _extract_features_from_loader(
        model, test_loader, device, desc="test features"
    )

    _save_feature_bundle(
        args.save_train_features,
        features=feature_bank,
        labels=feature_labels,
    )
    _save_feature_bundle(
        args.save_test_features,
        features=test_features,
        filenames=test_filenames,
    )

    predictions = _run_knn_classifier(
        feature_bank=feature_bank,
        feature_labels=feature_labels.long(),
        query_features=test_features,
        k=args.cub_k,
        num_classes=args.cub_num_classes,
        device=device,
    )

    submission_df = pd.DataFrame(
        {
            "id": test_filenames,
            "class_id": predictions,
        }
    )
    _validate_submission(submission_df, sample_path, args.cub_num_classes)
    submission_path = Path(args.submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file written to {submission_path} ({len(submission_df)} predictions).")


if __name__ == "__main__":
    args = utils.parse_eval_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Device setup
    device = _prepare_device(args.device)
    print(f"Using device: {device}")

    # Restore checkpoint
    model = _load_model(args, device)

    # Generate Kaggle submission
    _run_cub_submission(args, model, device)

