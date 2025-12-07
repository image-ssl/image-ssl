"""
Create Kaggle Submission with Linear Probing Classifier
========================================================

This script provides a linear probing baseline using:
- Pretrained SSL model for feature extraction (frozen encoder)
- Linear classifier trained on extracted features

NOTE: This is a BASELINE example. For the competition, you MUST:
- Train your own SSL model from scratch (no pretrained weights!)
- This script is just to understand the submission format

Usage:
    python create_submission_linear.py \
        --data_dir ./kaggle_data \
        --output submission.csv \
        --max_iter 1000
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import argparse
from torchvision import transforms
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

from src.models import VisionTransformer, VisionTransformerWithPretrainingHeads


# ============================================================================
#                          MODEL SECTION (Modular)
# ============================================================================

class FeatureExtractor:
    """
    Modular feature extractor - REPLACE THIS with your own SSL model!
    
    This example uses pretrained DINO, but for the competition you must
    train your own model from scratch.
    """
    
    def __init__(self, model_name, device='cuda', model_class='auto'):
        """
        Initialize feature extractor.
        
        Args:
            model_name: HuggingFace model name or local path
            device: 'cuda', 'mps', or 'cpu'
            model_class: 'auto', 'base', or 'pretraining'
        """
        print(f"Loading model: {model_name}")
        
        # Try to determine model class automatically if not specified
        if model_class == 'auto':
            # Try pretraining first (most common for SSL models)
            try:
                self.model = VisionTransformerWithPretrainingHeads.from_pretrained(model_name)
                self.model_class = 'pretraining'
                print("  ✓ Loaded as VisionTransformerWithPretrainingHeads")
            except Exception as e1:
                try:
                    self.model = VisionTransformer.from_pretrained(model_name)
                    self.model_class = 'base'
                    print("  ✓ Loaded as VisionTransformer")
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to load model from {model_name}.\n"
                        f"Tried pretraining: {e1}\n"
                        f"Tried base: {e2}\n"
                        f"Make sure it's a valid checkpoint saved with PyTorchModelHubMixin."
                    )
        elif model_class == 'pretraining':
            self.model = VisionTransformerWithPretrainingHeads.from_pretrained(model_name)
            self.model_class = 'pretraining'
            print("  ✓ Loaded as VisionTransformerWithPretrainingHeads")
        else:
            self.model = VisionTransformer.from_pretrained(model_name)
            self.model_class = 'base'
            print("  ✓ Loaded as VisionTransformer")
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device
        
    def extract_batch_features(self, images):
        """
        Extract CLS features from the encoder ONLY.
        Supports DINO, ViT, and HuggingFace models.
        """

        # Correct normalization
        normalize = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=[0.519446, 0.497993, 0.470205],
                std=[0.305701, 0.301156, 0.312470]
            )
        ])

        # Prepare batch
        inputs = [normalize(img) for img in images]
        inputs = torch.stack(inputs).to(self.device)

        with torch.no_grad():
            # --------------------------------------------
            # 🔥 BEST PATH: DINO models with intermediate layers
            # --------------------------------------------
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "get_intermediate_layers"):
                feats = self.model.encoder.get_intermediate_layers(inputs, n=1)[0][:, 0]

            # --------------------------------------------
            # 🔥 Common ViT models (forward_features)
            # --------------------------------------------
            elif hasattr(self.model, "forward_features"):
                out = self.model.forward_features(inputs)

                # If dict output (HF / DINOv2)
                if isinstance(out, dict):
                    if "x_norm_clstoken" in out:
                        feats = out["x_norm_clstoken"]
                    elif "cls_token" in out:
                        feats = out["cls_token"]
                    else:
                        raise ValueError(f"Cannot find CLS token in output dict: {out.keys()}")

                # If tokens (B, N, D)
                elif out.ndim == 3:
                    feats = out[:, 0]

                # Already CLS
                else:
                    feats = out

            # --------------------------------------------
            # 🔥 FALLBACK — call encoder directly
            # (never call self.model()!)
            # --------------------------------------------
            elif hasattr(self.model, "encoder"):
                out = self.model.encoder(inputs)
                feats = out.cls if hasattr(out, "cls") else out

            else:
                raise RuntimeError("Model does not support feature extraction")

        return feats.cpu().numpy()


# ============================================================================
#                          DATA SECTION
# ============================================================================

class ImageDataset(Dataset):
    """Simple dataset for loading images"""
    
    def __init__(self, image_dir, image_list, labels=None, resolution=224):
        """
        Args:
            image_dir: Directory containing images
            image_list: List of image filenames
            labels: List of labels (optional, for train/val)
            resolution: Image resolution (96 for competition, 224 for DINO baseline)
        """
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
        self.resolution = resolution
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / img_name
        
        # Load and resize image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        
        if self.labels is not None:
            return image, self.labels[idx], img_name
        return image, img_name


def collate_fn(batch):
    """Custom collate function to handle PIL images"""
    if len(batch[0]) == 3:  # train/val (image, label, filename)
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        filenames = [item[2] for item in batch]
        return images, labels, filenames
    else:  # test (image, filename)
        images = [item[0] for item in batch]
        filenames = [item[1] for item in batch]
        return images, filenames


# ============================================================================
#                          FEATURE EXTRACTION
# ============================================================================

def extract_features_from_dataloader(feature_extractor, dataloader, split_name='train'):
    """
    Extract features from a dataloader.
    
    Args:
        feature_extractor: FeatureExtractor instance
        dataloader: DataLoader
        split_name: Name of split (for progress bar)
    
    Returns:
        features: numpy array (N, feature_dim)
        labels: list of labels (or None for test)
        filenames: list of filenames
    """
    all_features = []
    all_labels = []
    all_filenames = []
    
    print(f"\nExtracting features from {split_name} set...")
    
    for batch in tqdm(dataloader, desc=f"{split_name} features"):
        if len(batch) == 3:  # train/val
            images, labels, filenames = batch
            all_labels.extend(labels)
        else:  # test
            images, filenames = batch
        
        # Extract features for batch
        features = feature_extractor.extract_batch_features(images)
        all_features.append(features)
        all_filenames.extend(filenames)
    
    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels) if all_labels else None
    
    print(f"  Extracted {features.shape[0]} features of dimension {features.shape[1]}")
    
    return features, labels, all_filenames


# ============================================================================
#                          Linear CLASSIFIER
# ============================================================================

class LinearClassifier(nn.Module):
    """PyTorch linear classifier for linear probing"""
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        print(f"Creating Linear Classifier: feature_dim={feature_dim}, num_classes={num_classes}")
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)


def train_linear_classifier(
    train_features, 
    train_labels, 
    val_features, 
    val_labels, 
    num_classes,
    epochs=100,
    batch_size=256,
    learning_rate=0.1,
    weight_decay=0.0,
    use_scaler=True,
    optimizer='sgd',
    lr_scheduler='cosine',
    device='cuda'
):
    """
    Train linear classifier on features using PyTorch (similar to solo-learn).
    
    Args:
        train_features: Training features (N_train, feature_dim)
        train_labels: Training labels (N_train,)
        val_features: Validation features (N_val, feature_dim)
        val_labels: Validation labels (N_val,)
        num_classes: Number of classes
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        use_scaler: Whether to standardize features
        optimizer: Optimizer type ('sgd' or 'adam')
        lr_scheduler: Learning rate scheduler ('cosine' or 'step' or None)
        device: Device to use ('cuda', 'mps', or 'cpu')
    
    Returns:
        model: Trained PyTorch linear classifier
        scaler: Feature scaler (if used)
    """
    print(f"\nTraining Linear Classifier (PyTorch-based, similar to solo-learn)...")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"  Optimizer: {optimizer}, Weight decay: {weight_decay}")
    
    # Move to device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    device = torch.device(device)
    print(f"  Device: {device}")
    
    # Optionally standardize features
    scaler = None
    if use_scaler:
        print("  Standardizing features...")
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
    
    # Convert to tensors
    train_features_tensor = torch.FloatTensor(train_features).to(device)
    train_labels_tensor = torch.LongTensor(train_labels).to(device)
    val_features_tensor = torch.FloatTensor(val_features).to(device)
    val_labels_tensor = torch.LongTensor(val_labels).to(device)
    
    # Create model
    feature_dim = train_features.shape[1]
    model = LinearClassifier(feature_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    if optimizer.lower() == 'sgd':
        optimizer_obj = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    elif optimizer.lower() == 'adam':
        optimizer_obj = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer.lower() == 'adamw':
        optimizer_obj = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Setup learning rate scheduler
    scheduler = None
    if lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer_obj, T_max=epochs, eta_min=1e-6)
    elif lr_scheduler == 'step':
        scheduler = StepLR(optimizer_obj, step_size=30, gamma=0.1)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_features_tensor, train_labels_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # No multiprocessing for tensors
    )
    
    val_dataset = torch.utils.data.TensorDataset(val_features_tensor, val_labels_tensor)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    
    print("\n  Training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            optimizer_obj.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_obj.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer_obj.param_groups[0]['lr']
        else:
            current_lr = learning_rate
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Print progress every 10 epochs or on last epoch
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch [{epoch+1}/{epochs}] - "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_outputs = model(train_features_tensor)
        _, train_predicted = torch.max(train_outputs.data, 1)
        train_acc = (train_predicted == train_labels_tensor).float().mean().item()
        
        val_outputs = model(val_features_tensor)
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_acc = (val_predicted == val_labels_tensor).float().mean().item()
    
    print(f"\nLinear Probing Results:")
    print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return model, scaler


# ============================================================================
#                          SUBMISSION CREATION
# ============================================================================

def create_submission(test_features, test_filenames, classifier, output_path, 
                      num_classes, device='cuda', scaler=None, 
                      classifier_type='knn'):
    """
    Create submission.csv for Kaggle.
    
    Args:
        test_features: Test features (N_test, feature_dim)
        test_filenames: List of test image filenames
        classifier: Trained KNN classifier
        output_path: Path to save submission.csv
    """
    print("\nGenerating predictions on test set...")
    if classifier_type == 'linear':
        # Handle PyTorch linear classifier
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        elif device == 'mps' and not torch.backends.mps.is_available():
            device = 'cpu'
        device = torch.device(device)
        
        # Apply scaler if used
        if scaler is not None:
            test_features = scaler.transform(test_features)
        
        # Convert to tensor and predict
        test_features_tensor = torch.FloatTensor(test_features).to(device)
        classifier.eval()
        with torch.no_grad():
            outputs = classifier(test_features_tensor)
            _, predictions = torch.max(outputs.data, 1)
            predictions = predictions.cpu().numpy()
    else:
        # Handle KNN classifier
        predictions = classifier.predict(test_features)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_filenames,
        'class_id': predictions
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Submission file created: {output_path}")
    print(f"{'='*60}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10))
    print(f"\nClass distribution in predictions:")
    print(submission_df['class_id'].value_counts().head(10))
    
    # Validate submission format
    print(f"\nValidating submission format...")
    assert list(submission_df.columns) == ['id', 'class_id'], "Invalid columns!"
    assert submission_df['class_id'].min() >= 0, "Invalid class_id < 0"
    assert submission_df.isnull().sum().sum() == 0, "Missing values found!"
    print("✓ Submission format is valid!")


# ============================================================================
#                          MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Create Kaggle Submission with KNN')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing train/val/test folders')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output path for submission file')
    parser.add_argument('--model_name', type=str, 
                        default='facebook/webssl-dino300m-full2b-224',
                        help='HuggingFace model name (baseline only!)')
    parser.add_argument('--resolution', type=int, default=96,
                        help='Image resolution (96 for competition, 224 for DINO)')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of neighbors for KNN')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--classifier', type=str, default='linear',
                        choices=['knn', 'linear'],
                        help='Classifier type (knn or linear)')
    parser.add_argument('--model_class', type=str, default='auto',
                        choices=['auto', 'base', 'pretraining'],
                        help='Model class: auto (try pretraining first), base, or pretraining')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (for linear classifier)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Initial learning rate (for linear classifier)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (for linear classifier)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer type (for linear classifier)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler (for linear classifier)')
    parser.add_argument('--no_scaler', action='store_true',
                        help='Disable feature standardization (for linear classifier)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA requested but not available. Using CPU.")
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
        print("MPS requested but not available. Using CPU.")
    else:
        device = args.device
    print(f"Using device: {device}")
    
    data_dir = Path(args.data_dir)
    
    # Load CSV files
    print("\nLoading dataset metadata...")
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_df = pd.read_csv(data_dir / 'test_images.csv')
    
    print(f"  Train: {len(train_df)} images")
    print(f"  Val: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    print(f"  Classes: {train_df['class_id'].nunique()}")
    
    # Create datasets
    print(f"\nCreating datasets (resolution={args.resolution}px)...")
    train_dataset = ImageDataset(
        data_dir / 'train',
        train_df['filename'].tolist(),
        train_df['class_id'].tolist(),
        resolution=args.resolution
    )
    
    val_dataset = ImageDataset(
        data_dir / 'val',
        val_df['filename'].tolist(),
        val_df['class_id'].tolist(),
        resolution=args.resolution
    )
    
    test_dataset = ImageDataset(
        data_dir / 'test',
        test_df['filename'].tolist(),
        labels=None,
        resolution=args.resolution
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(
        model_name=args.model_name, 
        device=device,
        model_class=args.model_class
    )
    
    # Extract features
    train_features, train_labels, _ = extract_features_from_dataloader(
        feature_extractor, train_loader, 'train'
    )
    val_features, val_labels, _ = extract_features_from_dataloader(
        feature_extractor, val_loader, 'val'
    )
    test_features, _, test_filenames = extract_features_from_dataloader(
        feature_extractor, test_loader, 'test'
    )

    # Print feature statistics for debugging
    print(f"\n{'='*60}")
    print("Feature Statistics (for debugging):")
    print(f"{'='*60}")
    print(f"  Train: shape={train_features.shape}, mean={train_features.mean():.4f}, std={train_features.std():.4f}")
    print(f"         min={train_features.min():.4f}, max={train_features.max():.4f}")
    print(f"  Val:   shape={val_features.shape}, mean={val_features.mean():.4f}, std={val_features.std():.4f}")
    print(f"         min={val_features.min():.4f}, max={val_features.max():.4f}")
    print(f"  Test:  shape={test_features.shape}, mean={test_features.mean():.4f}, std={test_features.std():.4f}")
    print(f"         min={test_features.min():.4f}, max={test_features.max():.4f}")
    
    # Check for potential issues
    if abs(train_features.mean()) > 10 or train_features.std() < 0.01:
        print(f"\n⚠️  WARNING: Unusual feature statistics detected!")
        print(f"   This might indicate the model isn't loading correctly or features are broken.")
    
    # Train classifier based on type
    if args.classifier == 'knn':
        print(f"\n{'='*60}")
        print("Training KNN Classifier")
        print(f"{'='*60}")
        classifier = train_knn_classifier(
            train_features, train_labels,
            val_features, val_labels,
            k=train_df['class_id'].nunique()
        )
        scaler = None
        
    elif args.classifier == 'linear':
        print(f"\n{'='*60}")
        print("Training Linear Classifier")
        print(f"{'='*60}")
        classifier, scaler = train_linear_classifier(
            train_features, train_labels,
            val_features, val_labels,
            num_classes=train_df['class_id'].nunique(),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_scaler=not args.no_scaler,
            optimizer=args.optimizer,
            lr_scheduler=None if args.lr_scheduler == 'none' else args.lr_scheduler,
            device=device
        )
    
    # Create submission
    create_submission(
        test_features, 
        test_filenames, 
        classifier, 
        args.output,
        num_classes=train_df['class_id'].nunique(),
        device=device,
        scaler=scaler,
        classifier_type=args.classifier
    )
    
    print("\n" + "="*60)
    print("DONE! Now upload your submission.csv to Kaggle.")
    print("="*60)
    print(f"\nUsed classifier: {args.classifier.upper()}")
    if args.classifier == 'knn':
        print(f"KNN parameters: k={args.k}")
    else:
        print(f"Linear classifier trained for {args.epochs} epochs")


if __name__ == "__main__":
    main()

