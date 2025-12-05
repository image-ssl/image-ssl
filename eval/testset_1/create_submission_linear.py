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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import argparse
from torchvision import transforms

from src.models import VisionTransformer, VisionTransformerWithPretrainingHeads


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
        Extract features from a batch of PIL Images.
        
        Args:
            images: List of PIL Images
        
        Returns:
            features: numpy array of shape (batch_size, feature_dim)
        """
        # Process batch
        normalize = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                # Using ImageNet stats (same as training)
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        inputs = [normalize(img) for img in images]
        inputs = torch.stack(inputs).to(self.device).to(torch.float32)
        
        with torch.no_grad():
            # If using pretraining model, extract from encoder (not the DINO head!)
            if self.model_class == 'pretraining' and hasattr(self.model, 'encoder'):
                outputs = self.model.encoder(inputs)
            else:
                outputs = self.model(inputs)
            
            # Handle different output formats
            if hasattr(outputs, 'cls'):
                features = outputs.cls
            elif isinstance(outputs, torch.Tensor):
                # If output is just a tensor, assume it's the CLS token
                features = outputs
            else:
                raise ValueError(f"Unexpected output type: {type(outputs)}")
        
        return features.cpu().numpy()


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

def train_linear_classifier(
    train_features, 
    train_labels, 
    val_features, 
    val_labels, 
    max_iter=1000,
    C=1.0,
    use_scaler=True
):
    """
    Train linear classifier on features.
    
    Args:
        train_features: Training features (N_train, feature_dim)
        train_labels: Training labels (N_train,)
        val_features: Validation features (N_val, feature_dim)
        val_labels: Validation labels (N_val,)
        max_iter: Maximum iterations for logistic regression
        C: Inverse of regularization strength (smaller = stronger regularization)
        use_scaler: Whether to standardize features
    
    Returns:
        classifier: Trained linear classifier
        scaler: Feature scaler (if used)
    """
    print(f"\nTraining Linear Classifier (max_iter={max_iter}, C={C})...")
    
    # Optionally standardize features
    scaler = None
    if use_scaler:
        print("  Standardizing features...")
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
    
    # Train logistic regression (multi-class)
    classifier = LogisticRegression(
        max_iter=max_iter,
        C=C,
        multi_class='multinomial',  # For multi-class classification
        solver='lbfgs',  # Good for small-medium datasets
        random_state=42,
        n_jobs=-1
    )
    
    print("  Fitting classifier...")
    classifier.fit(train_features, train_labels)
    
    # Evaluate
    train_acc = classifier.score(train_features, train_labels)
    val_acc = classifier.score(val_features, val_labels)
    
    print(f"\nLinear Probing Results:")
    print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return classifier, scaler

def create_submission(test_features, test_filenames, classifier, scaler, output_path, num_classes=200):
    """
    Create submission.csv for Kaggle.
    
    Args:
        test_features: Test features (N_test, feature_dim)
        test_filenames: List of test image filenames
        classifier: Trained linear classifier
        scaler: Feature scaler (if used, None otherwise)
        output_path: Path to save submission.csv
        num_classes: Number of classes (for validation)
    """
    print("\nGenerating predictions on test set...")
    
    # Apply scaler if used
    if scaler is not None:
        test_features = scaler.transform(test_features)
    
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
    assert submission_df['class_id'].min() >= 0, f"Invalid class_id < 0"
    assert submission_df['class_id'].max() < num_classes, f"Invalid class_id >= {num_classes}"
    assert submission_df.isnull().sum().sum() == 0, "Missing values found!"
    print("✓ Submission format is valid!")

def main():
    parser = argparse.ArgumentParser(description='Create Kaggle Submission with Linear Probing')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing train/val/test folders')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output path for submission file')
    parser.add_argument('--model_name', type=str, required=True,
                        help='HuggingFace model repo ID or local path to your trained model checkpoint')
    parser.add_argument('--model_class', type=str, default='auto',
                        choices=['auto', 'base', 'pretraining'],
                        help='Model class: auto (try pretraining first), base, or pretraining')
    parser.add_argument('--resolution', type=int, default=96,
                        help='Image resolution (96 for competition, 224 for DINO)')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Maximum iterations for logistic regression')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Inverse of regularization strength (smaller = stronger regularization)')
    parser.add_argument('--no_scaler', action='store_true',
                        help='Disable feature standardization')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda, mps, or cpu)')
    parser.add_argument('--num_classes', type=int, default=200,
                        help='Number of classes in the dataset')
    
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
    
    # Train linear classifier
    classifier, scaler = train_linear_classifier(
        train_features, train_labels,
        val_features, val_labels,
        max_iter=args.max_iter,
        C=args.C,
        use_scaler=not args.no_scaler
    )
    
    # Create submission
    create_submission(
        test_features, 
        test_filenames, 
        classifier, 
        scaler,
        args.output,
        num_classes=args.num_classes
    )
    
    print("\n" + "="*60)
    print("DONE! Now upload your submission.csv to Kaggle.")
    print("="*60)
    print("\nREMINDER: This baseline uses pretrained weights!")
    print("For the competition, you MUST train your own SSL model from scratch.")


if __name__ == "__main__":
    main()

