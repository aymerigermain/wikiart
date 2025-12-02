"""
dataset_style.py - PyTorch Dataset for WikiArt Style Classification

Simplified dataset focused exclusively on style labels.
Based on the original dataset.py but streamlined for single-task learning.
"""

from pathlib import Path
from typing import Optional, Callable, Literal

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

from data import discover_structure, build_label_mappings, download_or_use_cached


class WikiArtStyleDataset(Dataset):
    """
    PyTorch Dataset for WikiArt images with style labels only.

    Args:
        image_paths: List of paths to images.
        style_labels: List of style indices.
        transform: Optional transform to apply to images.
    """

    def __init__(
        self,
        image_paths: list[Path],
        style_labels: list[int],
        transform: Optional[Callable] = None,
    ):
        assert len(image_paths) == len(style_labels)

        self.image_paths = image_paths
        self.style_labels = style_labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            Dictionary with:
                - image: Transformed image tensor
                - label: Style class index
                - path: Original image path (for debugging)
        """
        image_path = self.image_paths[idx]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load {image_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(self.style_labels[idx], dtype=torch.long),
            "path": str(image_path),
        }


def get_transforms(
    mode: Literal["train", "val", "test"],
    image_size: int = 224,
    augmentation_strength: Literal["light", "medium", "strong"] = "strong",
    backbone_type: Literal["timm", "clip"] = "timm",
) -> transforms.Compose:
    """
    Get transforms for different modes.

    Args:
        mode: One of "train", "val", "test".
        image_size: Target image size (default 224 for ViT).
        augmentation_strength: Intensity of augmentation for training.
        backbone_type: "timm" (ImageNet norm) ou "clip" (CLIP norm).

    Returns:
        Composed transforms.
    """
    # Normalisation selon le backbone
    if backbone_type == "clip":
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    if mode == "train":
        if augmentation_strength == "light":
            scale = (0.8, 1.0)
            color_jitter = (0.1, 0.1, 0.1, 0.02)
            flip_p = 0.3
            rotation = 0
            erasing_p = 0.0
        elif augmentation_strength == "medium":
            scale = (0.7, 1.0)
            color_jitter = (0.2, 0.2, 0.2, 0.05)
            flip_p = 0.4
            rotation = 10
            erasing_p = 0.1
        else:  # strong
            scale = (0.6, 1.0)
            color_jitter = (0.3, 0.3, 0.3, 0.1)
            flip_p = 0.5
            rotation = 15
            erasing_p = 0.2

        transform_list = [
            transforms.RandomResizedCrop(
                image_size,
                scale=scale,
                ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomRotation(
                degrees=rotation,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.ColorJitter(
                brightness=color_jitter[0],
                contrast=color_jitter[1],
                saturation=color_jitter[2],
                hue=color_jitter[3],
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.1),
            transforms.ToTensor(),
            normalize,
        ]

        if erasing_p > 0:
            transform_list.append(
                transforms.RandomErasing(
                    p=erasing_p,
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3),
                    value="random",
                )
            )

        return transforms.Compose(transform_list)

    else:  # val or test
        return transforms.Compose([
            transforms.Resize(
                int(image_size * 1.14),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


def create_splits(
    structure: dict,
    style2idx: dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Create stratified train/val/test splits for style classification.

    Args:
        structure: Dataset structure from discover_structure().
        style2idx: Style to index mapping.
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.
        test_ratio: Proportion for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_data, val_data, test_data) where each is a list
        of (image_path, style_idx) tuples.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Collect all samples with their labels
    all_samples = []
    for style, images in structure['style_to_images'].items():
        style_idx = style2idx[style]
        for image_path in images:
            all_samples.append((image_path, style_idx))

    # Extract style labels for stratification
    style_labels = [s[1] for s in all_samples]

    # First split: train vs (val + test)
    train_samples, temp_samples, train_styles, temp_styles = train_test_split(
        all_samples,
        style_labels,
        train_size=train_ratio,
        stratify=style_labels,
        random_state=seed,
    )

    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples,
        train_size=val_ratio_adjusted,
        stratify=temp_styles,
        random_state=seed,
    )

    return train_samples, val_samples, test_samples


def create_dataloaders(
    train_samples: list,
    val_samples: list,
    test_samples: list,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    num_classes: int = 22,
    persistent_workers: bool = True,
    backbone_type: Literal["timm", "clip"] = "timm",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test sets.

    Args:
        train_samples: Training samples from create_splits().
        val_samples: Validation samples from create_splits().
        test_samples: Test samples from create_splits().
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        use_weighted_sampler: Use weighted sampling to handle class imbalance.
        num_classes: Number of style classes.
        persistent_workers: Keep workers alive between epochs.
        backbone_type: "timm" ou "clip" pour adapter la normalisation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Unpack samples
    train_paths, train_labels = zip(*train_samples)
    val_paths, val_labels = zip(*val_samples)
    test_paths, test_labels = zip(*test_samples)

    # Create datasets
    train_dataset = WikiArtStyleDataset(
        list(train_paths),
        list(train_labels),
        transform=get_transforms("train", augmentation_strength="strong", backbone_type=backbone_type),
    )
    val_dataset = WikiArtStyleDataset(
        list(val_paths),
        list(val_labels),
        transform=get_transforms("val", backbone_type=backbone_type),
    )
    test_dataset = WikiArtStyleDataset(
        list(test_paths),
        list(test_labels),
        transform=get_transforms("test", backbone_type=backbone_type),
    )

    # Create weighted sampler for training
    train_sampler = None
    shuffle_train = True

    if use_weighted_sampler:
        style_counts = np.bincount(train_labels, minlength=num_classes)
        style_weights = 1.0 / (style_counts + 1e-6)
        sample_weights = [style_weights[s] for s in train_labels]

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_samples),
            replacement=True,
        )
        shuffle_train = False

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        prefetch_factor=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        prefetch_factor=4,
    )

    return train_loader, val_loader, test_loader


def get_class_weights(
    samples: list,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute class weights for loss function (inverse frequency).

    Args:
        samples: List of (path, style_idx) tuples.
        num_classes: Number of classes.

    Returns:
        Tensor of class weights.
    """
    labels = [s[1] for s in samples]
    counts = np.bincount(labels, minlength=num_classes)

    # Inverse frequency weighting with smoothing
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Test the dataset pipeline
    print("Downloading and discovering dataset...")
    dataset_path = download_or_use_cached()
    structure = discover_structure(dataset_path)

    print(f"\nDataset contains {structure['total_images']} images")
    print(f"Styles: {len(structure['styles'])}")

    # Build mappings
    style2idx, idx2style, _, _ = build_label_mappings(structure)

    # Create splits
    print("\nCreating train/val/test splits...")
    train_samples, val_samples, test_samples = create_splits(
        structure, style2idx
    )
    print(f"Train: {len(train_samples)}")
    print(f"Val: {len(val_samples)}")
    print(f"Test: {len(test_samples)}")

    # Create dataloaders
    print("\nCreating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples,
        val_samples,
        test_samples,
        batch_size=32,
        num_workers=0,
        num_classes=len(structure['styles']),
    )

    # Test a batch
    print("\nTesting a batch...")
    batch = next(iter(train_loader))
    print(f"Image shape: {batch['image'].shape}")
    print(f"Label shape: {batch['label'].shape}")
    print(f"Sample labels: {batch['label'][:5].tolist()}")

    # Show class weights
    print("\nClass weights:")
    class_weights = get_class_weights(train_samples, len(structure['styles']))
    print(f"Min: {class_weights.min():.4f}, Max: {class_weights.max():.4f}")

    print("\nDataset pipeline ready!")
