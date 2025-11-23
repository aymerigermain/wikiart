"""
dataset.py - PyTorch Dataset for WikiArt

Provides WikiArtDataset class with:
- Multi-task labels (style + artist)
- Stratified train/val/test splits
- Configurable transforms for training and evaluation
- Support for caching and efficient loading
"""

from pathlib import Path
from typing import Optional, Callable, Literal

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

from data import discover_structure, build_label_mappings, download_or_use_cached, extract_artist_name


class WikiArtDataset(Dataset):
    """
    PyTorch Dataset for WikiArt images with style and artist labels.

    Args:
        image_paths: List of paths to images.
        style_labels: List of style indices.
        artist_labels: List of artist indices.
        transform: Optional transform to apply to images.
    """

    def __init__(
        self,
        image_paths: list[Path],
        style_labels: list[int],
        artist_labels: list[int],
        transform: Optional[Callable] = None,
    ):
        assert len(image_paths) == len(style_labels) == len(artist_labels)

        self.image_paths = image_paths
        self.style_labels = style_labels
        self.artist_labels = artist_labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            Dictionary with:
                - image: Transformed image tensor
                - style: Style class index
                - artist: Artist class index
                - path: Original image path (for debugging)
        """
        image_path = self.image_paths[idx]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Handle corrupted images by returning a black image
            print(f"Warning: Could not load {image_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "style": torch.tensor(self.style_labels[idx], dtype=torch.long),
            "artist": torch.tensor(self.artist_labels[idx], dtype=torch.long),
            "path": str(image_path),
        }


def get_transforms(
    mode: Literal["train", "val", "test"],
    image_size: int = 224,
    augmentation_strength: Literal["light", "medium", "strong"] = "strong",
) -> transforms.Compose:
    """
    Get transforms for different modes.

    Args:
        mode: One of "train", "val", "test".
        image_size: Target image size (default 224 for ViT).
        augmentation_strength: Intensity of augmentation for training.

    Returns:
        Composed transforms.
    """
    # ImageNet normalization (standard for pretrained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if mode == "train":
        # Configuration selon la force d'augmentation
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
        else:  # strong - recommandé pour combattre l'overfitting
            scale = (0.6, 1.0)
            color_jitter = (0.3, 0.3, 0.3, 0.1)
            flip_p = 0.5
            rotation = 15
            erasing_p = 0.2

        transform_list = [
            # Crop aléatoire plus agressif
            transforms.RandomResizedCrop(
                image_size,
                scale=scale,
                ratio=(0.75, 1.33),  # Permet des ratios variés
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            # Rotation légère (préserve la composition artistique)
            transforms.RandomRotation(
                degrees=rotation,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            # Flip horizontal (attention aux portraits, mais aide à généraliser)
            transforms.RandomHorizontalFlip(p=flip_p),
            # Variations de couleur plus fortes
            transforms.ColorJitter(
                brightness=color_jitter[0],
                contrast=color_jitter[1],
                saturation=color_jitter[2],
                hue=color_jitter[3],
            ),
            # Parfois convertir en niveaux de gris (force le modèle à regarder les formes)
            transforms.RandomGrayscale(p=0.05),
            # Gaussian blur léger (simule différentes qualités de scan)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.1),
            transforms.ToTensor(),
            normalize,
        ]

        # Random Erasing après ToTensor (simule occlusions, force features robustes)
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
    artist2idx: dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Create stratified train/val/test splits.

    Stratification is done on style to ensure all styles are represented
    in each split. Artist stratification is not feasible due to many
    artists having very few samples.

    Args:
        structure: Dataset structure from discover_structure().
        style2idx: Style to index mapping.
        artist2idx: Artist to index mapping.
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.
        test_ratio: Proportion for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_data, val_data, test_data) where each is a list
        of (image_path, style_idx, artist_idx) tuples.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Collect all samples with their labels
    all_samples = []
    for style, images in structure['style_to_images'].items():
        style_idx = style2idx[style]
        for image_path in images:
            artist_name = extract_artist_name(image_path.stem)
            artist_idx = artist2idx.get(artist_name, 0)  # Default to 0 if not found
            all_samples.append((image_path, style_idx, artist_idx))

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
    num_styles: int = 27,
    persistent_workers: bool = True,
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
        num_styles: Number of style classes (for weighted sampler).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Unpack samples
    train_paths, train_styles, train_artists = zip(*train_samples)
    val_paths, val_styles, val_artists = zip(*val_samples)
    test_paths, test_styles, test_artists = zip(*test_samples)

    # Create datasets avec augmentation forte pour combattre l'overfitting
    train_dataset = WikiArtDataset(
        list(train_paths),
        list(train_styles),
        list(train_artists),
        transform=get_transforms("train", augmentation_strength="strong"),
    )
    val_dataset = WikiArtDataset(
        list(val_paths),
        list(val_styles),
        list(val_artists),
        transform=get_transforms("val"),
    )
    test_dataset = WikiArtDataset(
        list(test_paths),
        list(test_styles),
        list(test_artists),
        transform=get_transforms("test"),
    )

    # Create weighted sampler for training (handles class imbalance)
    train_sampler = None
    shuffle_train = True

    if use_weighted_sampler:
        # Compute class weights based on style distribution
        style_counts = np.bincount(train_styles, minlength=num_styles)
        style_weights = 1.0 / (style_counts + 1e-6)  # Avoid division by zero
        sample_weights = [style_weights[s] for s in train_styles]

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_samples),
            replacement=True,
        )
        shuffle_train = False  # Sampler handles shuffling

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        persistent_workers=persistent_workers, # Maintient les workers actifs entre les epochs
        pin_memory=True,
        drop_last=True,  # For stable batch norm
        prefetch_factor=4, # Permet de précharger plusieurs batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        prefetch_factor=4, # Permet de précharger plusieurs batches
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        prefetch_factor=4, # Permet de précharger plusieurs batches
    )

    return train_loader, val_loader, test_loader


def get_class_weights(
    samples: list,
    num_classes: int,
    label_idx: int = 1,  # 1 for style, 2 for artist
) -> torch.Tensor:
    """
    Compute class weights for loss function (inverse frequency).

    Args:
        samples: List of (path, style_idx, artist_idx) tuples.
        num_classes: Number of classes.
        label_idx: Index in tuple for the label (1=style, 2=artist).

    Returns:
        Tensor of class weights.
    """
    labels = [s[label_idx] for s in samples]
    counts = np.bincount(labels, minlength=num_classes)

    # Inverse frequency weighting with smoothing
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * num_classes  # Normalize

    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Test the dataset pipeline
    print("Downloading and discovering dataset...")
    dataset_path = download_or_use_cached()
    structure = discover_structure(dataset_path)

    print(f"\nDataset contains {structure['total_images']} images")
    print(f"Styles: {len(structure['styles'])}")
    print(f"Artists: {len(structure['artists'])}")

    # Build mappings
    style2idx, idx2style, artist2idx, idx2artist = build_label_mappings(structure)

    # Create splits
    print("\nCreating train/val/test splits...")
    train_samples, val_samples, test_samples = create_splits(
        structure, style2idx, artist2idx
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
        num_workers=0,  # 0 for testing
        num_styles=len(structure['styles']),
    )

    # Test a batch
    print("\nTesting a batch...")
    batch = next(iter(train_loader))
    print(f"Image shape: {batch['image'].shape}")
    print(f"Style labels shape: {batch['style'].shape}")
    print(f"Artist labels shape: {batch['artist'].shape}")
    print(f"Sample styles: {batch['style'][:5].tolist()}")
    print(f"Sample artists: {batch['artist'][:5].tolist()}")

    # Show class weights
    print("\nClass weights (style):")
    style_weights = get_class_weights(train_samples, len(structure['styles']), label_idx=1)
    print(f"Min: {style_weights.min():.4f}, Max: {style_weights.max():.4f}")

    print("\nDataset pipeline ready!")
