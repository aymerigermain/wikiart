"""
data.py - WikiArt Dataset Download and Structure Discovery

Downloads the WikiArt dataset from Kaggle and provides utilities
to explore its structure (styles, artists, file organization).
"""

import os
from pathlib import Path
from collections import defaultdict
import json

import kagglehub


def download_dataset() -> Path:
    """
    Download WikiArt dataset from Kaggle.

    Returns:
        Path to the downloaded dataset root directory.
    """
    path = kagglehub.dataset_download("steubk/wikiart")
    return Path(path)


def discover_structure(dataset_path: Path) -> dict:
    """
    Discover the dataset structure by scanning directories.

    WikiArt is typically organized as:
        dataset_root/
        ├── style1/
        │   ├── artist1_painting1.jpg
        │   ├── artist1_painting2.jpg
        │   └── ...
        ├── style2/
        │   └── ...
        └── ...

    Args:
        dataset_path: Root path of the WikiArt dataset.

    Returns:
        Dictionary containing:
            - styles: list of style names
            - artists: list of unique artist names
            - style_to_images: mapping style -> list of image paths
            - artist_to_images: mapping artist -> list of image paths
            - style_to_artists: mapping style -> set of artists
            - total_images: total number of images
    """
    styles = []
    artists = set()
    style_to_images = defaultdict(list)
    artist_to_images = defaultdict(list)
    style_to_artists = defaultdict(set)

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}

    # Scan dataset directory
    for style_dir in sorted(dataset_path.iterdir()):
        if not style_dir.is_dir():
            continue

        style_name = style_dir.name
        styles.append(style_name)

        for image_file in style_dir.iterdir():
            if image_file.suffix.lower() not in image_extensions:
                continue

            # Extract artist name from filename
            # Format: "artistname_paintingtitle.jpg"
            filename = image_file.stem
            artist_name = extract_artist_name(filename)

            artists.add(artist_name)
            style_to_images[style_name].append(image_file)
            artist_to_images[artist_name].append(image_file)
            style_to_artists[style_name].add(artist_name)

    return {
        'styles': styles,
        'artists': sorted(artists),
        'style_to_images': dict(style_to_images),
        'artist_to_images': dict(artist_to_images),
        'style_to_artists': {k: sorted(v) for k, v in style_to_artists.items()},
        'total_images': sum(len(imgs) for imgs in style_to_images.values()),
    }


def extract_artist_name(filename: str) -> str:
    """
    Extract artist name from WikiArt filename.

    WikiArt filenames follow the pattern: "artistname_paintingtitle"
    Artist names use underscores between parts (e.g., "claude-monet" or "vincent-van-gogh")

    Args:
        filename: Image filename without extension.

    Returns:
        Extracted artist name.
    """
    # Split by underscore and take the first part as artist
    # This works for most WikiArt naming conventions
    parts = filename.split('_')
    if len(parts) >= 1:
        return parts[0]
    return filename


def build_label_mappings(structure: dict) -> tuple[dict, dict, dict, dict]:
    """
    Build label-to-index and index-to-label mappings.

    Args:
        structure: Dataset structure from discover_structure().

    Returns:
        Tuple of (style2idx, idx2style, artist2idx, idx2artist)
    """
    style2idx = {style: idx for idx, style in enumerate(structure['styles'])}
    idx2style = {idx: style for style, idx in style2idx.items()}

    artist2idx = {artist: idx for idx, artist in enumerate(structure['artists'])}
    idx2artist = {idx: artist for artist, idx in artist2idx.items()}

    return style2idx, idx2style, artist2idx, idx2artist


def save_metadata(structure: dict, output_path: Path) -> None:
    """
    Save dataset metadata to JSON for reproducibility.

    Args:
        structure: Dataset structure from discover_structure().
        output_path: Path to save the metadata JSON.
    """
    metadata = {
        'styles': structure['styles'],
        'artists': structure['artists'],
        'num_styles': len(structure['styles']),
        'num_artists': len(structure['artists']),
        'total_images': structure['total_images'],
        'images_per_style': {k: len(v) for k, v in structure['style_to_images'].items()},
    }

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {output_path}")


def print_dataset_stats(structure: dict) -> None:
    """Print dataset statistics."""
    print("=" * 50)
    print("WikiArt Dataset Statistics")
    print("=" * 50)
    print(f"Total images: {structure['total_images']:,}")
    print(f"Number of styles: {len(structure['styles'])}")
    print(f"Number of artists: {len(structure['artists'])}")
    print()
    print("Images per style:")
    print("-" * 30)
    for style in sorted(structure['styles']):
        count = len(structure['style_to_images'][style])
        print(f"  {style}: {count:,}")
    print("=" * 50)


if __name__ == "__main__":
    # Download dataset
    print("Downloading WikiArt dataset...")
    if not Path("~/.cache/kagglehub/datasets/steubk/wikiart/versions/1").expanduser().exists():
        dataset_path = download_dataset()
    else:
        print("Dataset already downloaded.")
        print("Using cached dataset.")
        dataset_path = Path("~/.cache/kagglehub/datasets/steubk/wikiart/versions/1").expanduser()
    
    print(f"Dataset path: {dataset_path}")

    # Discover structure
    print("\nDiscovering dataset structure...")
    structure = discover_structure(dataset_path)

    # Print stats
    print_dataset_stats(structure)

    # Save metadata
    metadata_path = Path(__file__).parent / "metadata.json"
    save_metadata(structure, metadata_path)