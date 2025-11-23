"""
test_model.py - Script d'inférence pour tester le modèle WikiArt

Usage:
    python test_model.py                           # Teste toutes les images dans test_images/
    python test_model.py --image path/to/image.jpg # Teste une image spécifique
    python test_model.py --checkpoint best         # Utilise le meilleur modèle
    python test_model.py --top-k 5                 # Affiche les top 5 prédictions

Place tes images de test dans le dossier test_images/
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from data import download_or_use_cached, discover_structure, build_label_mappings
from models.classifier import WikiArtClassifier


# ============================================================================
# CONFIGURATION
# ============================================================================

TEST_IMAGES_DIR = Path("test_images")
CHECKPOINTS_DIR = Path("checkpoints")

# Transforms pour l'inférence (identique à val/test)
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(
        int(224 * 1.14),
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def list_checkpoints() -> list[Path]:
    """Liste tous les checkpoints disponibles (y compris dans les sous-dossiers)."""
    if not CHECKPOINTS_DIR.exists():
        return []

    # Chercher les .pt dans checkpoints/ et ses sous-dossiers
    checkpoints = list(CHECKPOINTS_DIR.glob("**/*.pt"))
    return sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)


def list_test_images() -> list[Path]:
    """Liste toutes les images dans le dossier test_images/."""
    if not TEST_IMAGES_DIR.exists():
        return []

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    images = [
        f for f in TEST_IMAGES_DIR.iterdir()
        if f.suffix.lower() in extensions
    ]
    return sorted(images)


def load_model(
    checkpoint_path: Path,
    num_styles: int,
    num_artists: int,
    device: torch.device,
) -> tuple[WikiArtClassifier, dict]:
    """
    Charge un modèle depuis un checkpoint.

    Returns:
        Tuple (modèle, infos du checkpoint)
    """
    print(f"\nChargement du checkpoint: {checkpoint_path.name}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Récupérer la config du modèle
    config = checkpoint.get("config", {})
    backbone = config.get("backbone", "vit_base_patch16_224")
    dropout = config.get("dropout", 0.1)

    # Créer le modèle
    model = WikiArtClassifier(
        num_styles=num_styles,
        num_artists=num_artists,
        backbone_name=backbone,
        pretrained=False,  # On charge les poids du checkpoint
        freeze_backbone=False,
        dropout=dropout,
    )

    # Charger les poids
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Infos sur le checkpoint
    info = {
        "epoch": checkpoint.get("epoch", "?"),
        "phase": checkpoint.get("phase", "?"),
        "backbone": backbone,
        "val_metrics": checkpoint.get("val_metrics", {}),
    }

    return model, info


def predict_image(
    model: WikiArtClassifier,
    image_path: Path,
    idx2style: dict,
    idx2artist: dict,
    device: torch.device,
    top_k: int = 5,
) -> dict:
    """
    Fait une prédiction sur une image.

    Returns:
        Dict avec les prédictions style et artiste
    """
    # Charger et transformer l'image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Impossible de charger l'image: {e}"}

    # Appliquer les transforms
    input_tensor = INFERENCE_TRANSFORMS(image).unsqueeze(0).to(device)

    # Inférence
    with torch.no_grad():
        outputs = model(input_tensor)

    style_logits = outputs["style_logits"]
    artist_logits = outputs["artist_logits"]

    # Softmax pour obtenir les probabilités
    style_probs = torch.softmax(style_logits, dim=-1).squeeze()
    artist_probs = torch.softmax(artist_logits, dim=-1).squeeze()

    # Top-K styles
    style_topk = torch.topk(style_probs, min(top_k, len(style_probs)))
    style_predictions = [
        {
            "rank": i + 1,
            "style": idx2style[idx.item()],
            "confidence": prob.item() * 100,
        }
        for i, (prob, idx) in enumerate(zip(style_topk.values, style_topk.indices))
    ]

    # Top-K artistes
    artist_topk = torch.topk(artist_probs, min(top_k, len(artist_probs)))
    artist_predictions = [
        {
            "rank": i + 1,
            "artist": idx2artist[idx.item()],
            "confidence": prob.item() * 100,
        }
        for i, (prob, idx) in enumerate(zip(artist_topk.values, artist_topk.indices))
    ]

    return {
        "image": image_path.name,
        "image_size": image.size,
        "style_predictions": style_predictions,
        "artist_predictions": artist_predictions,
    }


def print_prediction(result: dict, show_all: bool = False):
    """Affiche les résultats de prédiction de manière lisible."""
    if "error" in result:
        print(f"  ❌ Erreur: {result['error']}")
        return

    print(f"\n{'='*60}")
    print(f"📷 Image: {result['image']}")
    print(f"   Dimensions: {result['image_size'][0]}x{result['image_size'][1]}")
    print(f"{'='*60}")

    # Style
    print("\n🎨 STYLE:")
    print("-" * 40)
    for pred in result["style_predictions"]:
        bar = "█" * int(pred["confidence"] / 5)
        print(f"  {pred['rank']}. {pred['style']:<25} {pred['confidence']:>5.1f}% {bar}")

    # Artiste
    print("\n👤 ARTISTE:")
    print("-" * 40)
    for pred in result["artist_predictions"]:
        bar = "█" * int(pred["confidence"] / 5)
        print(f"  {pred['rank']}. {pred['artist']:<25} {pred['confidence']:>5.1f}% {bar}")


def print_model_info(checkpoint_info: dict):
    """Affiche les informations sur le modèle."""
    print("\n" + "=" * 60)
    print("📊 INFORMATIONS DU MODÈLE")
    print("=" * 60)
    print(f"  Backbone:     {checkpoint_info['backbone']}")
    print(f"  Epoch:        {checkpoint_info['epoch']}")
    print(f"  Phase:        {checkpoint_info['phase']}")

    if checkpoint_info["val_metrics"]:
        metrics = checkpoint_info["val_metrics"]
        print(f"\n  Métriques de validation:")
        if "style_top1" in metrics:
            print(f"    Style Top-1:  {metrics['style_top1']*100:.1f}%")
        if "style_top5" in metrics:
            print(f"    Style Top-5:  {metrics['style_top5']*100:.1f}%")
        if "artist_top1" in metrics:
            print(f"    Artist Top-1: {metrics['artist_top1']*100:.1f}%")
        if "artist_top5" in metrics:
            print(f"    Artist Top-5: {metrics['artist_top5']*100:.1f}%")


def select_checkpoint_interactive() -> Optional[Path]:
    """Permet à l'utilisateur de choisir un checkpoint interactivement."""
    checkpoints = list_checkpoints()

    if not checkpoints:
        print("❌ Aucun checkpoint trouvé dans checkpoints/")
        return None

    print("\n📁 Checkpoints disponibles:")
    print("-" * 50)
    for i, cp in enumerate(checkpoints):
        # Charger les infos basiques
        try:
            info = torch.load(cp, map_location="cpu", weights_only=False)
            epoch = info.get("epoch", "?")
            phase = info.get("phase", "?")
            val_acc = info.get("val_metrics", {}).get("style_top1", 0)
            print(f"  [{i+1}] {cp.name:<35} (Phase {phase}, Epoch {epoch}, Val: {val_acc*100:.1f}%)")
        except Exception:
            print(f"  [{i+1}] {cp.name}")

    print()
    choice = input("Choisis un checkpoint (numéro ou 'best' pour le meilleur): ").strip()

    if choice.lower() == "best":
        # Chercher best_model.pt dans les sous-dossiers
        best_paths = list(CHECKPOINTS_DIR.glob("**/best_model.pt"))
        if best_paths:
            # Prendre le plus récent
            return max(best_paths, key=lambda x: x.stat().st_mtime)
        print("⚠️  best_model.pt non trouvé, utilisation du plus récent")
        return checkpoints[0]

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(checkpoints):
            return checkpoints[idx]
    except ValueError:
        pass

    print("❌ Choix invalide")
    return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test du modèle WikiArt sur des images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python test_model.py                           # Mode interactif
  python test_model.py --image mon_image.jpg     # Tester une image spécifique
  python test_model.py --checkpoint best --top-k 3
        """
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Chemin vers une image spécifique à tester"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Checkpoint à utiliser ('best' ou chemin)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Nombre de prédictions à afficher (défaut: 5)"
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="Liste les checkpoints disponibles et quitte"
    )
    parser.add_argument(
        "--list-images",
        action="store_true",
        help="Liste les images dans test_images/ et quitte"
    )

    args = parser.parse_args()

    # Mode liste checkpoints
    if args.list_checkpoints:
        checkpoints = list_checkpoints()
        if checkpoints:
            print("Checkpoints disponibles:")
            for cp in checkpoints:
                print(f"  - {cp.name}")
        else:
            print("Aucun checkpoint trouvé")
        return

    # Mode liste images
    if args.list_images:
        images = list_test_images()
        if images:
            print("Images dans test_images/:")
            for img in images:
                print(f"  - {img.name}")
        else:
            print("Aucune image trouvée dans test_images/")
        return

    # Créer le dossier test_images si nécessaire
    TEST_IMAGES_DIR.mkdir(exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Charger les mappings style/artiste
    print("\n📚 Chargement des métadonnées du dataset...")
    dataset_path = download_or_use_cached()
    structure = discover_structure(dataset_path)
    style2idx, idx2style, artist2idx, idx2artist = build_label_mappings(structure)

    num_styles = len(style2idx)
    num_artists = len(artist2idx)
    print(f"   {num_styles} styles, {num_artists} artistes")

    # Sélection du checkpoint
    if args.checkpoint:
        if args.checkpoint.lower() == "best":
            # Chercher best_model.pt dans les sous-dossiers
            best_paths = list(CHECKPOINTS_DIR.glob("**/best_model.pt"))
            if best_paths:
                checkpoint_path = max(best_paths, key=lambda x: x.stat().st_mtime)
            else:
                print("❌ best_model.pt non trouvé")
                return
        else:
            checkpoint_path = Path(args.checkpoint)

        if not checkpoint_path.exists():
            print(f"❌ Checkpoint non trouvé: {checkpoint_path}")
            return
    else:
        checkpoint_path = select_checkpoint_interactive()
        if checkpoint_path is None:
            return

    # Charger le modèle
    model, checkpoint_info = load_model(
        checkpoint_path, num_styles, num_artists, device
    )
    print_model_info(checkpoint_info)

    # Déterminer les images à tester
    if args.image:
        image_paths = [Path(args.image)]
        if not image_paths[0].exists():
            print(f"❌ Image non trouvée: {args.image}")
            return
    else:
        image_paths = list_test_images()
        if not image_paths:
            print(f"\n⚠️  Aucune image trouvée dans {TEST_IMAGES_DIR}/")
            print(f"   Place des images (.jpg, .png, etc.) dans ce dossier pour les tester.")
            return

    # Tester chaque image
    print(f"\n🔍 Test de {len(image_paths)} image(s)...")

    for image_path in image_paths:
        result = predict_image(
            model, image_path, idx2style, idx2artist, device, top_k=args.top_k
        )
        print_prediction(result)

    print("\n" + "=" * 60)
    print("✅ Test terminé!")
    print("=" * 60)


if __name__ == "__main__":
    main()
