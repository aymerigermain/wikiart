"""
inference.py - Module d'inférence pour l'interface web

Gère les prédictions pour styles, artistes et descriptions
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au path pour importer les modules du projet
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torchvision import transforms
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

from data import download_or_use_cached, discover_structure, build_label_mappings
from models.classifier import WikiArtClassifier


# ============================================================================
# CONFIGURATION
# ============================================================================

# Transforms pour l'inférence (identique à test_model.py)
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
# CLASSE PRINCIPALE D'INFÉRENCE
# ============================================================================

class WikiArtInference:
    """
    Gestionnaire d'inférence pour les modèles WikiArt.
    Charge les modèles une seule fois et les garde en mémoire.
    """

    def __init__(self, checkpoint_path: str = None, load_llava: bool = True):
        """
        Initialise le système d'inférence.

        Args:
            checkpoint_path: Chemin vers le checkpoint du modèle (None = auto-détection)
            load_llava: Si True, charge aussi LLaVA pour les descriptions
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Device: {self.device}")

        # Charger les mappings style/artiste
        print("📚 Chargement des métadonnées...")
        dataset_path = download_or_use_cached()
        structure = discover_structure(dataset_path)
        self.style2idx, self.idx2style, self.artist2idx, self.idx2artist = build_label_mappings(structure)

        # Charger le modèle de classification
        print("🎨 Chargement du modèle de classification...")
        self.classifier_model = self._load_classifier(checkpoint_path)

        # Charger LLaVA si demandé
        self.llava_model = None
        self.llava_processor = None
        if load_llava:
            try:
                print("📝 Chargement de LLaVA pour les descriptions...")
                self.llava_model, self.llava_processor = self._load_llava()
            except Exception as e:
                print(f"⚠️  Impossible de charger LLaVA: {e}")
                print("   Les descriptions ne seront pas disponibles")

        print("✅ Système d'inférence prêt!")

    def _find_best_checkpoint(self) -> Path:
        """Trouve le meilleur checkpoint disponible."""
        checkpoints_dir = Path(__file__).parent.parent / "checkpoints"

        # Chercher best_model.pt
        best_paths = list(checkpoints_dir.glob("**/best_model.pt"))
        if best_paths:
            return max(best_paths, key=lambda x: x.stat().st_mtime)

        # Sinon, prendre le plus récent
        all_checkpoints = list(checkpoints_dir.glob("**/*.pt"))
        if all_checkpoints:
            return max(all_checkpoints, key=lambda x: x.stat().st_mtime)

        raise FileNotFoundError("Aucun checkpoint trouvé dans checkpoints/")

    def _load_classifier(self, checkpoint_path: str = None) -> WikiArtClassifier:
        """Charge le modèle de classification depuis un checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()
        else:
            checkpoint_path = Path(checkpoint_path)

        print(f"   Checkpoint: {checkpoint_path.name}")

        # Charger le checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Récupérer la config
        config = checkpoint.get("config", {})
        backbone = config.get("backbone", "vit_base_patch16_224")
        dropout = config.get("dropout", 0.1)
        backbone_type = config.get("backbone_type", "timm")

        # Créer le modèle
        model = WikiArtClassifier(
            num_styles=len(self.style2idx),
            num_artists=len(self.artist2idx),
            backbone_name=backbone,
            backbone_type=backbone_type,
            pretrained=False,
            freeze_backbone=False,
            dropout=dropout,
        )

        # Charger les poids
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        return model

    def _load_llava(self) -> tuple:
        """Charge LLaVA pour la génération de descriptions."""
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

        # Configuration pour quantization 4-bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )

        return model, processor

    def predict_style_artist(self, image_path: str, top_k: int = 5) -> dict:
        """
        Prédit le style et l'artiste d'une image.

        Args:
            image_path: Chemin vers l'image
            top_k: Nombre de prédictions à retourner

        Returns:
            Dict avec style_predictions et artist_predictions
        """
        try:
            # Charger l'image
            image = Image.open(image_path).convert("RGB")

            # Transformer l'image
            input_tensor = INFERENCE_TRANSFORMS(image).unsqueeze(0).to(self.device)

            # Inférence
            with torch.no_grad():
                outputs = self.classifier_model(input_tensor)

            style_logits = outputs["style_logits"]
            artist_logits = outputs["artist_logits"]

            # Softmax pour obtenir les probabilités
            style_probs = torch.softmax(style_logits, dim=-1).squeeze()
            artist_probs = torch.softmax(artist_logits, dim=-1).squeeze()

            # Top-K styles
            style_topk = torch.topk(style_probs, min(top_k, len(style_probs)))
            style_predictions = [
                {
                    "label": self.idx2style[idx.item()],
                    "confidence": float(prob.item() * 100),
                }
                for prob, idx in zip(style_topk.values, style_topk.indices)
            ]

            # Top-K artistes
            artist_topk = torch.topk(artist_probs, min(top_k, len(artist_probs)))
            artist_predictions = [
                {
                    "label": self.idx2artist[idx.item()],
                    "confidence": float(prob.item() * 100),
                }
                for prob, idx in zip(artist_topk.values, artist_topk.indices)
            ]

            return {
                "style_predictions": style_predictions,
                "artist_predictions": artist_predictions,
            }

        except Exception as e:
            return {"error": str(e)}

    def generate_description(self, image_path: str) -> dict:
        """
        Génère une description de l'œuvre d'art.

        Args:
            image_path: Chemin vers l'image

        Returns:
            Dict avec la description
        """
        if self.llava_model is None or self.llava_processor is None:
            return {"error": "LLaVA n'est pas disponible"}

        try:
            # Charger l'image
            image = Image.open(image_path).convert("RGB")

            # Prompt pour l'art
            prompt = "[INST] <image>\nDescribe this artwork in detail. Include information about the style, colors, composition, and mood. [/INST]"

            # Préparation des inputs
            inputs = self.llava_processor(text=prompt, images=image, return_tensors="pt").to(self.device)

            # Génération
            with torch.no_grad():
                output = self.llava_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                )

            # Décoder
            description = self.llava_processor.decode(output[0], skip_special_tokens=True)
            description = description.split("[/INST]")[-1].strip()

            return {"description": description}

        except Exception as e:
            return {"error": str(e)}

    def predict_all(self, image_path: str, top_k: int = 5) -> dict:
        """
        Fait toutes les prédictions (styles, artistes, description).

        Args:
            image_path: Chemin vers l'image
            top_k: Nombre de prédictions à retourner

        Returns:
            Dict avec toutes les prédictions
        """
        # Prédictions de classification
        result = self.predict_style_artist(image_path, top_k)

        # Description si LLaVA est disponible
        if self.llava_model is not None:
            desc_result = self.generate_description(image_path)
            result.update(desc_result)

        return result


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test simple
    print("Test du module d'inférence")

    inference = WikiArtInference(load_llava=False)

    # Chercher une image de test
    test_images_dir = Path(__file__).parent.parent / "test_images"
    test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))

    if test_images:
        test_image = test_images[0]
        print(f"\nTest sur: {test_image.name}")

        result = inference.predict_style_artist(str(test_image), top_k=3)

        print("\n🎨 Styles:")
        for pred in result["style_predictions"]:
            print(f"  - {pred['label']}: {pred['confidence']:.1f}%")

        print("\n👤 Artistes:")
        for pred in result["artist_predictions"]:
            print(f"  - {pred['label']}: {pred['confidence']:.1f}%")
    else:
        print("Aucune image de test trouvée")
