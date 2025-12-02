"""
test_llava.py - Test de génération de descriptions avec LLaVA

Script simple pour tester LLaVA sur quelques images.
"""

from pathlib import Path
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def generate_description(image_path: str, processor, model, device: str = "cuda"):
    """
    Génère une description pour une image.

    Args:
        image_path: Chemin vers l'image
        processor: LLaVA processor
        model: LLaVA model
        device: Device (cuda/cpu)

    Returns:
        Description générée
    """
    # Charger l'image
    image = Image.open(image_path).convert("RGB")

    # Prompt pour l'art
    prompt = "[INST] <image>\nDescribe this artwork in detail. Include information about the style, colors, composition, and mood. [/INST]"

    # Préparation des inputs (text first, then images)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    # Génération
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,  # Greedy decoding pour stabilité
        )

    # Décoder
    description = processor.decode(output[0], skip_special_tokens=True)

    # Nettoyer (retirer le prompt)
    description = description.split("[/INST]")[-1].strip()

    return description


def main():
    print("=" * 80)
    print("Test LLaVA pour génération de descriptions d'art")
    print("=" * 80)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Charger le modèle LLaVA-NeXT (LLaVA 1.6)
    # Modèle 7B = bon compromis qualité/vitesse
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    print(f"\nChargement du modèle: {model_id}")
    print("(Cela peut prendre quelques minutes au premier lancement)")

    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    print("Modèle chargé !")

    # Images de test
    test_images_dir = Path("test_images")
    test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))

    print(f"\n{len(test_images)} images trouvées dans {test_images_dir}/")

    # Générer des descriptions
    for image_path in test_images:
        print("\n" + "-" * 80)
        print(f"Image: {image_path.name}")
        print("-" * 80)

        description = generate_description(str(image_path), processor, model, device)

        print(f"\nDescription:\n{description}")

    print("\n" + "=" * 80)
    print("Test terminé !")
    print("=" * 80)


if __name__ == "__main__":
    main()
