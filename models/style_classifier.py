"""
style_classifier.py - Style-only classifier for WikiArt

Single-task architecture focused exclusively on artistic style classification.
Simplified version of the multi-task classifier with optimizations for style prediction.
"""

from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Import conditionnel de CLIP
try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: open_clip not installed. Install with: pip install open-clip-torch")


class FocalLoss(nn.Module):
    """
    Focal Loss pour gérer le déséquilibre des classes.

    Args:
        alpha: Poids par classe (optionnel)
        gamma: Facteur de focalisation (défaut: 2.0)
        reduction: 'mean', 'sum' ou 'none'
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcule la Focal Loss.

        Args:
            inputs: Logits du modèle, shape (batch_size, num_classes)
            targets: Labels ground truth, shape (batch_size,)

        Returns:
            Scalar loss
        """
        probs = F.softmax(inputs, dim=-1)
        targets_one_hot = targets.unsqueeze(1)
        pt = probs.gather(1, targets_one_hot).squeeze(1)
        ce_loss = -torch.log(pt + 1e-8)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingFocalLoss(nn.Module):
    """
    Focal Loss avec Label Smoothing pour réduire l'overconfidence.

    Args:
        alpha: Poids par classe (optionnel)
        gamma: Facteur de focalisation (défaut: 2.0)
        smoothing: Facteur de label smoothing (défaut: 0.1)
        reduction: 'mean', 'sum' ou 'none'
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcule la Focal Loss avec Label Smoothing."""
        num_classes = inputs.size(-1)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        focal_weights = (1 - probs) ** self.gamma
        loss = -smooth_targets * focal_weights * log_probs

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            loss = loss * self.alpha.unsqueeze(0)

        loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CLIPVisionBackbone(nn.Module):
    """
    Wrapper pour utiliser CLIP ViT comme backbone.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
    ):
        super().__init__()

        if not CLIP_AVAILABLE:
            raise ImportError(
                "open_clip is required for CLIP backbone. "
                "Install with: pip install open-clip-torch"
            )

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        self.visual = self.clip_model.visual
        self.num_features = self.visual.output_dim

        if hasattr(self.visual, 'transformer'):
            self.blocks = self.visual.transformer.resblocks
        else:
            self.blocks = []

        print(f"CLIP backbone chargé: {model_name} (pretrained={pretrained})")
        print(f"  - Feature dim: {self.num_features}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extrait les features visuelles."""
        return self.visual(x)


class StyleClassifier(nn.Module):
    """
    Classificateur simple pour la prédiction de style artistique.

    Architecture:
    ```
    Image (224x224x3)
           │
           ▼
    ┌─────────────────┐
    │   ViT Backbone  │  <- timm (ImageNet) ou CLIP
    │  (frozen ou     │
    │   fine-tuned)   │
    └────────┬────────┘
             │
             ▼
      Features [CLS]
      (768 ou 512 dim)
             │
             ▼
      ┌─────────────┐
      │  MLP Head   │
      └──────┬──────┘
             │
             ▼
        N classes
    ```

    Args:
        num_classes: Nombre de styles artistiques
        backbone_name: Nom du modèle (timm ou CLIP selon backbone_type)
        backbone_type: "timm" ou "clip"
        pretrained: Utiliser les poids pré-entraînés
        freeze_backbone: Geler le backbone
        dropout: Taux de dropout dans la tête
        hidden_dim_multiplier: Multiplicateur pour la couche cachée (défaut: 2)
    """

    def __init__(
        self,
        num_classes: int = 22,
        backbone_name: str = "vit_base_patch16_224",
        backbone_type: Literal["timm", "clip"] = "timm",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        hidden_dim_multiplier: int = 2,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.backbone_type = backbone_type

        # ===== BACKBONE =====
        if backbone_type == "clip":
            clip_model_map = {
                "vit_base_patch16_224": "ViT-B-16",
                "vit_large_patch14_224": "ViT-L-14",
                "ViT-B-16": "ViT-B-16",
                "ViT-L-14": "ViT-L-14",
                "ViT-B-32": "ViT-B-32",
            }
            clip_name = clip_model_map.get(backbone_name, backbone_name)

            self.backbone = CLIPVisionBackbone(
                model_name=clip_name,
                pretrained="openai" if pretrained else None,
            )
            self.feature_dim = self.backbone.num_features

        else:  # timm
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
            )
            self.feature_dim = self.backbone.num_features

        if freeze_backbone:
            self._freeze_backbone()

        # ===== CLASSIFICATION HEAD =====
        hidden_dim = self.feature_dim * hidden_dim_multiplier

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _freeze_backbone(self) -> None:
        """Gèle tous les paramètres du backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone gelé - seule la tête sera entraînée")

    def unfreeze_backbone(self, unfreeze_layers: Optional[int] = None) -> None:
        """
        Dégèle le backbone pour le fine-tuning.

        Args:
            unfreeze_layers: Nombre de couches à dégeler depuis la fin.
                            Si None, dégèle tout le backbone.
        """
        if unfreeze_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Backbone entièrement dégelé")
        else:
            blocks = list(self.backbone.blocks)
            num_blocks = len(blocks)

            for param in self.backbone.parameters():
                param.requires_grad = False

            for block in blocks[-unfreeze_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

            if hasattr(self.backbone, 'norm'):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True

            print(f"Derniers {unfreeze_layers}/{num_blocks} blocs dégelés")

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait les features du backbone.

        Args:
            x: Images, shape (batch_size, 3, 224, 224)

        Returns:
            Features [CLS], shape (batch_size, feature_dim)
        """
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Batch d'images, shape (batch_size, 3, 224, 224)

        Returns:
            Logits, shape (batch_size, num_classes)
        """
        features = self.get_features(x)
        logits = self.head(features)
        return logits

    def get_trainable_params(self) -> dict[str, list]:
        """
        Retourne les paramètres groupés pour des learning rates différentiels.

        Returns:
            Dictionnaire avec 'backbone' et 'head' contenant les paramètres.
        """
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())

        return {
            "backbone": [p for p in backbone_params if p.requires_grad],
            "head": head_params,
        }


class StyleLoss(nn.Module):
    """
    Loss pour la classification de style avec support de différentes variantes.

    Args:
        use_focal_loss: Utiliser Focal Loss au lieu de CrossEntropy
        focal_gamma: Paramètre gamma de la Focal Loss
        label_smoothing: Facteur de label smoothing (défaut: 0.0)
        class_weights: Poids par classe (optionnel)
    """

    def __init__(
        self,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if use_focal_loss:
            if label_smoothing > 0:
                self.loss_fn = LabelSmoothingFocalLoss(
                    alpha=class_weights,
                    gamma=focal_gamma,
                    smoothing=label_smoothing,
                )
            else:
                self.loss_fn = FocalLoss(
                    alpha=class_weights,
                    gamma=focal_gamma,
                )
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcule la loss.

        Args:
            logits: Prédictions, shape (batch_size, num_classes)
            targets: Labels, shape (batch_size,)

        Returns:
            Scalar loss
        """
        return self.loss_fn(logits, targets)


def create_style_classifier(
    num_classes: int = 22,
    backbone: str = "vit_base_patch16_224",
    backbone_type: Literal["timm", "clip"] = "timm",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.1,
) -> StyleClassifier:
    """
    Factory function pour créer le modèle de classification de style.

    Args:
        num_classes: Nombre de classes style
        backbone: Nom du modèle (timm ou CLIP)
        backbone_type: "timm" ou "clip"
        pretrained: Utiliser les poids pré-entraînés
        freeze_backbone: Geler le backbone (recommandé pour phase 1)
        dropout: Taux de dropout

    Returns:
        Instance de StyleClassifier configurée
    """
    model = StyleClassifier(
        num_classes=num_classes,
        backbone_name=backbone,
        backbone_type=backbone_type,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Modèle créé: {backbone} (type: {backbone_type})")
    print(f"  - Paramètres totaux: {total_params:,}")
    print(f"  - Paramètres entraînables: {trainable_params:,}")
    print(f"  - Backbone gelé: {freeze_backbone}")

    return model


if __name__ == "__main__":
    # Test du modèle
    print("=" * 50)
    print("Test du StyleClassifier")
    print("=" * 50)

    # Création du modèle
    model = create_style_classifier(
        num_classes=22,
        freeze_backbone=True,
    )

    # Test forward pass
    print("\nTest forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {outputs.shape}")
    print(f"  - Expected: (4, 22)")

    # Test de la loss
    print("\nTest StyleLoss...")
    criterion = StyleLoss(
        use_focal_loss=True,
        label_smoothing=0.1,
    )

    targets = torch.randint(0, 22, (4,))
    loss = criterion(outputs, targets)

    print(f"  - Loss: {loss.item():.4f}")

    # Test dégel progressif
    print("\nTest dégel progressif du backbone...")
    model.unfreeze_backbone(unfreeze_layers=4)

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Paramètres entraînables après dégel: {trainable_after:,}")

    print("\n" + "=" * 50)
    print("Tous les tests passés!")
    print("=" * 50)