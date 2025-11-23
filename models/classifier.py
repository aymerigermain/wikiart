"""
classifier.py - Modèle de classification WikiArt

Architecture multi-tâche basée sur Vision Transformer (ViT) avec deux têtes :
- Tête de classification des styles artistiques (27 classes)
- Tête de classification des artistes (1119 classes)

Supporte deux types de backbone:
- timm: ViT pré-entraîné sur ImageNet (défaut)
- clip: ViT pré-entraîné avec CLIP (meilleur pour l'art)
"""

from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Import conditionnel de CLIP (open_clip est plus flexible que le CLIP original)
try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: open_clip not installed. Install with: pip install open-clip-torch")


class FocalLoss(nn.Module):
    """
    Focal Loss pour gérer le déséquilibre des classes.

    La Focal Loss réduit l'importance des exemples bien classifiés et se concentre
    sur les exemples difficiles. Particulièrement utile quand certaines classes
    ont beaucoup plus d'exemples que d'autres (ex: 1119 artistes).

    Formule: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Poids par classe (optionnel). Si None, toutes les classes ont le même poids.
        gamma: Facteur de focalisation. Plus gamma est élevé, plus on se concentre
               sur les exemples difficiles. Valeur recommandée: 2.0
        reduction: 'mean', 'sum' ou 'none'

    Référence:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha  # Poids par classe
        self.gamma = gamma  # Facteur de focalisation
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcule la Focal Loss.

        Args:
            inputs: Logits du modèle, shape (batch_size, num_classes)
            targets: Labels ground truth, shape (batch_size,)

        Returns:
            Scalar loss (si reduction='mean' ou 'sum') ou tensor (si 'none')
        """
        # Calcul des probabilités avec softmax
        probs = F.softmax(inputs, dim=-1)

        # Récupération de la probabilité de la classe cible
        # targets.unsqueeze(1) -> (batch_size, 1)
        # gather récupère la probabilité à l'indice de la classe cible
        targets_one_hot = targets.unsqueeze(1)
        pt = probs.gather(1, targets_one_hot).squeeze(1)  # (batch_size,)

        # Calcul de la cross-entropy standard: -log(pt)
        ce_loss = -torch.log(pt + 1e-8)  # epsilon pour stabilité numérique

        # Facteur focal: (1 - pt)^gamma
        # Plus pt est proche de 1 (bien classifié), plus ce facteur est petit
        focal_weight = (1 - pt) ** self.gamma

        # Application des poids par classe si fournis
        if self.alpha is not None:
            # Déplacer alpha sur le même device que les inputs
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # Récupérer le poids alpha pour chaque exemple selon sa classe
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_t

        # Loss finale
        loss = focal_weight * ce_loss

        # Réduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MLPHead(nn.Module):
    """
    Tête de classification MLP (Multi-Layer Perceptron).

    Architecture: Linear -> ReLU -> Dropout -> Linear

    Cette architecture avec une couche cachée permet d'apprendre des représentations
    plus complexes que la simple régression logistique, tout en restant légère.

    Args:
        in_features: Dimension des features en entrée (768 pour ViT-B)
        hidden_features: Dimension de la couche cachée
        out_features: Nombre de classes en sortie
        dropout: Taux de dropout pour la régularisation
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # Première couche: projection vers l'espace caché
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # Deuxième couche: projection vers les classes
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Features du backbone, shape (batch_size, in_features)

        Returns:
            Logits, shape (batch_size, out_features)
        """
        return self.net(x)


class CLIPVisionBackbone(nn.Module):
    """
    Wrapper pour utiliser CLIP ViT comme backbone.

    CLIP a été entraîné sur 400M paires image-texte incluant beaucoup d'art,
    ce qui le rend plus adapté que ImageNet pour la classification artistique.
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

        # Charger le modèle CLIP
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        # On ne garde que la partie vision
        self.visual = self.clip_model.visual

        # Dimension des features (768 pour ViT-B, 1024 pour ViT-L)
        self.num_features = self.visual.output_dim

        # Pour compatibilité avec l'API timm (utilisé dans unfreeze_backbone)
        # CLIP ViT utilise transformer.resblocks au lieu de blocks
        if hasattr(self.visual, 'transformer'):
            self.blocks = self.visual.transformer.resblocks
        else:
            self.blocks = []

        print(f"CLIP backbone chargé: {model_name} (pretrained={pretrained})")
        print(f"  - Feature dim: {self.num_features}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extrait les features visuelles."""
        # CLIP normalise différemment, mais on gère ça dans le dataset
        return self.visual(x)


class WikiArtClassifier(nn.Module):
    """
    Classificateur multi-tâche pour WikiArt.

    Architecture:
    ```
    Image (224x224x3)
           │
           ▼
    ┌─────────────────┐
    │   ViT-B/16      │  <- Backbone: timm (ImageNet) ou CLIP
    │  (frozen ou     │
    │   fine-tuned)   │
    └────────┬────────┘
             │
             ▼
      Features [CLS]
      (768 ou 512 dim)
             │
        ┌────┴────┐
        │         │
        ▼         ▼
    ┌───────┐ ┌───────┐
    │ Head  │ │ Head  │
    │ Style │ │Artist │
    └───┬───┘ └───┬───┘
        │         │
        ▼         ▼
    27 classes  1119 classes
    ```

    Args:
        num_styles: Nombre de styles artistiques (défaut: 27)
        num_artists: Nombre d'artistes (défaut: 1119)
        backbone_name: Nom du modèle (timm ou CLIP selon backbone_type)
        backbone_type: "timm" ou "clip"
        pretrained: Utiliser les poids pré-entraînés
        freeze_backbone: Geler le backbone (entraîner seulement les têtes)
        dropout: Taux de dropout dans les têtes
    """

    def __init__(
        self,
        num_styles: int = 27,
        num_artists: int = 1119,
        backbone_name: str = "vit_base_patch16_224",
        backbone_type: Literal["timm", "clip"] = "timm",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_styles = num_styles
        self.num_artists = num_artists
        self.backbone_type = backbone_type

        # ===== BACKBONE =====
        if backbone_type == "clip":
            # Utiliser CLIP comme backbone
            # Mapping des noms pour CLIP
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

        else:  # timm (défaut)
            # Chargement du ViT pré-entraîné via timm
            # num_classes=0 retire la tête de classification originale
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,  # Pas de tête, on veut juste les features
            )
            self.feature_dim = self.backbone.num_features

        # Gel du backbone si demandé
        if freeze_backbone:
            self._freeze_backbone()

        # ===== TÊTES DE CLASSIFICATION =====
        # Dimension cachée = 2x feature_dim pour plus de capacité
        hidden_dim = self.feature_dim * 2

        # Tête pour les styles (27 classes)
        self.style_head = MLPHead(
            in_features=self.feature_dim,
            hidden_features=hidden_dim,
            out_features=num_styles,
            dropout=dropout,
        )

        # Tête pour les artistes (1119 classes)
        self.artist_head = MLPHead(
            in_features=self.feature_dim,
            hidden_features=hidden_dim,
            out_features=num_artists,
            dropout=dropout,
        )

    def _freeze_backbone(self) -> None:
        """Gèle tous les paramètres du backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone gelé - seules les têtes seront entraînées")

    def unfreeze_backbone(self, unfreeze_layers: Optional[int] = None) -> None:
        """
        Dégèle le backbone pour le fine-tuning.

        Args:
            unfreeze_layers: Nombre de couches à dégeler depuis la fin.
                            Si None, dégèle tout le backbone.
        """
        if unfreeze_layers is None:
            # Dégeler tout
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Backbone entièrement dégelé")
        else:
            # Dégeler seulement les dernières couches
            # Pour ViT, les blocs sont dans backbone.blocks
            blocks = list(self.backbone.blocks)
            num_blocks = len(blocks)

            # D'abord tout geler
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Dégeler les N derniers blocs
            for block in blocks[-unfreeze_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

            # Dégeler aussi la layer norm finale et le head
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass complet.

        Args:
            x: Batch d'images, shape (batch_size, 3, 224, 224)

        Returns:
            Dictionnaire contenant:
                - 'style_logits': Logits pour les styles (batch_size, num_styles)
                - 'artist_logits': Logits pour les artistes (batch_size, num_artists)
                - 'features': Features du backbone (batch_size, feature_dim)
        """
        # Extraction des features via le backbone ViT
        features = self.get_features(x)

        # Classification via les deux têtes
        style_logits = self.style_head(features)
        artist_logits = self.artist_head(features)

        return {
            "style_logits": style_logits,
            "artist_logits": artist_logits,
            "features": features,
        }

    def get_trainable_params(self) -> dict[str, list]:
        """
        Retourne les paramètres groupés pour des learning rates différentiels.

        Permet d'utiliser un lr plus faible pour le backbone (fine-tuning)
        et un lr plus élevé pour les têtes (apprentissage from scratch).

        Returns:
            Dictionnaire avec 'backbone' et 'heads' contenant les paramètres.
        """
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.style_head.parameters()) + list(self.artist_head.parameters())

        return {
            "backbone": [p for p in backbone_params if p.requires_grad],
            "heads": head_params,
        }


class LabelSmoothingFocalLoss(nn.Module):
    """
    Focal Loss avec Label Smoothing pour réduire l'overconfidence.

    Combine les avantages de:
    - Focal Loss: focus sur les exemples difficiles
    - Label Smoothing: évite que le modèle soit trop confiant

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

        # Appliquer label smoothing aux targets
        # Au lieu de [0, 0, 1, 0, 0], on obtient [0.025, 0.025, 0.9, 0.025, 0.025]
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Log softmax pour la stabilité numérique
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p)^gamma pour chaque classe
        focal_weights = (1 - probs) ** self.gamma

        # Loss avec label smoothing
        loss = -smooth_targets * focal_weights * log_probs

        # Application des poids par classe si fournis
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # Pondérer par alpha
            loss = loss * self.alpha.unsqueeze(0)

        # Somme sur les classes
        loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiTaskLoss(nn.Module):
    """
    Loss combinée pour l'entraînement multi-tâche.

    Combine les pertes de classification style et artiste avec des poids
    configurables et optionnellement la Focal Loss avec Label Smoothing.

    Loss totale = lambda_style * L_style + lambda_artist * L_artist

    Args:
        style_weight: Poids pour la loss style (défaut: 1.0)
        artist_weight: Poids pour la loss artiste (défaut: 0.5)
        use_focal_loss: Utiliser Focal Loss au lieu de CrossEntropy
        focal_gamma: Paramètre gamma de la Focal Loss
        label_smoothing: Facteur de label smoothing (défaut: 0.0)
        style_class_weights: Poids par classe pour les styles (optionnel)
        artist_class_weights: Poids par classe pour les artistes (optionnel)
    """

    def __init__(
        self,
        style_weight: float = 1.0,
        artist_weight: float = 0.5,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        style_class_weights: Optional[torch.Tensor] = None,
        artist_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.style_weight = style_weight
        self.artist_weight = artist_weight

        # Création des fonctions de loss
        if use_focal_loss:
            if label_smoothing > 0:
                # Focal Loss + Label Smoothing (recommandé pour éviter overfitting)
                self.style_loss_fn = LabelSmoothingFocalLoss(
                    alpha=style_class_weights,
                    gamma=focal_gamma,
                    smoothing=label_smoothing,
                )
                self.artist_loss_fn = LabelSmoothingFocalLoss(
                    alpha=artist_class_weights,
                    gamma=focal_gamma,
                    smoothing=label_smoothing,
                )
            else:
                # Focal Loss standard
                self.style_loss_fn = FocalLoss(
                    alpha=style_class_weights,
                    gamma=focal_gamma,
                )
                self.artist_loss_fn = FocalLoss(
                    alpha=artist_class_weights,
                    gamma=focal_gamma,
                )
        else:
            # CrossEntropy avec label smoothing optionnel
            self.style_loss_fn = nn.CrossEntropyLoss(
                weight=style_class_weights,
                label_smoothing=label_smoothing,
            )
            self.artist_loss_fn = nn.CrossEntropyLoss(
                weight=artist_class_weights,
                label_smoothing=label_smoothing,
            )

    def forward(
        self,
        style_logits: torch.Tensor,
        artist_logits: torch.Tensor,
        style_targets: torch.Tensor,
        artist_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Calcule la loss combinée.

        Args:
            style_logits: Prédictions style (batch_size, num_styles)
            artist_logits: Prédictions artiste (batch_size, num_artists)
            style_targets: Labels style (batch_size,)
            artist_targets: Labels artiste (batch_size,)

        Returns:
            Dictionnaire contenant:
                - 'total': Loss totale combinée
                - 'style': Loss style seule
                - 'artist': Loss artiste seule
        """
        style_loss = self.style_loss_fn(style_logits, style_targets)
        artist_loss = self.artist_loss_fn(artist_logits, artist_targets)

        total_loss = self.style_weight * style_loss + self.artist_weight * artist_loss

        return {
            "total": total_loss,
            "style": style_loss,
            "artist": artist_loss,
        }


def create_model(
    num_styles: int = 27,
    num_artists: int = 1119,
    backbone: str = "vit_base_patch16_224",
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> WikiArtClassifier:
    """
    Factory function pour créer le modèle avec la configuration recommandée.

    Configuration par défaut optimisée pour WikiArt:
    - ViT-B/16 pré-entraîné ImageNet
    - Backbone gelé initialement (phase 1)
    - Dropout 0.1 dans les têtes

    Args:
        num_styles: Nombre de classes style
        num_artists: Nombre de classes artiste
        backbone: Nom du modèle timm
        pretrained: Utiliser les poids pré-entraînés
        freeze_backbone: Geler le backbone (recommandé pour phase 1)

    Returns:
        Instance de WikiArtClassifier configurée
    """
    model = WikiArtClassifier(
        num_styles=num_styles,
        num_artists=num_artists,
        backbone_name=backbone,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=0.1,
    )

    # Affichage des infos du modèle
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Modèle créé: {backbone}")
    print(f"  - Paramètres totaux: {total_params:,}")
    print(f"  - Paramètres entraînables: {trainable_params:,}")
    print(f"  - Backbone gelé: {freeze_backbone}")

    return model


if __name__ == "__main__":
    # Test du modèle
    print("=" * 50)
    print("Test du modèle WikiArtClassifier")
    print("=" * 50)

    # Création du modèle
    model = create_model(
        num_styles=27,
        num_artists=1119,
        freeze_backbone=True,
    )

    # Test forward pass
    print("\nTest forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch de 4 images

    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Style logits shape: {outputs['style_logits'].shape}")
    print(f"  - Artist logits shape: {outputs['artist_logits'].shape}")
    print(f"  - Features shape: {outputs['features'].shape}")

    # Test de la loss
    print("\nTest MultiTaskLoss...")
    criterion = MultiTaskLoss(
        style_weight=1.0,
        artist_weight=0.5,
        use_focal_loss=True,
    )

    # Labels factices
    style_targets = torch.randint(0, 27, (4,))
    artist_targets = torch.randint(0, 1119, (4,))

    losses = criterion(
        outputs["style_logits"],
        outputs["artist_logits"],
        style_targets,
        artist_targets,
    )

    print(f"  - Total loss: {losses['total'].item():.4f}")
    print(f"  - Style loss: {losses['style'].item():.4f}")
    print(f"  - Artist loss: {losses['artist'].item():.4f}")

    # Test dégel progressif
    print("\nTest dégel progressif du backbone...")
    model.unfreeze_backbone(unfreeze_layers=4)

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Paramètres entraînables après dégel: {trainable_after:,}")

    print("\n" + "=" * 50)
    print("Tous les tests passés!")
    print("=" * 50)
