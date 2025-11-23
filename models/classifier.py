"""
classifier.py - Modèle de classification WikiArt

Architecture multi-tâche basée sur Vision Transformer (ViT) avec deux têtes :
- Tête de classification des styles artistiques (27 classes)
- Tête de classification des artistes (1119 classes)

Le backbone ViT est pré-entraîné sur ImageNet et fine-tuné sur WikiArt.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


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


class WikiArtClassifier(nn.Module):
    """
    Classificateur multi-tâche pour WikiArt.

    Architecture:
    ```
    Image (224x224x3)
           │
           ▼
    ┌─────────────────┐
    │   ViT-B/16      │  <- Backbone pré-entraîné ImageNet
    │  (frozen ou     │
    │   fine-tuned)   │
    └────────┬────────┘
             │
             ▼
      Features [CLS]
        (768 dim)
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
        backbone_name: Nom du modèle timm à utiliser
        pretrained: Utiliser les poids pré-entraînés ImageNet
        freeze_backbone: Geler le backbone (entraîner seulement les têtes)
        dropout: Taux de dropout dans les têtes
    """

    def __init__(
        self,
        num_styles: int = 27,
        num_artists: int = 1119,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_styles = num_styles
        self.num_artists = num_artists

        # ===== BACKBONE =====
        # Chargement du ViT pré-entraîné via timm
        # num_classes=0 retire la tête de classification originale
        # et retourne directement les features [CLS]
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Pas de tête, on veut juste les features
        )

        # Récupération de la dimension des features
        # Pour ViT-B/16: 768 dimensions
        self.feature_dim = self.backbone.num_features

        # Gel du backbone si demandé
        # Utile pour la première phase d'entraînement où on entraîne
        # seulement les têtes avec un lr plus élevé
        if freeze_backbone:
            self._freeze_backbone()

        # ===== TÊTES DE CLASSIFICATION =====
        # Dimension cachée = 2x feature_dim pour plus de capacité
        hidden_dim = self.feature_dim * 2

        # Tête pour les styles (27 classes)
        # Tâche plus facile -> MLP standard
        self.style_head = MLPHead(
            in_features=self.feature_dim,
            hidden_features=hidden_dim,
            out_features=num_styles,
            dropout=dropout,
        )

        # Tête pour les artistes (1119 classes)
        # Tâche plus difficile avec beaucoup de classes
        # Même architecture mais la Focal Loss aidera à gérer le déséquilibre
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


class MultiTaskLoss(nn.Module):
    """
    Loss combinée pour l'entraînement multi-tâche.

    Combine les pertes de classification style et artiste avec des poids
    configurables et optionnellement la Focal Loss.

    Loss totale = lambda_style * L_style + lambda_artist * L_artist

    Args:
        style_weight: Poids pour la loss style (défaut: 1.0)
        artist_weight: Poids pour la loss artiste (défaut: 0.5)
                      Plus faible car tâche plus difficile
        use_focal_loss: Utiliser Focal Loss au lieu de CrossEntropy
        focal_gamma: Paramètre gamma de la Focal Loss
        style_class_weights: Poids par classe pour les styles (optionnel)
        artist_class_weights: Poids par classe pour les artistes (optionnel)
    """

    def __init__(
        self,
        style_weight: float = 1.0,
        artist_weight: float = 0.5,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        style_class_weights: Optional[torch.Tensor] = None,
        artist_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.style_weight = style_weight
        self.artist_weight = artist_weight

        # Création des fonctions de loss
        if use_focal_loss:
            self.style_loss_fn = FocalLoss(
                alpha=style_class_weights,
                gamma=focal_gamma,
            )
            self.artist_loss_fn = FocalLoss(
                alpha=artist_class_weights,
                gamma=focal_gamma,
            )
        else:
            # CrossEntropy standard avec poids optionnels
            self.style_loss_fn = nn.CrossEntropyLoss(weight=style_class_weights)
            self.artist_loss_fn = nn.CrossEntropyLoss(weight=artist_class_weights)

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
