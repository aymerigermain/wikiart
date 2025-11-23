"""
train.py - Script d'entraînement pour WikiArt

Entraînement multi-tâche du classificateur ViT avec:
- Phase 1: Backbone gelé, entraînement des têtes uniquement
- Phase 2: Fine-tuning progressif du backbone
- Logging des métriques (loss, accuracy, F1)
- Sauvegarde des meilleurs modèles
"""

import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from data import download_or_use_cached, discover_structure, build_label_mappings
from dataset import create_splits, create_dataloaders, get_class_weights
from models.classifier import WikiArtClassifier, MultiTaskLoss


# ============================================================================
# CONFIGURATION PAR DÉFAUT
# ============================================================================

DEFAULT_CONFIG = {
    # Données
    "batch_size": 192,  # Augmenté pour meilleure utilisation GPU
    "num_workers": 8,
    "image_size": 224,

    # Modèle
    "backbone": "ViT-B-16", # Avec timm : "vit_base_patch16_224",
    "backbone_type": "clip",  # "timm" (ImageNet) ou "clip" (recommandé pour l'art)
    "dropout": 0.3,  # Augmenté de 0.1 à 0.3 pour régularisation

    # Entraînement Phase 1 (backbone gelé)
    "phase1_epochs": 15,  # Augmenté pour mieux entraîner les têtes
    "phase1_lr": 5e-4,    # Réduit de 1e-3 pour éviter l'overfitting rapide

    # Entraînement Phase 2 (fine-tuning)
    "phase2_epochs": 25,
    "phase2_lr_backbone": 5e-6,   # Réduit pour fine-tuning plus doux
    "phase2_lr_heads": 5e-5,      # Réduit pour stabilité
    "unfreeze_layers": 4,         # Moins de couches dégelées (était 6)

    # Optimisation
    "weight_decay": 0.05,  # Augmenté de 0.01 pour plus de régularisation
    "warmup_epochs": 3,    # Plus de warmup
    "use_amp": True,

    # Loss
    "style_weight": 1.0,
    "artist_weight": 0.3,      # Réduit (artiste est très difficile)
    "use_focal_loss": True,
    "focal_gamma": 2.0,
    "label_smoothing": 0.1,    # NOUVEAU: évite l'overconfidence

    # Early Stopping
    "early_stopping_patience": 4,  # Arrêter si pas d'amélioration pendant 4 epochs
    "early_stopping_min_delta": 0.001,  # Amélioration minimale requise

    # Sauvegarde
    "save_dir": "checkpoints",
    "save_every": 5,
}


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def format_time(seconds: float) -> str:
    """Formate un temps en secondes en format lisible (HH:MM:SS ou MM:SS)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def setup_device() -> torch.device:
    """Configure et retourne le device (GPU si disponible)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Optimisations GPU
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        print(f"Utilisation du GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("TensorFloat32 activé, cuDNN benchmark activé")
    else:
        device = torch.device("cpu")
        print("GPU non disponible, utilisation du CPU")
    return device


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> dict[str, float]:
    """
    Calcule les métriques de classification.

    Args:
        preds: Logits du modèle (batch_size, num_classes)
        targets: Labels ground truth (batch_size,)
        k: K pour le calcul du top-k accuracy

    Returns:
        Dictionnaire avec top1_acc, top5_acc
    """
    # Top-1 accuracy
    _, pred_classes = preds.max(dim=1)
    top1_correct = (pred_classes == targets).float().sum()
    top1_acc = top1_correct / targets.size(0)

    # Top-k accuracy
    _, topk_preds = preds.topk(k, dim=1)
    targets_expanded = targets.unsqueeze(1).expand_as(topk_preds)
    topk_correct = (topk_preds == targets_expanded).any(dim=1).float().sum()
    topk_acc = topk_correct / targets.size(0)

    return {
        "top1_acc": top1_acc.item(),
        f"top{k}_acc": topk_acc.item(),
    }


class MetricTracker:
    """
    Suivi des métriques pendant l'entraînement.

    Accumule les métriques sur plusieurs batches et calcule les moyennes.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Réinitialise les compteurs."""
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: dict, batch_size: int = 1):
        """
        Met à jour les métriques avec un nouveau batch.

        Args:
            metrics: Dictionnaire de métriques pour ce batch
            batch_size: Taille du batch pour la moyenne pondérée
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            self.metrics[key] += value * batch_size
            self.counts[key] += batch_size

    def get_averages(self) -> dict[str, float]:
        """Retourne les moyennes des métriques."""
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics
        }


class EarlyStopping:
    """
    Early Stopping pour éviter l'overfitting.

    Arrête l'entraînement si la métrique de validation ne s'améliore plus.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "max"):
        """
        Args:
            patience: Nombre d'epochs sans amélioration avant l'arrêt
            min_delta: Amélioration minimale pour considérer un progrès
            mode: 'max' pour accuracy, 'min' pour loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Vérifie si l'entraînement doit s'arrêter.

        Args:
            score: Métrique de validation actuelle

        Returns:
            True si l'entraînement doit s'arrêter
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def reset(self):
        """Réinitialise l'early stopping (pour une nouvelle phase)."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


# ============================================================================
# BOUCLES D'ENTRAÎNEMENT ET VALIDATION
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True,
) -> dict[str, float]:
    """
    Entraîne le modèle pour une epoch.

    Args:
        model: Le modèle à entraîner
        dataloader: DataLoader d'entraînement
        criterion: Fonction de loss (MultiTaskLoss)
        optimizer: Optimiseur
        device: Device (cuda/cpu)
        scaler: GradScaler pour mixed precision
        use_amp: Utiliser automatic mixed precision

    Returns:
        Dictionnaire des métriques moyennes sur l'epoch
    """
    model.train()
    tracker = MetricTracker()

    # Barre de progression
    pbar = tqdm(dataloader, desc="Train", leave=False)

    for batch in pbar:
        # Déplacer les données sur le device
        images = batch["image"].to(device)
        style_targets = batch["style"].to(device)
        artist_targets = batch["artist"].to(device)
        batch_size = images.size(0)

        # Forward pass avec mixed precision optionnel
        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast("cuda"):
                outputs = model(images)
                losses = criterion(
                    outputs["style_logits"],
                    outputs["artist_logits"],
                    style_targets,
                    artist_targets,
                )

            # Backward pass avec scaling
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            losses = criterion(
                outputs["style_logits"],
                outputs["artist_logits"],
                style_targets,
                artist_targets,
            )
            losses["total"].backward()
            optimizer.step()

        # Calcul des métriques (sans gradient)
        with torch.no_grad():
            style_metrics = compute_metrics(outputs["style_logits"], style_targets)
            artist_metrics = compute_metrics(outputs["artist_logits"], artist_targets)

        # Mise à jour du tracker
        metrics = {
            "loss_total": losses["total"].item(),
            "loss_style": losses["style"].item(),
            "loss_artist": losses["artist"].item(),
            "style_top1": style_metrics["top1_acc"],
            "style_top5": style_metrics["top5_acc"],
            "artist_top1": artist_metrics["top1_acc"],
            "artist_top5": artist_metrics["top5_acc"],
        }
        tracker.update(metrics, batch_size)

        # Mise à jour de la barre de progression
        pbar.set_postfix({
            "loss": f"{losses['total'].item():.4f}",
            "s_acc": f"{style_metrics['top1_acc']:.2%}",
            "a_acc": f"{artist_metrics['top1_acc']:.2%}",
        })

    return tracker.get_averages()


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Évalue le modèle sur le set de validation.

    Args:
        model: Le modèle à évaluer
        dataloader: DataLoader de validation
        criterion: Fonction de loss
        device: Device (cuda/cpu)

    Returns:
        Dictionnaire des métriques moyennes
    """
    model.eval()
    tracker = MetricTracker()

    pbar = tqdm(dataloader, desc="Val", leave=False)

    for batch in pbar:
        images = batch["image"].to(device)
        style_targets = batch["style"].to(device)
        artist_targets = batch["artist"].to(device)
        batch_size = images.size(0)

        # Forward pass
        outputs = model(images)
        losses = criterion(
            outputs["style_logits"],
            outputs["artist_logits"],
            style_targets,
            artist_targets,
        )

        # Métriques
        style_metrics = compute_metrics(outputs["style_logits"], style_targets)
        artist_metrics = compute_metrics(outputs["artist_logits"], artist_targets)

        metrics = {
            "loss_total": losses["total"].item(),
            "loss_style": losses["style"].item(),
            "loss_artist": losses["artist"].item(),
            "style_top1": style_metrics["top1_acc"],
            "style_top5": style_metrics["top5_acc"],
            "artist_top1": artist_metrics["top1_acc"],
            "artist_top5": artist_metrics["top5_acc"],
        }
        tracker.update(metrics, batch_size)

    return tracker.get_averages()


# ============================================================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# ============================================================================

def train(config: dict) -> None:
    """
    Fonction principale d'entraînement.

    Exécute l'entraînement en deux phases:
    1. Phase 1: Backbone gelé, entraînement des têtes
    2. Phase 2: Fine-tuning progressif du backbone

    Args:
        config: Dictionnaire de configuration
    """
    print("=" * 60)
    print("ENTRAÎNEMENT WIKIART CLASSIFIER")
    print("=" * 60)

    # ===== SETUP =====
    device = setup_device()

    # Création du dossier de sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config["save_dir"]) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDossier de sauvegarde: {save_dir}")

    # Sauvegarde de la config
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ===== TENSORBOARD =====
    writer = SummaryWriter(log_dir=save_dir / "tensorboard")
    print(f"TensorBoard: tensorboard --logdir {save_dir / 'tensorboard'}")

    # ===== DONNÉES =====
    print("\n" + "-" * 40)
    print("Chargement des données...")
    print("-" * 40)

    dataset_path = download_or_use_cached()
    structure = discover_structure(dataset_path)
    style2idx, idx2style, artist2idx, idx2artist = build_label_mappings(structure)

    num_styles = len(structure["styles"])
    num_artists = len(structure["artists"])
    print(f"Styles: {num_styles}, Artistes: {num_artists}")

    # Création des splits
    train_samples, val_samples, test_samples = create_splits(
        structure, style2idx, artist2idx
    )
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Création des DataLoaders (avec normalisation adaptée au backbone)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples,
        val_samples,
        test_samples,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        use_weighted_sampler=True,
        num_styles=num_styles,
        persistent_workers=True,
        backbone_type=config["backbone_type"],
    )

    # Calcul des poids de classe pour la loss
    style_weights = get_class_weights(train_samples, num_styles, label_idx=1)
    artist_weights = get_class_weights(train_samples, num_artists, label_idx=2)

    # ===== MODÈLE =====
    print("\n" + "-" * 40)
    print("Création du modèle...")
    print("-" * 40)

    model = WikiArtClassifier(
        num_styles=num_styles,
        num_artists=num_artists,
        backbone_name=config["backbone"],
        backbone_type=config["backbone_type"],
        pretrained=True,
        freeze_backbone=True,  # Phase 1: backbone gelé
        dropout=config["dropout"],
    )
    print(f"Backbone: {config['backbone']} (type: {config['backbone_type']})")
    model = model.to(device)
    # torch.compile désactivé - la compilation JIT bloque plusieurs minutes au premier batch
    # model = torch.compile(model)

    # ===== LOSS =====
    criterion = MultiTaskLoss(
        style_weight=config["style_weight"],
        artist_weight=config["artist_weight"],
        use_focal_loss=config["use_focal_loss"],
        focal_gamma=config["focal_gamma"],
        label_smoothing=config.get("label_smoothing", 0.0),
        style_class_weights=style_weights.to(device),
        artist_class_weights=artist_weights.to(device),
    )

    # ===== MIXED PRECISION =====
    scaler = GradScaler('cuda') if config["use_amp"] else None

    # Historique des métriques
    history = {
        "train": [],
        "val": [],
    }
    best_val_acc = 0.0

    # =========================================================================
    # PHASE 1: ENTRAÎNEMENT DES TÊTES (BACKBONE GELÉ)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Entraînement des têtes (backbone gelé)")
    print("=" * 60)

    # Temps de début de l'entraînement
    training_start_time = time.time()

    # Early Stopping pour Phase 1
    early_stopping = EarlyStopping(
        patience=config.get("early_stopping_patience", 5),
        min_delta=config.get("early_stopping_min_delta", 0.001),
        mode="max",
    )

    # Optimiseur pour les têtes uniquement
    optimizer = AdamW(
        model.get_trainable_params()["heads"],
        lr=config["phase1_lr"],
        weight_decay=config["weight_decay"],
    )

    # Scheduler avec warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config["warmup_epochs"],
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["phase1_epochs"] - config["warmup_epochs"],
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config["warmup_epochs"]],
    )

    for epoch in range(config["phase1_epochs"]):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{config['phase1_epochs']} (Phase 1)")
        print("-" * 40)

        # Entraînement
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config["use_amp"]
        )

        # Validation
        val_metrics = validate(model, val_loader, criterion, device)

        # Mise à jour du scheduler
        scheduler.step()

        # Calcul des temps
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time

        # Logging
        print(f"Train - Loss: {train_metrics['loss_total']:.4f}, "
              f"Style Acc: {train_metrics['style_top1']:.2%}, "
              f"Artist Acc: {train_metrics['artist_top1']:.2%}")
        print(f"Val   - Loss: {val_metrics['loss_total']:.4f}, "
              f"Style Acc: {val_metrics['style_top1']:.2%}, "
              f"Artist Acc: {val_metrics['artist_top1']:.2%}")
        print(f"Temps - Epoch: {format_time(epoch_time)}, Total: {format_time(total_elapsed)}")

        # TensorBoard logging
        global_epoch = epoch
        writer.add_scalars("Loss", {
            "train": train_metrics["loss_total"],
            "val": val_metrics["loss_total"],
        }, global_epoch)
        writer.add_scalars("Loss/Style", {
            "train": train_metrics["loss_style"],
            "val": val_metrics["loss_style"],
        }, global_epoch)
        writer.add_scalars("Loss/Artist", {
            "train": train_metrics["loss_artist"],
            "val": val_metrics["loss_artist"],
        }, global_epoch)
        writer.add_scalars("Accuracy/Style_Top1", {
            "train": train_metrics["style_top1"],
            "val": val_metrics["style_top1"],
        }, global_epoch)
        writer.add_scalars("Accuracy/Style_Top5", {
            "train": train_metrics["style_top5"],
            "val": val_metrics["style_top5"],
        }, global_epoch)
        writer.add_scalars("Accuracy/Artist_Top1", {
            "train": train_metrics["artist_top1"],
            "val": val_metrics["artist_top1"],
        }, global_epoch)
        writer.add_scalars("Accuracy/Artist_Top5", {
            "train": train_metrics["artist_top5"],
            "val": val_metrics["artist_top5"],
        }, global_epoch)
        writer.add_scalar("Time/Epoch_seconds", epoch_time, global_epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], global_epoch)

        # Sauvegarde de l'historique
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Sauvegarde du meilleur modèle (basé sur style accuracy)
        if val_metrics["style_top1"] > best_val_acc:
            best_val_acc = val_metrics["style_top1"]
            torch.save({
                "epoch": epoch,
                "phase": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, save_dir / "best_model.pt")
            print(f"  -> Nouveau meilleur modèle sauvegardé (acc: {best_val_acc:.2%})")

        # Early Stopping check
        if early_stopping(val_metrics["style_top1"]):
            print(f"\n  Early stopping déclenché après {epoch + 1} epochs (Phase 1)")
            print(f"  Pas d'amélioration depuis {early_stopping.patience} epochs")
            break

    # =========================================================================
    # PHASE 2: FINE-TUNING DU BACKBONE
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning du backbone")
    print("=" * 60)

    # Reset Early Stopping pour Phase 2
    early_stopping.reset()

    # Dégeler les dernières couches du backbone
    model.unfreeze_backbone(unfreeze_layers=config["unfreeze_layers"])

    # Nouvel optimiseur avec learning rates différentiels
    param_groups = model.get_trainable_params()
    optimizer = AdamW([
        {"params": param_groups["backbone"], "lr": config["phase2_lr_backbone"]},
        {"params": param_groups["heads"], "lr": config["phase2_lr_heads"]},
    ], weight_decay=config["weight_decay"])

    # Nouveau scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config["phase2_epochs"])

    for epoch in range(config["phase2_epochs"]):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{config['phase2_epochs']} (Phase 2)")
        print("-" * 40)

        # Entraînement
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config["use_amp"]
        )

        # Validation
        val_metrics = validate(model, val_loader, criterion, device)

        # Mise à jour du scheduler
        scheduler.step()

        # Calcul des temps
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time

        # Logging
        print(f"Train - Loss: {train_metrics['loss_total']:.4f}, "
              f"Style Acc: {train_metrics['style_top1']:.2%}, "
              f"Artist Acc: {train_metrics['artist_top1']:.2%}")
        print(f"Val   - Loss: {val_metrics['loss_total']:.4f}, "
              f"Style Acc: {val_metrics['style_top1']:.2%}, "
              f"Artist Acc: {val_metrics['artist_top1']:.2%}")
        print(f"Temps - Epoch: {format_time(epoch_time)}, Total: {format_time(total_elapsed)}")

        # TensorBoard logging
        global_epoch = config["phase1_epochs"] + epoch
        writer.add_scalars("Loss", {
            "train": train_metrics["loss_total"],
            "val": val_metrics["loss_total"],
        }, global_epoch)
        writer.add_scalars("Loss/Style", {
            "train": train_metrics["loss_style"],
            "val": val_metrics["loss_style"],
        }, global_epoch)
        writer.add_scalars("Loss/Artist", {
            "train": train_metrics["loss_artist"],
            "val": val_metrics["loss_artist"],
        }, global_epoch)
        writer.add_scalars("Accuracy/Style_Top1", {
            "train": train_metrics["style_top1"],
            "val": val_metrics["style_top1"],
        }, global_epoch)
        writer.add_scalars("Accuracy/Style_Top5", {
            "train": train_metrics["style_top5"],
            "val": val_metrics["style_top5"],
        }, global_epoch)
        writer.add_scalars("Accuracy/Artist_Top1", {
            "train": train_metrics["artist_top1"],
            "val": val_metrics["artist_top1"],
        }, global_epoch)
        writer.add_scalars("Accuracy/Artist_Top5", {
            "train": train_metrics["artist_top5"],
            "val": val_metrics["artist_top5"],
        }, global_epoch)
        writer.add_scalar("Time/Epoch_seconds", epoch_time, global_epoch)
        writer.add_scalar("LearningRate/Backbone", optimizer.param_groups[0]["lr"], global_epoch)
        writer.add_scalar("LearningRate/Heads", optimizer.param_groups[1]["lr"], global_epoch)

        # Sauvegarde de l'historique
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Sauvegarde du meilleur modèle
        if val_metrics["style_top1"] > best_val_acc:
            best_val_acc = val_metrics["style_top1"]
            torch.save({
                "epoch": config["phase1_epochs"] + epoch,
                "phase": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, save_dir / "best_model.pt")
            print(f"  -> Nouveau meilleur modèle sauvegardé (acc: {best_val_acc:.2%})")

        # Sauvegarde périodique
        if (epoch + 1) % config["save_every"] == 0:
            torch.save({
                "epoch": config["phase1_epochs"] + epoch,
                "phase": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, save_dir / f"checkpoint_epoch_{config['phase1_epochs'] + epoch + 1}.pt")

        # Early Stopping check
        if early_stopping(val_metrics["style_top1"]):
            print(f"\n  Early stopping déclenché après {epoch + 1} epochs (Phase 2)")
            print(f"  Pas d'amélioration depuis {early_stopping.patience} epochs")
            break

    # =========================================================================
    # ÉVALUATION FINALE
    # =========================================================================
    print("\n" + "=" * 60)
    print("ÉVALUATION FINALE SUR LE SET DE TEST")
    print("=" * 60)

    # Charger le meilleur modèle
    checkpoint = torch.load(save_dir / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Évaluation sur le test set
    test_metrics = validate(model, test_loader, criterion, device)

    print(f"\nRésultats sur le test set:")
    print(f"  Style  - Top-1: {test_metrics['style_top1']:.2%}, Top-5: {test_metrics['style_top5']:.2%}")
    print(f"  Artist - Top-1: {test_metrics['artist_top1']:.2%}, Top-5: {test_metrics['artist_top5']:.2%}")

    # Sauvegarde des résultats finaux
    results = {
        "test_metrics": test_metrics,
        "best_val_acc": best_val_acc,
        "history": history,
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Fermeture du writer TensorBoard
    writer.close()

    print(f"\nEntraînement terminé!")
    print(f"Résultats sauvegardés dans: {save_dir}")
    print(f"TensorBoard: tensorboard --logdir {save_dir / 'tensorboard'}")


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraînement WikiArt Classifier")

    # Arguments principaux
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Taille du batch")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"],
                        help="Nombre de workers pour le DataLoader")

    # Phases d'entraînement
    parser.add_argument("--phase1-epochs", type=int, default=DEFAULT_CONFIG["phase1_epochs"],
                        help="Nombre d'epochs pour la phase 1")
    parser.add_argument("--phase2-epochs", type=int, default=DEFAULT_CONFIG["phase2_epochs"],
                        help="Nombre d'epochs pour la phase 2")

    # Learning rates
    parser.add_argument("--phase1-lr", type=float, default=DEFAULT_CONFIG["phase1_lr"],
                        help="Learning rate phase 1")
    parser.add_argument("--phase2-lr-backbone", type=float, default=DEFAULT_CONFIG["phase2_lr_backbone"],
                        help="Learning rate backbone phase 2")
    parser.add_argument("--phase2-lr-heads", type=float, default=DEFAULT_CONFIG["phase2_lr_heads"],
                        help="Learning rate heads phase 2")

    # Backbone
    parser.add_argument("--backbone-type", type=str, default=DEFAULT_CONFIG["backbone_type"],
                        choices=["timm", "clip"],
                        help="Type de backbone: 'timm' (ImageNet) ou 'clip' (recommandé pour l'art)")

    # Autres options
    parser.add_argument("--no-amp", action="store_true",
                        help="Désactiver mixed precision")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_CONFIG["save_dir"],
                        help="Dossier de sauvegarde")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Mise à jour de la config avec les arguments
    config = DEFAULT_CONFIG.copy()
    config.update({
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "phase1_epochs": args.phase1_epochs,
        "phase2_epochs": args.phase2_epochs,
        "phase1_lr": args.phase1_lr,
        "phase2_lr_backbone": args.phase2_lr_backbone,
        "phase2_lr_heads": args.phase2_lr_heads,
        "backbone_type": args.backbone_type,
        "use_amp": not args.no_amp,
        "save_dir": args.save_dir,
    })

    # Lancement de l'entraînement
    train(config)