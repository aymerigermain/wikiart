"""
train_style.py - Script d'entraînement pour le classificateur de style WikiArt

Entraînement simple-tâche du classificateur ViT avec:
- Phase 1: Backbone gelé, entraînement de la tête uniquement
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

from data import download_or_use_cached, discover_structure, build_label_mappings, filter_artists_by_count
from dataset_style import create_splits, create_dataloaders, get_class_weights
from models.style_classifier import StyleClassifier, StyleLoss


# ============================================================================
# CONFIGURATION PAR DÉFAUT
# ============================================================================

DEFAULT_CONFIG = {
    # Données
    "batch_size": 192,
    "num_workers": 8,
    "image_size": 224,

    # Modèle
    "backbone": "ViT-B-16",
    "backbone_type": "clip",  # "timm" ou "clip"
    "dropout": 0.3,

    # Entraînement Phase 1 (backbone gelé)
    "phase1_epochs": 15,
    "phase1_lr": 5e-4,

    # Entraînement Phase 2 (fine-tuning)
    "phase2_epochs": 25,
    "phase2_lr_backbone": 5e-6,
    "phase2_lr_head": 5e-5,
    "unfreeze_layers": 4,

    # Optimisation
    "weight_decay": 0.05,
    "warmup_epochs": 3,
    "use_amp": True,

    # Loss
    "use_focal_loss": True,
    "focal_gamma": 2.0,
    "label_smoothing": 0.1,

    # Early Stopping
    "early_stopping_patience": 4,
    "early_stopping_min_delta": 0.001,

    # Sauvegarde
    "save_dir": "checkpoints_style",
    "save_every": 5,
}


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def format_time(seconds: float) -> str:
    """Formate un temps en secondes en format lisible."""
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
        Dictionnaire avec top1_acc, topk_acc
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
    """Suivi des métriques pendant l'entraînement."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Réinitialise les compteurs."""
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: dict, batch_size: int = 1):
        """Met à jour les métriques avec un nouveau batch."""
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
    """Early Stopping pour éviter l'overfitting."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Vérifie si l'entraînement doit s'arrêter."""
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
        """Réinitialise l'early stopping."""
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
    """Entraîne le modèle pour une epoch."""
    model.train()
    tracker = MetricTracker()

    pbar = tqdm(dataloader, desc="Train", leave=False)

    for batch in pbar:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Calcul des métriques
        with torch.no_grad():
            batch_metrics = compute_metrics(outputs, targets)

        metrics = {
            "loss": loss.item(),
            "top1_acc": batch_metrics["top1_acc"],
            "top5_acc": batch_metrics["top5_acc"],
        }
        tracker.update(metrics, batch_size)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_metrics['top1_acc']:.2%}",
        })

    return tracker.get_averages()


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Évalue le modèle sur le set de validation."""
    model.eval()
    tracker = MetricTracker()

    pbar = tqdm(dataloader, desc="Val", leave=False)

    for batch in pbar:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)
        batch_size = images.size(0)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_metrics = compute_metrics(outputs, targets)

        metrics = {
            "loss": loss.item(),
            "top1_acc": batch_metrics["top1_acc"],
            "top5_acc": batch_metrics["top5_acc"],
        }
        tracker.update(metrics, batch_size)

    return tracker.get_averages()


# ============================================================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# ============================================================================

def train(config: dict) -> None:
    """Fonction principale d'entraînement."""
    print("=" * 60)
    print("ENTRAÎNEMENT WIKIART STYLE CLASSIFIER")
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

    style2idx, idx2style, _, _ = build_label_mappings(structure)

    num_classes = len(structure["styles"])
    print(f"Styles: {num_classes}")

    # Création des splits
    train_samples, val_samples, test_samples = create_splits(
        structure, style2idx
    )
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Création des DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples,
        val_samples,
        test_samples,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        use_weighted_sampler=True,
        num_classes=num_classes,
        persistent_workers=True,
        backbone_type=config["backbone_type"],
    )

    # Calcul des poids de classe pour la loss
    class_weights = get_class_weights(train_samples, num_classes)

    # ===== MODÈLE =====
    print("\n" + "-" * 40)
    print("Création du modèle...")
    print("-" * 40)

    model = StyleClassifier(
        num_classes=num_classes,
        backbone_name=config["backbone"],
        backbone_type=config["backbone_type"],
        pretrained=True,
        freeze_backbone=True,
        dropout=config["dropout"],
    )
    print(f"Backbone: {config['backbone']} (type: {config['backbone_type']})")
    model = model.to(device)

    # ===== LOSS =====
    criterion = StyleLoss(
        use_focal_loss=config["use_focal_loss"],
        focal_gamma=config["focal_gamma"],
        label_smoothing=config.get("label_smoothing", 0.0),
        class_weights=class_weights.to(device),
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
    # PHASE 1: ENTRAÎNEMENT DE LA TÊTE (BACKBONE GELÉ)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Entraînement de la tête (backbone gelé)")
    print("=" * 60)

    training_start_time = time.time()

    early_stopping = EarlyStopping(
        patience=config.get("early_stopping_patience", 5),
        min_delta=config.get("early_stopping_min_delta", 0.001),
        mode="max",
    )

    optimizer = AdamW(
        model.get_trainable_params()["head"],
        lr=config["phase1_lr"],
        weight_decay=config["weight_decay"],
    )

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

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config["use_amp"]
        )

        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time

        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Top-1: {train_metrics['top1_acc']:.2%}, "
              f"Top-5: {train_metrics['top5_acc']:.2%}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Top-1: {val_metrics['top1_acc']:.2%}, "
              f"Top-5: {val_metrics['top5_acc']:.2%}")
        print(f"Temps - Epoch: {format_time(epoch_time)}, Total: {format_time(total_elapsed)}")

        # TensorBoard logging
        global_epoch = epoch
        writer.add_scalars("Loss", {
            "train": train_metrics["loss"],
            "val": val_metrics["loss"],
        }, global_epoch)
        writer.add_scalars("Accuracy/Top1", {
            "train": train_metrics["top1_acc"],
            "val": val_metrics["top1_acc"],
        }, global_epoch)
        writer.add_scalars("Accuracy/Top5", {
            "train": train_metrics["top5_acc"],
            "val": val_metrics["top5_acc"],
        }, global_epoch)
        writer.add_scalar("Time/Epoch_seconds", epoch_time, global_epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], global_epoch)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Sauvegarde du meilleur modèle
        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
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
        if early_stopping(val_metrics["top1_acc"]):
            print(f"\n  Early stopping déclenché après {epoch + 1} epochs (Phase 1)")
            break

    # =========================================================================
    # PHASE 2: FINE-TUNING DU BACKBONE
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning du backbone")
    print("=" * 60)

    early_stopping.reset()

    model.unfreeze_backbone(unfreeze_layers=config["unfreeze_layers"])

    param_groups = model.get_trainable_params()
    optimizer = AdamW([
        {"params": param_groups["backbone"], "lr": config["phase2_lr_backbone"]},
        {"params": param_groups["head"], "lr": config["phase2_lr_head"]},
    ], weight_decay=config["weight_decay"])

    scheduler = CosineAnnealingLR(optimizer, T_max=config["phase2_epochs"])

    for epoch in range(config["phase2_epochs"]):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{config['phase2_epochs']} (Phase 2)")
        print("-" * 40)

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config["use_amp"]
        )

        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time

        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Top-1: {train_metrics['top1_acc']:.2%}, "
              f"Top-5: {train_metrics['top5_acc']:.2%}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Top-1: {val_metrics['top1_acc']:.2%}, "
              f"Top-5: {val_metrics['top5_acc']:.2%}")
        print(f"Temps - Epoch: {format_time(epoch_time)}, Total: {format_time(total_elapsed)}")

        # TensorBoard logging
        global_epoch = config["phase1_epochs"] + epoch
        writer.add_scalars("Loss", {
            "train": train_metrics["loss"],
            "val": val_metrics["loss"],
        }, global_epoch)
        writer.add_scalars("Accuracy/Top1", {
            "train": train_metrics["top1_acc"],
            "val": val_metrics["top1_acc"],
        }, global_epoch)
        writer.add_scalars("Accuracy/Top5", {
            "train": train_metrics["top5_acc"],
            "val": val_metrics["top5_acc"],
        }, global_epoch)
        writer.add_scalar("Time/Epoch_seconds", epoch_time, global_epoch)
        writer.add_scalar("LearningRate/Backbone", optimizer.param_groups[0]["lr"], global_epoch)
        writer.add_scalar("LearningRate/Head", optimizer.param_groups[1]["lr"], global_epoch)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            torch.save({
                "epoch": config["phase1_epochs"] + epoch,
                "phase": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, save_dir / "best_model.pt")
            print(f"  -> Nouveau meilleur modèle sauvegardé (acc: {best_val_acc:.2%})")

        if (epoch + 1) % config["save_every"] == 0:
            torch.save({
                "epoch": config["phase1_epochs"] + epoch,
                "phase": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, save_dir / f"checkpoint_epoch_{config['phase1_epochs'] + epoch + 1}.pt")

        if early_stopping(val_metrics["top1_acc"]):
            print(f"\n  Early stopping déclenché après {epoch + 1} epochs (Phase 2)")
            break

    # =========================================================================
    # ÉVALUATION FINALE
    # =========================================================================
    print("\n" + "=" * 60)
    print("ÉVALUATION FINALE SUR LE SET DE TEST")
    print("=" * 60)

    checkpoint = torch.load(save_dir / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = validate(model, test_loader, criterion, device)

    print(f"\nRésultats sur le test set:")
    print(f"  Top-1: {test_metrics['top1_acc']:.2%}")
    print(f"  Top-5: {test_metrics['top5_acc']:.2%}")
    print(f"  Loss: {test_metrics['loss']:.4f}")

    results = {
        "test_metrics": test_metrics,
        "best_val_acc": best_val_acc,
        "history": history,
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    writer.close()

    print(f"\nEntraînement terminé!")
    print(f"Résultats sauvegardés dans: {save_dir}")


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraînement WikiArt Style Classifier")

    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Taille du batch")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"],
                        help="Nombre de workers pour le DataLoader")
    parser.add_argument("--phase1-epochs", type=int, default=DEFAULT_CONFIG["phase1_epochs"],
                        help="Nombre d'epochs pour la phase 1")
    parser.add_argument("--phase2-epochs", type=int, default=DEFAULT_CONFIG["phase2_epochs"],
                        help="Nombre d'epochs pour la phase 2")
    parser.add_argument("--phase1-lr", type=float, default=DEFAULT_CONFIG["phase1_lr"],
                        help="Learning rate phase 1")
    parser.add_argument("--phase2-lr-backbone", type=float, default=DEFAULT_CONFIG["phase2_lr_backbone"],
                        help="Learning rate backbone phase 2")
    parser.add_argument("--phase2-lr-head", type=float, default=DEFAULT_CONFIG["phase2_lr_head"],
                        help="Learning rate head phase 2")
    parser.add_argument("--backbone-type", type=str, default=DEFAULT_CONFIG["backbone_type"],
                        choices=["timm", "clip"],
                        help="Type de backbone: 'timm' ou 'clip'")
    parser.add_argument("--no-amp", action="store_true",
                        help="Désactiver mixed precision")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_CONFIG["save_dir"],
                        help="Dossier de sauvegarde")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "phase1_epochs": args.phase1_epochs,
        "phase2_epochs": args.phase2_epochs,
        "phase1_lr": args.phase1_lr,
        "phase2_lr_backbone": args.phase2_lr_backbone,
        "phase2_lr_head": args.phase2_lr_head,
        "backbone_type": args.backbone_type,
        "use_amp": not args.no_amp,
        "save_dir": args.save_dir,
    })

    train(config)
