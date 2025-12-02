# WikiArt Style Classifier

Classification simple-tâche pour la prédiction de styles artistiques uniquement.

## 📁 Nouveaux Fichiers

Cette implémentation ajoute les fichiers suivants sans modifier les fichiers existants :

```
adl/
├── models/
│   └── style_classifier.py     # Modèle de classification de style
├── dataset_style.py             # Dataset simplifié (style uniquement)
├── train_style.py               # Script d'entraînement
├── test_style_model.py          # Script d'inférence
├── checkpoints_style/           # Checkpoints du modèle style
└── README_STYLE.md              # Cette documentation
```

## 🎯 Architecture

### Modèle

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
  (768 dim)
         │
         ▼
  ┌─────────────┐
  │  MLP Head   │  (768 → 1536 → 22)
  └──────┬──────┘
         │
         ▼
    22 classes
```

### Différences avec le classificateur multi-tâche

| Aspect | Multi-tâche | Style seul |
|--------|-------------|------------|
| **Tâches** | Style + Artiste | Style uniquement |
| **Têtes** | 2 têtes MLP | 1 tête MLP |
| **Loss** | Combinée pondérée | Simple (Focal/CE) |
| **Complexité** | Plus élevée | Simplifiée |
| **Performance** | Légèrement inférieure sur style | Optimisée pour style |
| **Entraînement** | ~6 heures | ~4 heures |

## 🚀 Installation

Utilise les mêmes dépendances que le projet principal :

```bash
pip install -r requirements.txt
```

## 📊 Entraînement

### Entraînement par défaut

```bash
python train_style.py
```

Configuration par défaut :
- **Backbone**: CLIP ViT-B/16
- **Batch size**: 192
- **Phase 1**: 15 epochs (backbone gelé)
- **Phase 2**: 25 epochs (fine-tuning)
- **Loss**: Focal Loss + Label Smoothing
- **Augmentation**: Strong (combat overfitting)

### Options d'entraînement

```bash
# Utiliser timm au lieu de CLIP
python train_style.py --backbone-type timm

# Ajuster les hyperparamètres
python train_style.py --batch-size 128 --phase1-epochs 10 --phase2-epochs 20

# Désactiver mixed precision (pour debugging)
python train_style.py --no-amp

# Changer le dossier de sauvegarde
python train_style.py --save-dir my_checkpoints
```

### Stratégie d'entraînement

**Phase 1** (15 epochs) :
- Backbone **gelé**
- Entraînement de la tête uniquement
- LR: 5e-4
- Warmup: 3 epochs
- Scheduler: CosineAnnealing

**Phase 2** (25 epochs) :
- Dégel des 4 dernières couches du backbone
- Fine-tuning progressif
- LR différentiels:
  - Backbone: 5e-6
  - Head: 5e-5
- Early stopping: patience=4

## 🧪 Test / Inférence

### Tester une image

```bash
# Mode interactif (choisir le checkpoint)
python test_style_model.py

# Tester une image spécifique
python test_style_model.py --image path/to/image.jpg

# Utiliser le meilleur modèle avec top-3
python test_style_model.py --checkpoint best --top-k 3
```

### Exemple de sortie

```
============================================================
📷 Image: monet_waterlilies.jpg
   Dimensions: 800x600
============================================================

🎨 STYLE:
----------------------------------------
  1. Impressionism                    87.3% █████████████████
  2. Post_Impressionism               8.2% █
  3. Pointillism                      2.1%
  4. Fauvism                          1.4%
  5. Expressionism                    0.7%
```

### Lister les checkpoints disponibles

```bash
python test_style_model.py --list-checkpoints
```

## 📈 Métriques attendues

Basé sur l'architecture et la configuration :

| Métrique | Cible | Note |
|----------|-------|------|
| **Top-1 Accuracy** | >75% | Amélioration vs multi-tâche |
| **Top-5 Accuracy** | >95% | Excellente couverture |
| **Loss (test)** | <0.4 | Avec label smoothing |

### Comparaison multi-tâche vs simple-tâche

| Modèle | Top-1 (style) | Top-5 (style) | Temps entraînement |
|--------|---------------|---------------|-------------------|
| Multi-tâche | 70-75% | 90-95% | ~6h |
| **Style seul** | **75-80%** | **95%+** | **~4h** |

**Avantages du style seul** :
- ✅ Focus complet sur la tâche de style
- ✅ Pas de compromis avec la tâche artiste
- ✅ Entraînement plus rapide
- ✅ Modèle plus léger (une seule tête)
- ✅ Plus simple à maintenir

## 🔧 Hyperparamètres

### Configuration complète

```python
DEFAULT_CONFIG = {
    # Données
    "batch_size": 192,          # Optimisé pour GPU 24GB
    "num_workers": 8,

    # Modèle
    "backbone": "ViT-B-16",
    "backbone_type": "clip",     # "timm" ou "clip"
    "dropout": 0.3,

    # Phase 1 (backbone gelé)
    "phase1_epochs": 15,
    "phase1_lr": 5e-4,

    # Phase 2 (fine-tuning)
    "phase2_epochs": 25,
    "phase2_lr_backbone": 5e-6,  # LR faible pour backbone
    "phase2_lr_head": 5e-5,      # LR plus élevé pour head
    "unfreeze_layers": 4,         # Nombre de couches dégelées

    # Optimisation
    "weight_decay": 0.05,
    "warmup_epochs": 3,
    "use_amp": True,              # Mixed precision

    # Loss
    "use_focal_loss": True,
    "focal_gamma": 2.0,
    "label_smoothing": 0.1,

    # Early Stopping
    "early_stopping_patience": 4,
    "early_stopping_min_delta": 0.001,
}
```

## 📦 Structure du code

### `models/style_classifier.py`

**Classes principales** :
- `StyleClassifier`: Modèle principal (ViT + MLP head)
- `StyleLoss`: Loss avec support Focal/CrossEntropy
- `FocalLoss`: Focal Loss pour déséquilibre de classes
- `LabelSmoothingFocalLoss`: Focal Loss + Label Smoothing
- `CLIPVisionBackbone`: Wrapper pour CLIP

**Fonctions utilitaires** :
- `create_style_classifier()`: Factory pour créer le modèle

### `dataset_style.py`

**Classes principales** :
- `WikiArtStyleDataset`: Dataset PyTorch (style uniquement)

**Fonctions utilitaires** :
- `get_transforms()`: Transforms train/val/test
- `create_splits()`: Splits stratifiés
- `create_dataloaders()`: DataLoaders avec weighted sampling
- `get_class_weights()`: Poids pour loss pondérée

### `train_style.py`

**Classes utilitaires** :
- `MetricTracker`: Suivi des métriques
- `EarlyStopping`: Arrêt anticipé

**Fonctions principales** :
- `train_one_epoch()`: Entraînement sur une epoch
- `validate()`: Validation
- `train()`: Boucle complète (phase 1 + phase 2)

## 🎨 Styles supportés

22 styles artistiques :

```
Abstract_Expressionism, Action_painting, Analytical_Cubism,
Art_Nouveau_Modern, Baroque, Color_Field_Painting,
Contemporary_Realism, Cubism, Early_Renaissance, Expressionism,
Fauvism, High_Renaissance, Impressionism, Mannerism_Late_Renaissance,
Minimalism, Naive_Art_Primitivism, New_Realism, Northern_Renaissance,
Pointillism, Pop_Art, Post_Impressionism, Realism
```

## 🔍 Monitoring

### TensorBoard

```bash
tensorboard --logdir checkpoints_style/TIMESTAMP/tensorboard
```

Métriques trackées :
- Loss (train/val)
- Top-1 Accuracy (train/val)
- Top-5 Accuracy (train/val)
- Learning Rate (head + backbone)
- Temps par epoch

### Fichiers sauvegardés

```
checkpoints_style/TIMESTAMP/
├── best_model.pt              # Meilleur modèle (val accuracy)
├── checkpoint_epoch_N.pt      # Checkpoints périodiques
├── config.json                # Configuration utilisée
├── results.json               # Résultats finaux
└── tensorboard/              # Logs TensorBoard
```

## 💡 Conseils d'utilisation

### Quand utiliser le modèle style seul ?

✅ **Utiliser si** :
- Focus exclusif sur la classification de style
- Budget computationnel limité
- Besoin de performances optimales sur style
- Déploiement avec contraintes mémoire

❌ **Utiliser multi-tâche si** :
- Besoin de prédire style ET artiste
- Volonté de partager les représentations
- Intérêt pour le learning multi-tâche

### Optimisations possibles

1. **Réduire batch size** si OOM:
   ```bash
   python train_style.py --batch-size 128
   ```

2. **Utiliser timm** si CLIP indisponible:
   ```bash
   python train_style.py --backbone-type timm
   ```

3. **Réduire epochs** pour test rapide:
   ```bash
   python train_style.py --phase1-epochs 5 --phase2-epochs 10
   ```

4. **Désactiver early stopping** (modifier `DEFAULT_CONFIG`):
   ```python
   "early_stopping_patience": 999,
   ```

## 🐛 Troubleshooting

### Erreur CUDA Out of Memory

```bash
# Réduire batch size
python train_style.py --batch-size 96

# Désactiver mixed precision
python train_style.py --no-amp
```

### Convergence lente

- Augmenter learning rate phase 1 : `--phase1-lr 1e-3`
- Réduire warmup : modifier `warmup_epochs` dans config
- Vérifier augmentation : peut-être trop forte

### Overfitting

- Augmenter dropout : modifier `dropout` dans config
- Augmenter weight_decay : `--weight-decay 0.1`
- Plus d'epochs en phase 1 avant fine-tuning

## 📝 Citation

Si tu utilises ce code, cite le projet WikiArt :

```
WikiArt Style Classifier
Based on Vision Transformer (ViT) with CLIP pretraining
Dataset: WikiArt (63K images, 22 styles)
```

## 🤝 Contribution

Pour améliorer le modèle :
1. Expérimenter avec d'autres backbones (ViT-L, Swin Transformer)
2. Tester différentes augmentations
3. Implémenter des techniques avancées (MixUp, CutMix)
4. Ajouter une validation croisée
5. Optimiser les hyperparamètres avec Optuna

## 📄 License

Même license que le projet principal.
