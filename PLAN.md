# Plan du Projet WikiArt - Deep Learning

## Compréhension du Problème

**Objectif** : Générer automatiquement une description d'une œuvre d'art (genre, style, artiste)

**Dataset** : WikiArt - 80K images, 27 styles, 1119 artistes

**Contraintes** :
- GPU 24Go vRAM disponible
- Projet multimodal requis (vision + texte)
- Phase 1 : Classification → Phase 2 : Génération de description

---

## Architecture Globale

Approche **multi-tâche multimodale** avec un backbone partagé :

```
Image → Backbone ViT → [Head Style] → 27 classes
                    → [Head Artiste] → 1119 classes
                    → [Décodeur LLM] → Description textuelle
```

---

## Choix du Backbone

Avec 24Go vRAM, on peut utiliser des modèles conséquents :

| Modèle | vRAM estimée | Recommandation |
|--------|--------------|----------------|
| **ViT-B/16** | ~8Go | ✅ Phase 1 - Classification |
| **ViT-L/14** | ~16Go | ✅ Si besoin de plus de performance |
| **CLIP ViT-L/14** | ~12Go | ✅ Phase 2 - Déjà aligné vision-texte |
| **BLIP-2** | ~20Go | ✅ Phase 2 - Génération de description |

**Recommandation finale** :
- Phase 1 : `ViT-B/16` (timm) pour classification
- Phase 2 : `BLIP-2` ou `GIT-large` pour génération multimodale

---

## Phase 1 - Classification (Prioritaire)

### Architecture
```python
ViT-B/16 (frozen ou fine-tuned)
    ↓
Features [CLS] (768 dim)
    ↓
    ├── MLP Head Style → 27 classes (CrossEntropyLoss)
    └── MLP Head Artiste → 1119 classes (Focal Loss)
```

### Stratégie d'entraînement
1. **Freeze backbone** + entraîner heads (5-10 epochs)
2. **Unfreeze dernières couches** + fine-tuning (10-20 epochs)
3. **Learning rate** : 1e-4 (heads) / 1e-5 (backbone)

### Loss combinée
```python
L_total = L_style + λ * L_artiste
# λ = 0.5 (artiste plus difficile, moins de poids)
```

---

## Phase 2 - Génération Multimodale

### Option A : BLIP-2 (Recommandé)
```
Image → ViT → Q-Former → LLM (OPT/Flan-T5) → Description
```
- Pré-entraîné pour image captioning
- Fine-tuning sur WikiArt avec descriptions générées/annotées

### Option B : GIT (Generative Image-to-Text)
```
Image → ViT → Décodeur GPT-like → Description
```
- Plus simple à fine-tuner
- Moins puissant mais suffisant

### Option C : CLIP + GPT-2 (DIY)
```
Image → CLIP ViT → Projection → GPT-2 → Description
```
- Plus de contrôle
- Plus de travail d'implémentation

### Format de sortie attendu
```
"Style impressionniste, œuvre de Claude Monet, représentant un paysage
maritime avec des reflets de lumière caractéristiques du mouvement."
```

---

## Métriques d'Évaluation

### Phase 1 - Classification

| Métrique | Description | Cible |
|----------|-------------|-------|
| **Top-1 Accuracy** | Prédiction exacte | >70% style, >40% artiste |
| **Top-5 Accuracy** | Bonne classe dans les 5 premiers | >90% style, >70% artiste |
| **F1-Score Macro** | Performance équilibrée par classe | >0.65 |
| **Confusion Matrix** | Visualiser les erreurs style similaires | - |

### Phase 2 - Génération

| Métrique | Description | Usage |
|----------|-------------|-------|
| **BLEU** | N-gram overlap avec référence | Fluence |
| **ROUGE-L** | Longest common subsequence | Couverture |
| **CIDEr** | Consensus-based scoring | Captioning standard |
| **CLIPScore** | Alignement image-texte | Pertinence sémantique |
| **Évaluation humaine** | Qualité perçue | Gold standard |

### Pourquoi ces métriques ?

- **Top-1** : La métrique de base, mais sévère pour 1119 artistes
- **Top-5** : Plus réaliste - si Monet est dans le top 5, c'est déjà bien
- **F1 Macro** : Évite que le modèle ignore les classes rares
- **CLIPScore** : Mesure si la description "correspond" visuellement à l'image

---

## Augmentations Spécifiques à l'Art

```python
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.3),  # Prudent pour portraits
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

---

## Gestion des Défis

| Défi | Solution |
|------|----------|
| **Déséquilibre artistes** (1119 classes) | Focal Loss (γ=2) + Sampling stratifié |
| **Similarité inter-styles** | Label Smoothing (0.1) + Hierarchical softmax |
| **80K images = dataset moyen** | Transfer learning agressif, data augmentation |
| **Génération de descriptions** | Templates + fine-tuning BLIP-2 |

---

## Pipeline Technique (PyTorch)

### Dépendances principales
```python
torch >= 2.0
torchvision
timm                 # Modèles ViT pré-entraînés
transformers         # BLIP-2, GPT-2, T5
datasets             # Gestion données HuggingFace
albumentations       # Augmentations avancées (optionnel)
wandb                # Tracking expériences
```

### Structure du projet
```
adl/
├── data.py          # Chargement WikiArt
├── dataset.py       # Dataset PyTorch custom
├── models/
│   ├── classifier.py    # Phase 1 - ViT + heads
│   └── captioner.py     # Phase 2 - BLIP-2/GIT
├── train.py         # Boucle d'entraînement
├── evaluate.py      # Métriques
├── config.py        # Hyperparamètres
└── PLAN.md
```

---

## Planning d'Implémentation

### Étape 1 : Data Pipeline
- [ ] Explorer la structure du dataset WikiArt
- [ ] Créer Dataset PyTorch avec labels style/artiste
- [ ] Implémenter train/val/test split stratifié
- [ ] Visualiser distribution des classes

### Étape 2 : Classification (Phase 1)
- [ ] Implémenter modèle ViT + dual heads
- [ ] Entraîner sur style uniquement (baseline)
- [ ] Ajouter classification artiste (multi-task)
- [ ] Optimiser hyperparamètres

### Étape 3 : Génération (Phase 2)
- [ ] Préparer données textuelles (templates ou annotations)
- [ ] Fine-tuner BLIP-2 sur WikiArt
- [ ] Évaluer qualité des descriptions
- [ ] Combiner avec classification

### Étape 4 : Finalisation
- [ ] Interface de démonstration
- [ ] Documentation
- [ ] Rapport final