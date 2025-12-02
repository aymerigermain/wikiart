# Interface Web WikiArt

Interface web moderne et design pour tester les modèles de classification WikiArt.

## 🎨 Fonctionnalités

- **Classification de Style** : Top 5 des styles artistiques prédits avec pourcentages de confiance
- **Classification d'Artiste** : Top 5 des artistes prédits avec pourcentages de confiance
- **Description d'Œuvre** : Génération automatique de descriptions via LLaVA (optionnel)
- **Interface Moderne** : Design responsive avec thème sombre et jauges visuelles
- **Drag & Drop** : Import d'images par glisser-déposer ou sélection

## 📋 Prérequis

- Python 3.8+
- Un checkpoint de modèle entraîné dans `../checkpoints/`
- GPU recommandé (mais peut fonctionner sur CPU)

## 🚀 Installation

1. Installer les dépendances :
```bash
cd interface/
pip install -r requirements.txt
```

2. S'assurer qu'un checkpoint existe dans le dossier parent :
```bash
ls ../checkpoints/
```

## 🎯 Utilisation

### Lancement basique (avec LLaVA)

```bash
python app.py
```

L'interface sera accessible sur [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Options de lancement

```bash
# Sans LLaVA (plus rapide, pas de descriptions)
python app.py --no-llava

# Changer le port
python app.py --port 8080

# Rendre accessible sur le réseau local
python app.py --host 0.0.0.0

# Mode debug (redémarrage automatique)
python app.py --debug
```

### Utilisation de l'interface

1. **Ouvrir** l'interface dans un navigateur
2. **Importer** une image (cliquer ou glisser-déposer)
3. **Cocher/décocher** l'option "Inclure la description" selon vos besoins
4. **Cliquer** sur "Analyser l'œuvre"
5. **Consulter** les résultats avec les jauges de confiance

## 📁 Structure

```
interface/
├── app.py                  # Serveur Flask
├── inference.py            # Module d'inférence
├── requirements.txt        # Dépendances
├── README.md              # Documentation
├── templates/
│   └── index.html         # Template HTML
├── static/
│   ├── css/
│   │   └── style.css      # Styles CSS
│   └── js/
│       └── script.js      # JavaScript
└── uploads/               # Dossier temporaire pour les uploads
```

## 🎨 Interface

L'interface présente :
- **Header** : Titre et description
- **Zone d'upload** : Drag & drop ou sélection de fichier
- **Aperçu** : Prévisualisation de l'image
- **Options** : Checkbox pour inclure/exclure la description
- **Résultats** :
  - Carte "Style Artistique" avec top 5 et jauges
  - Carte "Artiste" avec top 5 et jauges
  - Carte "Description" (si activée)
- **Actions** : Bouton pour analyser une autre image

## ⚙️ Configuration

### Modifier le checkpoint par défaut

Éditez [inference.py](inference.py) ligne ~95 pour spécifier un checkpoint :

```python
inference_system = WikiArtInference(
    checkpoint_path="../checkpoints/votre_modele.pt",
    load_llava=True
)
```

### Désactiver LLaVA par défaut

Modifiez [app.py](app.py) ligne ~140 :

```python
init_inference(load_llava=False)
```

### Changer le nombre de prédictions

Éditez [app.py](app.py) ligne ~77 :

```python
top_k = request.form.get('top_k', 5, type=int)  # Changer 5 par le nombre souhaité
```

## 🐛 Dépannage

### "Aucun checkpoint trouvé"
- Vérifier qu'un fichier `.pt` existe dans `../checkpoints/`
- Ou spécifier un chemin explicite dans `inference.py`

### "LLaVA n'est pas disponible"
- Lancer avec `--no-llava` pour désactiver les descriptions
- Ou installer les dépendances LLaVA : `pip install transformers accelerate bitsandbytes`

### Erreur mémoire GPU
- Utiliser `--no-llava` pour réduire l'utilisation mémoire
- Ou fermer d'autres programmes utilisant le GPU

### L'interface ne charge pas
- Vérifier que le port 5000 n'est pas déjà utilisé
- Essayer un autre port : `python app.py --port 8080`

## 🔧 Développement

### Modifier les styles

Éditez [static/css/style.css](static/css/style.css) pour personnaliser l'apparence.

### Modifier la logique client

Éditez [static/js/script.js](static/js/script.js) pour modifier les interactions.

### Ajouter des endpoints

Éditez [app.py](app.py) et ajoutez de nouvelles routes Flask.

## 📊 Performance

- **Temps de chargement initial** : 30-60 secondes (chargement des modèles)
- **Temps d'inférence sans LLaVA** : ~1-2 secondes
- **Temps d'inférence avec LLaVA** : ~5-10 secondes
- **Mémoire GPU** :
  - Sans LLaVA : ~2-3 GB
  - Avec LLaVA (4-bit) : ~5-7 GB

## 🎓 Technologies

- **Backend** : Flask (Python)
- **Frontend** : HTML5, CSS3, JavaScript (Vanilla)
- **ML** : PyTorch, Transformers, timm
- **Modèles** :
  - Vision Transformer (ViT) pour classification
  - LLaVA-NeXT pour descriptions

## 📝 Notes

- Les images uploadées sont temporairement stockées dans `uploads/`
- Format d'images supportés : PNG, JPG, JPEG, WEBP, BMP
- Taille maximale : 16 MB
- Le serveur doit rester actif pendant l'utilisation

## 🤝 Contribution

Pour améliorer l'interface :
1. Modifier les fichiers dans `interface/`
2. Tester localement avec `python app.py --debug`
3. Soumettre les modifications

## 📄 Licence

Voir le fichier LICENSE du projet principal.
