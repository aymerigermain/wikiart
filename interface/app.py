"""
app.py - Serveur Flask pour l'interface web WikiArt

Serveur web minimal et moderne pour tester les modèles WikiArt
"""

import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from inference import WikiArtInference


# ============================================================================
# CONFIGURATION
# ============================================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# Créer le dossier d'upload s'il n'existe pas
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Initialiser le système d'inférence (une seule fois au démarrage)
print("🚀 Initialisation du système d'inférence...")
inference_system = None


def allowed_file(filename):
    """Vérifie si l'extension du fichier est autorisée."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Page d'accueil."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint pour faire une prédiction sur une image uploadée.

    Returns:
        JSON avec les prédictions (styles, artistes, description)
    """
    # Vérifier qu'un fichier a été envoyé
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image fournie'}), 400

    file = request.files['image']

    # Vérifier que le fichier a un nom
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'}), 400

    # Vérifier l'extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Format d\'image non supporté'}), 400

    # Sauvegarder le fichier
    filename = secure_filename(file.filename)
    filepath = app.config['UPLOAD_FOLDER'] / filename
    file.save(str(filepath))

    try:
        # Obtenir les paramètres
        top_k = request.form.get('top_k', 5, type=int)
        include_description = request.form.get('include_description', 'true').lower() == 'true'

        # Faire les prédictions
        if include_description:
            result = inference_system.predict_all(str(filepath), top_k=top_k)
        else:
            result = inference_system.predict_style_artist(str(filepath), top_k=top_k)

        # Nettoyer le fichier uploadé (optionnel)
        # filepath.unlink()

        return jsonify(result)

    except Exception as e:
        # Nettoyer en cas d'erreur
        if filepath.exists():
            filepath.unlink()
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Endpoint de santé pour vérifier que le serveur fonctionne."""
    return jsonify({
        'status': 'ok',
        'inference_loaded': inference_system is not None,
        'device': str(inference_system.device) if inference_system else 'N/A',
    })


# ============================================================================
# MAIN
# ============================================================================

def init_inference(load_llava: bool = True):
    """Initialise le système d'inférence."""
    global inference_system
    inference_system = WikiArtInference(load_llava=load_llava)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Serveur web WikiArt")
    parser.add_argument('--port', type=int, default=5000, help='Port du serveur')
    parser.add_argument('--host', default='127.0.0.1', help='Hôte du serveur')
    parser.add_argument('--no-llava', action='store_true', help='Ne pas charger LLaVA')
    parser.add_argument('--debug', action='store_true', help='Mode debug')

    args = parser.parse_args()

    # Initialiser l'inférence
    init_inference(load_llava=not args.no_llava)

    # Lancer le serveur
    print(f"\n🌐 Serveur lancé sur http://{args.host}:{args.port}")
    print(f"   Appuyez sur Ctrl+C pour arrêter\n")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )
