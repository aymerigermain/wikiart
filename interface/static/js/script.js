// ============================================================================
// VARIABLES GLOBALES
// ============================================================================

let selectedFile = null;

// ============================================================================
// ÉLÉMENTS DOM
// ============================================================================

const imageInput = document.getElementById('imageInput');
const uploadBtn = document.getElementById('uploadBtn');
const uploadArea = document.getElementById('uploadArea');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const changeImageBtn = document.getElementById('changeImageBtn');
const options = document.getElementById('options');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const resultImg = document.getElementById('resultImg');
const stylePredictions = document.getElementById('stylePredictions');
const artistPredictions = document.getElementById('artistPredictions');
const descriptionCard = document.getElementById('descriptionCard');
const descriptionText = document.getElementById('descriptionText');
const retryBtn = document.getElementById('retryBtn');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const includeDescription = document.getElementById('includeDescription');

// ============================================================================
// EVENT LISTENERS
// ============================================================================

// Bouton upload
uploadBtn.addEventListener('click', () => {
    imageInput.click();
});

// Zone de drop
uploadArea.addEventListener('click', () => {
    imageInput.click();
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// Sélection de fichier
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Changer d'image
changeImageBtn.addEventListener('click', () => {
    imageInput.click();
});

// Analyser
analyzeBtn.addEventListener('click', () => {
    analyzeImage();
});

// Réessayer
retryBtn.addEventListener('click', () => {
    resetInterface();
});

// ============================================================================
// FONCTIONS
// ============================================================================

/**
 * Gère la sélection d'un fichier
 */
function handleFileSelect(file) {
    // Vérifier le type de fichier
    if (!file.type.startsWith('image/')) {
        showError('Veuillez sélectionner une image valide');
        return;
    }

    // Vérifier la taille (max 16 MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('L\'image est trop grande (max 16 MB)');
        return;
    }

    selectedFile = file;

    // Afficher l'aperçu
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadArea.style.display = 'none';
        imagePreview.style.display = 'block';
        options.style.display = 'block';
        analyzeBtn.style.display = 'block';
        hideError();
    };
    reader.readAsDataURL(file);
}

/**
 * Analyse l'image
 */
async function analyzeImage() {
    if (!selectedFile) {
        showError('Aucune image sélectionnée');
        return;
    }

    // Masquer les éléments
    imagePreview.style.display = 'none';
    options.style.display = 'none';
    analyzeBtn.style.display = 'none';
    resultsSection.style.display = 'none';
    hideError();

    // Afficher le loading
    loading.style.display = 'block';

    // Préparer les données
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('top_k', 5);
    formData.append('include_description', includeDescription.checked);

    try {
        // Envoyer la requête
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Erreur lors de l\'analyse');
        }

        // Afficher les résultats
        displayResults(data);

    } catch (error) {
        console.error('Erreur:', error);
        showError(error.message);
        loading.style.display = 'none';
        imagePreview.style.display = 'block';
        options.style.display = 'block';
        analyzeBtn.style.display = 'block';
    }
}

/**
 * Affiche les résultats
 */
function displayResults(data) {
    loading.style.display = 'none';

    // Vérifier les erreurs
    if (data.error) {
        showError(data.error);
        imagePreview.style.display = 'block';
        options.style.display = 'block';
        analyzeBtn.style.display = 'block';
        return;
    }

    // Afficher l'image analysée
    resultImg.src = previewImg.src;

    // Afficher la description si disponible (AVANT les prédictions)
    if (data.description) {
        descriptionText.textContent = data.description;
        descriptionCard.style.display = 'block';
    } else {
        descriptionCard.style.display = 'none';
    }

    // Afficher les prédictions de style
    if (data.style_predictions) {
        stylePredictions.innerHTML = '';
        data.style_predictions.forEach((pred) => {
            const item = createPredictionItem(pred.label, pred.confidence);
            stylePredictions.appendChild(item);
        });
    }

    // Afficher les prédictions d'artiste
    if (data.artist_predictions) {
        artistPredictions.innerHTML = '';
        data.artist_predictions.forEach((pred) => {
            const item = createPredictionItem(pred.label, pred.confidence);
            artistPredictions.appendChild(item);
        });
    }

    // Afficher la section résultats
    resultsSection.style.display = 'block';
}

/**
 * Crée un élément de prédiction avec jauge
 */
function createPredictionItem(label, confidence) {
    const item = document.createElement('div');
    item.className = 'prediction-item';

    const header = document.createElement('div');
    header.className = 'prediction-header';

    const labelSpan = document.createElement('span');
    labelSpan.className = 'prediction-label';
    labelSpan.textContent = label;

    const confidenceSpan = document.createElement('span');
    confidenceSpan.className = 'prediction-confidence';
    confidenceSpan.textContent = `${confidence.toFixed(1)}%`;

    header.appendChild(labelSpan);
    header.appendChild(confidenceSpan);

    const bar = document.createElement('div');
    bar.className = 'prediction-bar';

    const fill = document.createElement('div');
    fill.className = 'prediction-fill';
    fill.style.width = `${confidence}%`;

    bar.appendChild(fill);

    item.appendChild(header);
    item.appendChild(bar);

    return item;
}

/**
 * Affiche un message d'erreur
 */
function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
}

/**
 * Masque le message d'erreur
 */
function hideError() {
    errorMessage.style.display = 'none';
}

/**
 * Réinitialise l'interface
 */
function resetInterface() {
    selectedFile = null;
    imageInput.value = '';
    previewImg.src = '';

    uploadArea.style.display = 'block';
    imagePreview.style.display = 'none';
    options.style.display = 'none';
    analyzeBtn.style.display = 'none';
    loading.style.display = 'none';
    resultsSection.style.display = 'none';
    hideError();
}

// ============================================================================
// INITIALISATION
// ============================================================================

console.log('Interface WikiArt initialisée');
