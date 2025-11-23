#!/bin/bash

# Couleurs pour les logs
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
NC="\033[0m" # No Color

echo -e "${BLUE}=== Création du virtualenv ===${NC}"
/opt/dce/dce_venv.sh /mounts/datasets/venvs/torch-2.7.1 $TMPDIR/venv

echo -e "${BLUE}=== Activation du virtualenv ===${NC}"
source $TMPDIR/venv/bin/activate

echo -e "${BLUE}=== Installation des dépendances Python ===${NC}"
pip install -r requirements.txt

echo -e "${BLUE}=== Préparation des dossiers pour le dataset ===${NC}"
mkdir -p $TMPDIR/datasets/steubk/wikiart/versions/

echo -e "${BLUE}=== Copie du dataset avec progression ===${NC}"
rsync -ah --info=progress2 ~/.cache/kagglehub/datasets/steubk/wikiart/versions/1/ \
      $TMPDIR/datasets/steubk/wikiart/versions/

echo -e "${GREEN}✅ Setup terminé avec succès !${NC}"
