#!/bin/bash
# Setup per riprendere refactoring Esercitazione 3
# Creato: 28 Novembre 2024

set -e

echo "=================================================="
echo "Setup Refactoring Esercitazione 3"
echo "=================================================="
echo ""

# Variabili
PROJECT_ROOT="/home/brusc/Projects/bioimmagini_positano"
ES3_DIR="esercitazioni/esercitazioni_python/es_3__23_03_2022_clustering"
VENV_PATH="esercitazioni/esercitazioni_python/venv"

cd "$PROJECT_ROOT"

# 1. Crea backup
echo "[1/6] Creando backup..."
if [ ! -d "${ES3_DIR}_BACKUP_20241128" ]; then
    cp -r "$ES3_DIR" "${ES3_DIR}_BACKUP_20241128"
    echo "  ✓ Backup creato: ${ES3_DIR}_BACKUP_20241128"
else
    echo "  ⚠ Backup già esistente, skip"
fi

# 2. Crea branch Git
echo ""
echo "[2/6] Creando branch Git..."
git checkout -b refactor/es3-modernization 2>/dev/null || echo "  ⚠ Branch già esistente"
echo "  ✓ Branch: refactor/es3-modernization"

# 3. Attiva virtual environment
echo ""
echo "[3/6] Attivando virtual environment..."
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "  ✓ Venv attivato: $VENV_PATH"
else
    echo "  ❌ Venv non trovato in $VENV_PATH"
    exit 1
fi

# 4. Installa dipendenze dev
echo ""
echo "[4/6] Installando dipendenze development..."
pip install -q pytest pytest-cov pytest-benchmark hypothesis pydantic ruff mypy 2>&1 | grep -v "Requirement already satisfied" || true
echo "  ✓ Dipendenze installate"

# 5. Crea directory strutturali
echo ""
echo "[5/6] Creando directory strutturali..."
mkdir -p "$ES3_DIR/tests"
mkdir -p "$ES3_DIR/results/baseline"
echo "  ✓ Directory create"

# 6. Esegui baseline test (opzionale)
echo ""
echo "[6/6] Baseline test (opzionale)..."
read -p "Vuoi eseguire gli script originali per baseline? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$ES3_DIR"
    echo "  Running kmeans_segmentation.py..."
    python src/kmeans_segmentation.py --output-dir results/baseline 2>&1 | tail -5
    echo "  ✓ Baseline salvata in results/baseline/"
fi

echo ""
echo "=================================================="
echo "Setup completato!"
echo "=================================================="
echo ""
echo "Prossimi passi:"
echo "1. Apri Claude Code"
echo "2. Scrivi: 'Leggi docs/SESSION_PLAN_ES3_REFACTORING.md e iniziamo dalla Fase 1'"
echo ""
echo "File utili:"
echo "  - Piano:  docs/SESSION_PLAN_ES3_REFACTORING.md"
echo "  - Backup: ${ES3_DIR}_BACKUP_20241128/"
echo "  - Branch: refactor/es3-modernization"
echo ""
