#!/bin/bash
# Quick activation script for the virtual environment
# Usage: source activate.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "Activating Python Virtual Environment"
echo "=========================================="
echo "Location: $SCRIPT_DIR/venv"
echo ""

source "$SCRIPT_DIR/venv/bin/activate"

echo "✓ Virtual environment activated!"
echo ""
echo "Python version: $(python --version)"
echo "pip version: $(pip --version)"
echo ""
echo "Installed packages:"
pip list | grep -E "(numpy|scipy|matplotlib|pydicom|jupyter|ipykernel|scikit-image)" | head -8
echo ""
echo "=========================================="
echo "Available exercises:"
echo "  - esercitazione_1/src/calcolo_sd.py"
echo "  - esercitazione_1/src/esempio_calcolo_sd.py"
echo "  - esercitazione_1/src/test_m_sd.py"
echo ""
echo "To deactivate: deactivate"
echo "=========================================="
