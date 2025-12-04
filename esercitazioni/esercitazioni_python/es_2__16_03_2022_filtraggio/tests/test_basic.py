import sys
import os
from pathlib import Path
import pytest

# Add the exercise root to path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.dicom_utils import load_dicom_volume
from src.filters_3d import moving_average_filter_3d

def test_imports():
    assert True

def test_data_exists():
    data_path = Path(__file__).parent.parent / "data" / "Phantom_CT_PET"
    assert data_path.exists()
    assert data_path.is_dir()
    # Check for subdirectories
    subdirs = list(data_path.glob("*"))
    assert len(subdirs) > 0
