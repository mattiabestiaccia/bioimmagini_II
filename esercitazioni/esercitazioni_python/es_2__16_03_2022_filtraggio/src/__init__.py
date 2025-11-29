"""
Esercitazione 2 - Filtraggio 3D su Immagini CT
Bioimmagini - Positano

Moduli per il filtraggio 3D di immagini CT con valutazione SNR e acutezza.
"""

from .config import __version__

__author__ = "Bioimmagini Positano"

# Core modules
from .dicom_utils import load_dicom_volume, make_isotropic, check_isotropy
from .filters_3d import (
    moving_average_filter_3d,
    gaussian_filter_3d,
    wiener_filter_3d,
    estimate_noise_variance,
)
from .metrics import (
    calculate_snr,
    create_circular_roi,
    extract_profile,
    calculate_edge_sharpness,
)

# Configuration and types
from . import config
from . import types
from . import exceptions

__all__ = [
    # Version
    "__version__",
    "__author__",
    # DICOM utilities
    "load_dicom_volume",
    "make_isotropic",
    "check_isotropy",
    # Filters
    "moving_average_filter_3d",
    "gaussian_filter_3d",
    "wiener_filter_3d",
    "estimate_noise_variance",
    # Metrics
    "calculate_snr",
    "create_circular_roi",
    "extract_profile",
    "calculate_edge_sharpness",
    # Modules
    "config",
    "types",
    "exceptions",
]
