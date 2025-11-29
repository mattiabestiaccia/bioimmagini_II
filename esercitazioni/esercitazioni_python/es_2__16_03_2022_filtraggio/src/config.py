"""
Configuration constants for 3D CT filtering module.

This module centralizes all configuration parameters, default values,
and constants used across the CT filtering package.
"""

from pathlib import Path
import numpy as np

# ==================== Paths ====================

# Get project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
CT_SERIES_DIR = DATA_DIR / "Phantom_CT_PET" / "2-CT 2.5mm-5.464"

# Output directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR
TABLES_DIR = RESULTS_DIR

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)

# ==================== DICOM Loading ====================

DICOM_CONFIG = {
    "series_name": "2-CT 2.5mm-5.464",
    "rescale_hu": True,
    "check_consistency": True,
}

# ==================== Volume Processing ====================

VOLUME_CONFIG = {
    "make_isotropic": True,
    "interpolation_order": 1,  # Linear interpolation
    "target_spacing": None,  # Auto-detect minimum spacing
}

# ==================== Filter Parameters ====================

FILTER_CONFIG = {
    "kernel_size": 7,
    "default_sigma": 1.0,
    "sigma_range": np.linspace(0.5, 3.0, 20),
}

# Gaussian filter optimization
GAUSSIAN_OPTIMIZATION = {
    "sigma_min": 0.5,
    "sigma_max": 3.0,
    "n_samples": 20,
    "snr_weight": 0.7,
    "sharpness_weight": 0.3,
}

# Wiener filter
WIENER_CONFIG = {
    "noise_variance": None,  # Auto-estimate if None
    "estimate_method": "robust",  # 'robust' or 'background'
}

# ==================== ROI Parameters ====================

# Default ROI for SNR calculation (adjust based on specific image)
DEFAULT_ROI = {
    "center_x": 256,
    "center_y": 256,
    "radius": 80,
}

# Profile extraction for sharpness measurement
DEFAULT_PROFILE = {
    "start_x": 256,
    "start_y": 150,
    "end_x": 256,
    "end_y": 350,
}

ROI_CONFIG = {
    "default_roi": DEFAULT_ROI,
    "default_profile": DEFAULT_PROFILE,
    "min_radius": 10,
    "max_radius": 200,
}

# ==================== Metrics ====================

METRICS_CONFIG = {
    "snr_method": "mean_std",  # 'mean_std' or 'signal_background'
    "edge_sharpness_method": "gradient",  # 'gradient' or 'derivative'
    "profile_samples": 100,
}

# ==================== Visualization Parameters ====================

VISUALIZATION_CONFIG = {
    "figure_dpi": 150,
    "figure_format": "png",
    "cmap_ct": "gray",
    "cmap_difference": "RdBu_r",
    "save_figures": True,
    "show_figures": False,
    "vmin_hu": -100,
    "vmax_hu": 400,
}

PLOT_COLORS = {
    "original": "black",
    "moving_average": "blue",
    "gaussian": "green",
    "wiener": "red",
    "roi_outline": "yellow",
    "profile_line": "cyan",
}

PLOT_STYLES = {
    "original": {"linestyle": "-", "linewidth": 2},
    "moving_average": {"linestyle": "--", "linewidth": 1.5},
    "gaussian": {"linestyle": "-.", "linewidth": 1.5},
    "wiener": {"linestyle": ":", "linewidth": 2},
}

# ==================== Output Filenames ====================

OUTPUT_FILENAMES = {
    "comparison": "confronto_filtri.png",
    "profiles": "confronto_profili.png",
    "differences": "differenze_filtri.png",
    "optimization": "ottimizzazione_sigma.png",
    "results_table": "risultati.txt",
    "interactive_roi": "roi_selezionata.png",
}

# ==================== Slice Selection ====================

SLICE_CONFIG = {
    "default_slice": "middle",  # 'middle', 'auto', or int
    "auto_method": "max_contrast",  # 'max_contrast' or 'max_variance'
}

# ==================== Analysis Thresholds ====================

ANALYSIS_THRESHOLDS = {
    "min_snr": 1.0,
    "good_snr": 3.0,
    "excellent_snr": 5.0,
    "min_sharpness": 10.0,
    "good_sharpness": 50.0,
}

# ==================== CT-Specific Constants ====================

HU_CONFIG = {
    "air": -1000,
    "water": 0,
    "soft_tissue": 40,
    "bone": 400,
    "default_window_center": 40,
    "default_window_width": 400,
}

# ==================== Logging Configuration ====================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}

# ==================== Random Seed ====================

RANDOM_SEED = 42
"""Random seed for reproducibility."""

# ==================== Version ====================

__version__ = "2.0.0"
