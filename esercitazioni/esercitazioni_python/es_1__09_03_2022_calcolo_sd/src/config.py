"""
Configuration constants for noise analysis module.

This module centralizes all configuration parameters, default values,
and constants used across the noise analysis package.
"""

from pathlib import Path
import numpy as np

# ==================== Paths ====================

# Get project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DICOM_DIR = DATA_DIR
PHANTOM_FILE = DATA_DIR / "phantom.dcm"

# Output directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

# ==================== Synthetic Image Parameters ====================

SYNTHETIC_IMAGE_CONFIG = {
    "dimension": 512,
    "sigma_noise": 5.0,
    "base_intensity": 50.0,
    "patterns": [
        # (row_start, row_end, col_start, col_end, intensity)
        (50, 101, 50, 101, 120),
        (101, 181, 101, 451, 200),
        (200, 501, 200, 351, 90),
        (230, 271, 230, 271, 250),
        (5, 401, 450, 501, 150),
    ],
}

# ==================== SD Map Parameters ====================

SD_MAP_CONFIG = {
    "default_kernel_size": 5,
    "alternative_kernel_sizes": [3, 5, 7, 9, 11],
    "edge_mode": "constant",
    "edge_value": 0.0,
}

# ==================== Histogram Parameters ====================

HISTOGRAM_CONFIG = {
    "n_bins": 100,
    "density": False,
    "alpha": 0.7,
}

# ==================== ROI Parameters ====================

# Default ROI coordinates for phantom analysis
# Format: (center_row, center_col, radius)
DEFAULT_ROIS = {
    "oil": (100, 100, 20),
    "water": (100, 200, 20),
    "background": (50, 50, 15),
}

ROI_CONFIG = {
    "default_radius": 20,
    "min_radius": 5,
    "max_radius": 50,
}

# ==================== Noise Estimation ====================

RAYLEIGH_CORRECTION_FACTOR = float(np.sqrt(2.0 / (4.0 - np.pi)))
"""
Rayleigh correction factor for MRI background noise.
Value: sqrt(2/(4-π)) ≈ 1.526

Reference:
    Constantinides, C. D., Atalar, E., & McVeigh, E. R. (1997).
    Signal-to-noise measurements in magnitude images from NMR phased arrays.
"""

NOISE_ESTIMATION_CONFIG = {
    "apply_rayleigh_correction": True,
    "correction_factor": RAYLEIGH_CORRECTION_FACTOR,
    "zero_padding_threshold": 0.0,
    "intensity_threshold": 100.0,  # For automatic ROI selection
}

# ==================== Monte Carlo Parameters ====================

MONTE_CARLO_CONFIG = {
    "n_simulations": 100,
    "roi_sizes": [2, 4, 8, 16, 32, 64, 128],
    "true_mean": 100.0,
    "true_sigma": 10.0,
    "confidence_level": 0.95,
}

# ==================== Visualization Parameters ====================

VISUALIZATION_CONFIG = {
    "figure_dpi": 150,
    "figure_format": "png",
    "cmap_image": "gray",
    "cmap_sd": "hot",
    "save_figures": True,
    "show_figures": True,
}

PLOT_COLORS = {
    "true_value": "green",
    "mean_estimate": "blue",
    "median_estimate": "red",
    "histogram_estimate": "orange",
    "roi_oil": "yellow",
    "roi_water": "cyan",
    "roi_background": "magenta",
}

# ==================== Analysis Thresholds ====================

ANALYSIS_THRESHOLDS = {
    "min_roi_pixels": 10,
    "max_roi_pixels": 10000,
    "acceptable_error_percent": 5.0,
    "good_error_percent": 2.0,
    "excellent_error_percent": 1.0,
}

# ==================== Output Filenames ====================

OUTPUT_FILENAMES = {
    # Synthetic image analysis
    "synthetic_image": "synthetic_image.png",
    "synthetic_histogram": "synthetic_histogram.png",
    "synthetic_sd_map": "synthetic_sd_map.png",
    "synthetic_comparison": "synthetic_comparison.png",

    # Phantom analysis
    "phantom_manual_rois": "phantom_manual_rois.png",
    "phantom_auto_kernel3": "phantom_auto_kernel3.png",
    "phantom_auto_kernel9": "phantom_auto_kernel9.png",
    "phantom_comparison": "phantom_comparison.png",

    # Monte Carlo
    "monte_carlo_convergence": "monte_carlo_convergence.png",
    "monte_carlo_table": "monte_carlo_results.csv",
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
