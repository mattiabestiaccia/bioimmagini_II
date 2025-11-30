"""Configuration dataclasses, enums, and constants for cardiac segmentation.

This module provides type-safe configuration using dataclasses, enums for
categorical values, and constants for imaging parameters.

Author: Refactored for Python 3.12+ best practices
Date: 2024-11-29
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final

import numpy as np


# ============================================================================
# Enums for Type Safety
# ============================================================================

class TissueType(Enum):
    """Cardiac tissue types identified in segmentation.

    Each tissue type has associated properties including display colors
    and cluster priority for segmentation algorithms.
    """

    BACKGROUND = "background"
    RIGHT_VENTRICLE = "rv"
    LEFT_VENTRICLE = "lv"
    MYOCARDIUM = "myo"

    @property
    def color_rgb(self) -> tuple[float, float, float]:
        """RGB color for visualization (values in [0, 1])."""
        return {
            self.BACKGROUND: (0.5, 0.5, 0.5),      # Gray
            self.RIGHT_VENTRICLE: (0.0, 0.0, 1.0), # Blue
            self.LEFT_VENTRICLE: (1.0, 0.0, 0.0),  # Red
            self.MYOCARDIUM: (0.0, 1.0, 0.0),      # Green
        }[self]

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        return {
            self.BACKGROUND: "Background",
            self.RIGHT_VENTRICLE: "Right Ventricle",
            self.LEFT_VENTRICLE: "Left Ventricle",
            self.MYOCARDIUM: "Myocardium",
        }[self]

    @property
    def cluster_priority(self) -> int:
        """Priority for cluster assignment (lower = higher priority)."""
        return {
            self.BACKGROUND: 3,
            self.RIGHT_VENTRICLE: 2,
            self.LEFT_VENTRICLE: 0,
            self.MYOCARDIUM: 1,
        }[self]


class DistanceMetric(Enum):
    """Distance metrics for clustering and similarity computation."""

    EUCLIDEAN = "euclidean"
    CORRELATION = "correlation"
    COSINE = "cosine"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"

    @property
    def scipy_name(self) -> str:
        """Corresponding scipy.spatial.distance metric name."""
        return self.value


class DiceQuality(Enum):
    """Quality categories for Dice coefficient scores."""

    EXCELLENT = "excellent"  # Dice >= 0.9
    GOOD = "good"           # Dice >= 0.7
    MODERATE = "moderate"   # Dice >= 0.5
    FAIR = "moderate"       # Alias for MODERATE (backward compatibility)
    POOR = "poor"           # Dice < 0.5

    @classmethod
    def from_score(cls, dice_score: float) -> "DiceQuality":
        """Classify Dice score into quality category."""
        if dice_score >= 0.9:
            return cls.EXCELLENT
        if dice_score >= 0.7:
            return cls.GOOD
        if dice_score >= 0.5:
            return cls.MODERATE
        return cls.POOR

    @property
    def color_hex(self) -> str:
        """Hex color for quality visualization."""
        return {
            self.EXCELLENT: "#00FF00",  # Green
            self.GOOD: "#90EE90",       # Light green
            self.MODERATE: "#FFD700",   # Gold
            self.POOR: "#FF0000",       # Red
        }[self]


# ============================================================================
# Constants
# ============================================================================

class ImagingConstants:
    """Constants for cardiac MRI imaging parameters.

    These values are based on typical cardiac perfusion imaging protocols
    and should be adjusted based on specific acquisition parameters.
    """

    # Temporal parameters
    DEFAULT_PEAK_FRAME: Final[int] = 12
    """Default frame index for peak enhancement in perfusion sequences."""

    MIN_TEMPORAL_FRAMES: Final[int] = 10
    """Minimum number of temporal frames required for analysis."""

    MAX_TEMPORAL_FRAMES: Final[int] = 100
    """Maximum expected number of temporal frames."""

    DEFAULT_HEARTBEAT_MS: Final[float] = 800.0
    """Default heartbeat duration in milliseconds (typical cardiac cycle)."""

    MS_TO_SECONDS: Final[float] = 1000.0
    """Conversion factor from milliseconds to seconds."""

    # Clustering parameters
    N_CARDIAC_CLUSTERS: Final[int] = 4
    """Number of clusters for cardiac tissue segmentation (BG, RV, LV, MYO)."""

    # Spatial parameters
    MIN_IMAGE_SIZE: Final[int] = 64
    """Minimum acceptable image dimension (height or width)."""

    MAX_IMAGE_SIZE: Final[int] = 512
    """Maximum expected image dimension."""

    # Quality thresholds
    MIN_ACCEPTABLE_DICE: Final[float] = 0.5
    """Minimum Dice coefficient for acceptable segmentation quality."""

    EXCELLENT_DICE_THRESHOLD: Final[float] = 0.9
    """Dice coefficient threshold for excellent segmentation."""

    # Numerical stability
    EPSILON: Final[float] = 1e-8
    """Small constant for numerical stability in divisions."""

    # Default random seed for reproducibility
    RANDOM_SEED: Final[int] = 42
    """Default random seed for reproducible results."""


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass(frozen=True)
class KMeansConfig:
    """Configuration for K-means clustering algorithm.

    This frozen dataclass ensures immutability of clustering parameters
    throughout the pipeline execution.
    """

    n_clusters: int = ImagingConstants.N_CARDIAC_CLUSTERS
    """Number of clusters to identify."""

    metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    """Distance metric for clustering."""

    random_state: int = ImagingConstants.RANDOM_SEED
    """Random seed for reproducibility."""

    max_iter: int = 300
    """Maximum number of iterations for convergence."""

    n_init: int = 10
    """Number of times the algorithm runs with different centroid seeds."""

    tol: float = 1e-4
    """Relative tolerance for convergence."""

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {self.n_clusters}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {self.n_init}")
        if self.tol <= 0:
            raise ValueError(f"tol must be > 0, got {self.tol}")


@dataclass(frozen=True)
class PostProcessConfig:
    """Configuration for post-processing operations.

    Controls morphological operations and filtering applied to
    segmentation results.
    """

    apply_morphology: bool = True
    """Whether to apply morphological operations."""

    erosion_kernel_size: int = 3
    """Kernel size for morphological erosion."""

    dilation_kernel_size: int = 3
    """Kernel size for morphological dilation."""

    min_component_size: int = 50
    """Minimum size (in pixels) for connected components to keep."""

    fill_holes: bool = True
    """Whether to fill holes in segmented regions."""

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.erosion_kernel_size < 1 or self.erosion_kernel_size % 2 == 0:
            raise ValueError(f"erosion_kernel_size must be odd and >= 1, "
                           f"got {self.erosion_kernel_size}")
        if self.dilation_kernel_size < 1 or self.dilation_kernel_size % 2 == 0:
            raise ValueError(f"dilation_kernel_size must be odd and >= 1, "
                           f"got {self.dilation_kernel_size}")
        if self.min_component_size < 0:
            raise ValueError(f"min_component_size must be >= 0, "
                           f"got {self.min_component_size}")


@dataclass
class SegmentationResult:
    """Results from segmentation pipeline execution.

    This mutable dataclass stores all outputs from the segmentation
    pipeline for subsequent analysis and visualization.
    """

    # Input data reference
    image_stack_shape: tuple[int, int, int]
    """Shape of the input image stack (height, width, n_frames)."""

    # Segmentation outputs
    cluster_labels: np.ndarray
    """2D array of cluster labels for each pixel."""

    tissue_masks: dict[TissueType, np.ndarray]
    """Binary masks for each identified tissue type."""

    # Quality metrics
    dice_scores: dict[str, float] = field(default_factory=dict)
    """Dice coefficient scores for each tissue comparison."""

    inertia: float | None = None
    """K-means inertia (sum of squared distances to centroids)."""

    # Timing information
    elapsed_time_seconds: float = 0.0
    """Total pipeline execution time in seconds."""

    # Configuration used
    kmeans_config: KMeansConfig = field(default_factory=KMeansConfig)
    """K-means configuration used for this segmentation."""

    postprocess_config: PostProcessConfig = field(default_factory=PostProcessConfig)
    """Post-processing configuration used."""

    def get_quality_summary(self) -> dict[str, DiceQuality]:
        """Get quality classification for all Dice scores.

        Returns:
            Dictionary mapping tissue comparison names to quality categories.
        """
        return {
            name: DiceQuality.from_score(score)
            for name, score in self.dice_scores.items()
        }

    def get_average_dice(self) -> float:
        """Calculate average Dice coefficient across all comparisons.

        Returns:
            Mean Dice score, or 0.0 if no scores available.
        """
        if not self.dice_scores:
            return 0.0
        return float(np.mean(list(self.dice_scores.values())))


@dataclass(frozen=True)
class DicomLoadConfig:
    """Configuration for DICOM data loading.

    Controls how DICOM files are read and processed into image stacks.
    """

    dicom_dir: Path
    """Directory containing DICOM files."""

    n_frames: int | None = None
    """Number of frames to load (None = load all)."""

    validate_metadata: bool = True
    """Whether to validate DICOM metadata consistency."""

    sort_by_trigger_time: bool = True
    """Whether to sort frames by trigger time."""

    parallel_loading: bool = True
    """Whether to use parallel loading for performance."""

    max_workers: int = 8
    """Maximum number of worker threads for parallel loading."""

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not self.dicom_dir.exists():
            raise ValueError(f"DICOM directory does not exist: {self.dicom_dir}")
        if self.n_frames is not None and self.n_frames < 1:
            raise ValueError(f"n_frames must be >= 1 or None, got {self.n_frames}")
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
