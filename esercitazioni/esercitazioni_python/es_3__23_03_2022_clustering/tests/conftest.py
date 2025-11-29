"""Pytest fixtures for cardiac segmentation tests.

This module provides reusable test fixtures for unit and integration tests
of the cardiac segmentation pipeline.

Author: Refactored for Python 3.12+ best practices
Date: 2024-11-29
"""

from pathlib import Path

import numpy as np
import pytest

from src.config import (
    KMeansConfig,
    PostProcessConfig,
    TissueType,
)
from src.types import BinaryMask, ImageStack, TimeCurves, TriggerTimes


# ============================================================================
# Image Data Fixtures
# ============================================================================

@pytest.fixture
def sample_image_stack() -> ImageStack:
    """Generate synthetic cardiac perfusion image stack.

    Creates a 3D array simulating temporal perfusion MRI data with
    realistic intensity curves for different tissue types.

    Returns:
        ImageStack of shape (128, 128, 30) with synthetic perfusion data.
    """
    height, width, n_frames = 128, 128, 30
    stack = np.zeros((height, width, n_frames), dtype=np.float32)

    # Create circular ROIs for different tissues
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2

    # Left ventricle (center, high intensity, rapid wash-in/wash-out)
    lv_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= 15 ** 2
    lv_curve = _generate_blood_pool_curve(n_frames, peak_frame=8, peak_intensity=1.0)

    # Myocardium (ring around LV, moderate intensity, slower kinetics)
    myo_mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= 30 ** 2) & ~lv_mask
    myo_curve = _generate_myocardium_curve(n_frames, peak_frame=12, peak_intensity=0.6)

    # Right ventricle (offset from center, similar to LV but lower intensity)
    rv_mask = (x - center_x + 20) ** 2 + (y - center_y) ** 2 <= 12 ** 2
    rv_curve = _generate_blood_pool_curve(n_frames, peak_frame=7, peak_intensity=0.8)

    # Background (remaining pixels, low intensity, minimal variation)
    bg_mask = ~(lv_mask | myo_mask | rv_mask)
    bg_curve = _generate_background_curve(n_frames)

    # Assign curves to regions
    for t in range(n_frames):
        stack[:, :, t][lv_mask] = lv_curve[t]
        stack[:, :, t][myo_mask] = myo_curve[t]
        stack[:, :, t][rv_mask] = rv_curve[t]
        stack[:, :, t][bg_mask] = bg_curve[t]

    # Add realistic noise
    noise = np.random.normal(0, 0.02, stack.shape).astype(np.float32)
    stack += noise
    stack = np.clip(stack, 0, 1)

    return stack


@pytest.fixture
def sample_trigger_times() -> TriggerTimes:
    """Generate synthetic trigger times for perfusion sequence.

    Returns:
        TriggerTimes array with 30 evenly spaced time points.
    """
    n_frames = 30
    # Simulate trigger times in milliseconds (0 to 29 seconds)
    return np.linspace(0, 29000, n_frames, dtype=np.float32)


@pytest.fixture
def sample_time_curves() -> TimeCurves:
    """Generate synthetic time-intensity curves for testing.

    Returns:
        TimeCurves array of shape (1000, 30) with various curve patterns.
    """
    n_pixels = 1000
    n_frames = 30

    curves = np.zeros((n_pixels, n_frames), dtype=np.float32)

    # Generate different curve types
    for i in range(n_pixels):
        curve_type = i % 4
        if curve_type == 0:  # Blood pool
            curves[i] = _generate_blood_pool_curve(n_frames, peak_frame=8,
                                                   peak_intensity=1.0)
        elif curve_type == 1:  # Myocardium
            curves[i] = _generate_myocardium_curve(n_frames, peak_frame=12,
                                                   peak_intensity=0.6)
        elif curve_type == 2:  # Right ventricle
            curves[i] = _generate_blood_pool_curve(n_frames, peak_frame=7,
                                                   peak_intensity=0.8)
        else:  # Background
            curves[i] = _generate_background_curve(n_frames)

    # Add noise
    noise = np.random.normal(0, 0.02, curves.shape).astype(np.float32)
    curves += noise

    return curves


# ============================================================================
# Mask Fixtures
# ============================================================================

@pytest.fixture
def sample_binary_mask() -> BinaryMask:
    """Generate synthetic binary segmentation mask.

    Returns:
        BinaryMask of shape (128, 128) with circular ROI.
    """
    height, width = 128, 128
    mask = np.zeros((height, width), dtype=bool)

    # Create circular region
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= 30 ** 2

    return mask


@pytest.fixture
def sample_tissue_masks() -> dict[TissueType, BinaryMask]:
    """Generate synthetic tissue masks for all tissue types.

    Returns:
        Dictionary mapping TissueType to binary masks.
    """
    height, width = 128, 128
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2

    # Create non-overlapping regions
    lv_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= 15 ** 2
    myo_mask = (
        ((x - center_x) ** 2 + (y - center_y) ** 2 <= 30 ** 2) & ~lv_mask
    )
    rv_mask = (
        ((x - center_x + 20) ** 2 + (y - center_y) ** 2 <= 12 ** 2)
        & ~lv_mask
        & ~myo_mask
    )
    bg_mask = ~(lv_mask | myo_mask | rv_mask)

    return {
        TissueType.LEFT_VENTRICLE: lv_mask,
        TissueType.MYOCARDIUM: myo_mask,
        TissueType.RIGHT_VENTRICLE: rv_mask,
        TissueType.BACKGROUND: bg_mask,
    }


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def default_kmeans_config() -> KMeansConfig:
    """Provide default K-means configuration for testing.

    Returns:
        KMeansConfig with default parameters.
    """
    return KMeansConfig()


@pytest.fixture
def default_postprocess_config() -> PostProcessConfig:
    """Provide default post-processing configuration for testing.

    Returns:
        PostProcessConfig with default parameters.
    """
    return PostProcessConfig()


@pytest.fixture
def fast_kmeans_config() -> KMeansConfig:
    """Provide fast K-means configuration for quick testing.

    Returns:
        KMeansConfig with reduced iterations for faster tests.
    """
    return KMeansConfig(
        n_clusters=4,
        max_iter=50,
        n_init=3,
        random_state=42
    )


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_dicom_dir(tmp_path: Path) -> Path:
    """Create temporary directory for DICOM test files.

    Args:
        tmp_path: Pytest temporary path fixture.

    Yields:
        Path to temporary DICOM directory.
    """
    dicom_dir = tmp_path / "dicom_data"
    dicom_dir.mkdir()
    return dicom_dir


@pytest.fixture
def temp_results_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test results.

    Args:
        tmp_path: Pytest temporary path fixture.

    Yields:
        Path to temporary results directory.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir


# ============================================================================
# Helper Functions (Private)
# ============================================================================

def _generate_blood_pool_curve(
    n_frames: int,
    peak_frame: int = 8,
    peak_intensity: float = 1.0
) -> np.ndarray:
    """Generate synthetic blood pool perfusion curve.

    Simulates rapid wash-in followed by gradual wash-out characteristic
    of blood pool enhancement.

    Args:
        n_frames: Number of temporal frames.
        peak_frame: Frame index of peak enhancement.
        peak_intensity: Maximum intensity value.

    Returns:
        1D array of intensity values.
    """
    curve = np.zeros(n_frames, dtype=np.float32)
    baseline = 0.1

    for t in range(n_frames):
        if t < peak_frame:
            # Rapid wash-in
            curve[t] = baseline + (peak_intensity - baseline) * (t / peak_frame) ** 2
        else:
            # Gradual wash-out
            decay = np.exp(-0.1 * (t - peak_frame))
            curve[t] = baseline + (peak_intensity - baseline) * decay

    return curve


def _generate_myocardium_curve(
    n_frames: int,
    peak_frame: int = 12,
    peak_intensity: float = 0.6
) -> np.ndarray:
    """Generate synthetic myocardium perfusion curve.

    Simulates slower, more gradual enhancement and wash-out characteristic
    of myocardial tissue perfusion.

    Args:
        n_frames: Number of temporal frames.
        peak_frame: Frame index of peak enhancement.
        peak_intensity: Maximum intensity value.

    Returns:
        1D array of intensity values.
    """
    curve = np.zeros(n_frames, dtype=np.float32)
    baseline = 0.15

    for t in range(n_frames):
        if t < peak_frame:
            # Gradual wash-in
            curve[t] = baseline + (peak_intensity - baseline) * (t / peak_frame)
        else:
            # Slower wash-out
            decay = np.exp(-0.05 * (t - peak_frame))
            curve[t] = baseline + (peak_intensity - baseline) * decay

    return curve


def _generate_background_curve(n_frames: int) -> np.ndarray:
    """Generate synthetic background curve with minimal variation.

    Args:
        n_frames: Number of temporal frames.

    Returns:
        1D array of low, nearly constant intensity values.
    """
    baseline = 0.05
    variation = 0.01
    curve = baseline + variation * np.random.randn(n_frames)
    return curve.astype(np.float32)
