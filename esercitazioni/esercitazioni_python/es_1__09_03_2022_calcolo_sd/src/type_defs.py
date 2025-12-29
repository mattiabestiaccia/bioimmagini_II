"""
Type aliases for noise analysis module.

This module defines custom type hints used throughout the noise analysis package.
"""

from typing import TypeAlias
import numpy as np
import numpy.typing as npt

# Image types
ImageArray: TypeAlias = npt.NDArray[np.float64]
"""2D array representing an image."""

BinaryMask: TypeAlias = npt.NDArray[np.bool_]
"""2D boolean array representing a binary mask."""

IntArray: TypeAlias = npt.NDArray[np.int_]
"""Array of integers."""

# Statistical types
Statistics: TypeAlias = dict[str, float]
"""Dictionary containing statistical measures (mean, median, std, min, max)."""

HistogramData: TypeAlias = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
"""Tuple of (histogram counts, bin centers)."""

# ROI types
ROICoordinates: TypeAlias = tuple[int, int, int]
"""ROI coordinates as (center_row, center_col, radius)."""

ROIMask: TypeAlias = BinaryMask
"""Binary mask defining a region of interest."""

# Estimation results
SigmaEstimates: TypeAlias = dict[str, float]
"""Dictionary containing different sigma estimates (true, mean, median, histogram)."""
