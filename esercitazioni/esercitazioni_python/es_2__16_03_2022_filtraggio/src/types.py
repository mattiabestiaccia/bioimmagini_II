"""
Type aliases for 3D CT filtering module.

This module defines custom type hints used throughout the filtering package.
"""

from typing import TypeAlias, Tuple
import numpy as np
import numpy.typing as npt

# Volume types
Volume3D: TypeAlias = npt.NDArray[np.float64]
"""3D array representing a CT volume."""

Slice2D: TypeAlias = npt.NDArray[np.float64]
"""2D array representing a single CT slice."""

BinaryMask: TypeAlias = npt.NDArray[np.bool_]
"""2D or 3D boolean array representing a binary mask."""

# ROI types
ROICoordinates: TypeAlias = Tuple[int, int, int]
"""ROI coordinates as (center_x, center_y, radius)."""

ProfileCoordinates: TypeAlias = Tuple[Tuple[int, int], Tuple[int, int]]
"""Profile coordinates as ((start_x, start_y), (end_x, end_y))."""

# Metadata types
DicomMetadata: TypeAlias = dict[str, any]
"""Dictionary containing DICOM metadata (spacing, dimensions, etc.)."""

VoxelSpacing: TypeAlias = Tuple[float, float, float]
"""Voxel spacing as (x_spacing, y_spacing, z_spacing) in mm."""

# Metrics types
MetricsDict: TypeAlias = dict[str, float]
"""Dictionary containing image quality metrics (SNR, sharpness, etc.)."""

ProfileArray: TypeAlias = npt.NDArray[np.float64]
"""1D array representing intensity profile."""

# Filter results
FilterResults: TypeAlias = dict[str, Volume3D | MetricsDict]
"""Dictionary containing filtered volumes and associated metrics."""

OptimizationResults: TypeAlias = dict[str, npt.NDArray[np.float64] | float | int]
"""Results from filter parameter optimization."""
