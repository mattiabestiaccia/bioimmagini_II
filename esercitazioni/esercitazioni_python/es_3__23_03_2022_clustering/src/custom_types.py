"""Type aliases for cardiac segmentation pipeline.

This module defines type aliases using numpy.typing for improved type safety
and code readability throughout the segmentation pipeline.

Type Conventions:
- ImageStack: 3D array (height, width, time) of perfusion MRI frames
- TriggerTimes: 1D array of trigger times for each frame
- BinaryMask: 2D boolean array representing segmentation mask
- ClusterLabels: 2D integer array of cluster assignments

Author: Refactored for Python 3.12+ best practices
Date: 2024-11-29
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


# Image data types
ImageStack: TypeAlias = NDArray[np.float32]
"""3D array of shape (height, width, n_frames) representing temporal image sequence."""

TriggerTimes: TypeAlias = NDArray[np.float32]
"""1D array of shape (n_frames,) containing trigger times in milliseconds."""

# Segmentation types
BinaryMask: TypeAlias = NDArray[np.bool_]
"""2D boolean array of shape (height, width) representing binary segmentation mask."""

ClusterLabels: TypeAlias = NDArray[np.int32]
"""2D integer array of shape (height, width) with cluster labels for each pixel."""

# Time series data
TimeCurves: TypeAlias = NDArray[np.float32]
"""2D array of shape (n_pixels, n_frames) representing temporal intensity curves."""

# Distance matrices
DistanceMatrix: TypeAlias = NDArray[np.float64]
"""2D array of shape (n_samples, n_samples) representing pairwise distances."""

# Clustering results
Centroids: TypeAlias = NDArray[np.float32]
"""2D array of shape (n_clusters, n_frames) representing cluster centroids."""

# ROI coordinates
ROICoordinates: TypeAlias = NDArray[np.int32]
"""2D array of shape (n_points, 2) containing (row, col) coordinates of ROI pixels."""
