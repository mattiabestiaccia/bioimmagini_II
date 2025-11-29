#!/usr/bin/env python3
"""
utils.py - Demons Registration and Hierarchical Clustering for Temporal Series

This module implements the Demons algorithm for non-rigid (deformable) image
registration, designed for temporal series registration with respiratory motion.
Includes hierarchical clustering to group similar images before registration.

Esercitazione 8: Registrazione Serie Temporali
Dataset: Renal perfusion MRI with 70 frames (2D+T)

Author: Biomedical Imaging Course
Date: 2025
"""

import numpy as np
import pydicom
from pathlib import Path
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from skimage import filters
from typing import Tuple, List, Optional, Dict
import warnings


def load_dicom_series(dicom_dir: str, expected_frames: Optional[int] = None) -> Tuple[np.ndarray, List[pydicom.Dataset]]:
    """
    Load a DICOM series from a directory.

    Parameters
    ----------
    dicom_dir : str
        Path to directory containing DICOM files
    expected_frames : int, optional
        Expected number of frames for validation

    Returns
    -------
    volume : np.ndarray
        3D array of shape (n_frames, height, width)
    datasets : list of pydicom.Dataset
        List of DICOM datasets for metadata access

    Notes
    -----
    Temporal series are loaded in alphabetical order by filename.
    For proper temporal ordering, ensure filenames are sequential.
    """
    dicom_path = Path(dicom_dir)

    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    # Get all DICOM files
    dicom_files = sorted(dicom_path.glob("*.dcm"))

    if not dicom_files:
        # Try without .dcm extension
        dicom_files = sorted([f for f in dicom_path.iterdir() if f.is_file()])

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    n_frames = len(dicom_files)

    if expected_frames is not None and n_frames != expected_frames:
        warnings.warn(f"Expected {expected_frames} frames, found {n_frames}")

    # Read first file to get dimensions
    ds0 = pydicom.dcmread(dicom_files[0])
    height, width = ds0.pixel_array.shape

    # Allocate volume
    volume = np.zeros((n_frames, height, width), dtype=np.float32)
    datasets = []

    # Load all frames
    for i, dcm_file in enumerate(dicom_files):
        ds = pydicom.dcmread(dcm_file)
        volume[i] = ds.pixel_array.astype(np.float32)
        datasets.append(ds)

    print(f"Loaded {n_frames} frames of size {height}x{width}")

    return volume, datasets


def normalize_image(image: np.ndarray, percentile_clip: Tuple[float, float] = (1, 99)) -> np.ndarray:
    """
    Normalize image to [0, 1] range with optional percentile clipping.

    Parameters
    ----------
    image : np.ndarray
        Input image
    percentile_clip : tuple of float, optional
        Lower and upper percentiles for clipping (default: (1, 99))

    Returns
    -------
    normalized : np.ndarray
        Normalized image in [0, 1]
    """
    if percentile_clip is not None:
        p_low, p_high = np.percentile(image, percentile_clip)
        image_clipped = np.clip(image, p_low, p_high)
    else:
        image_clipped = image

    min_val = image_clipped.min()
    max_val = image_clipped.max()

    if max_val - min_val < 1e-10:
        warnings.warn("Image has constant intensity")
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image_clipped - min_val) / (max_val - min_val)
    return normalized.astype(np.float32)


def compute_distance_matrix(images: np.ndarray, metric: str = 'mse') -> np.ndarray:
    """
    Compute pairwise distance matrix between images.

    Parameters
    ----------
    images : np.ndarray
        Array of shape (n_images, height, width)
    metric : str, optional
        Distance metric: 'mse' (mean squared error) or 'mae' (mean absolute error)
        Default: 'mse'

    Returns
    -------
    distance_matrix : np.ndarray
        Symmetric distance matrix of shape (n_images, n_images)

    Notes
    -----
    MSE is computed as: D(i,j) = mean((I_i - I_j)^2)
    This metric is used in the PDF for hierarchical clustering.
    """
    n_images = images.shape[0]
    distance_matrix = np.zeros((n_images, n_images), dtype=np.float32)

    for i in range(n_images):
        for j in range(i+1, n_images):
            if metric == 'mse':
                dist = np.mean((images[i] - images[j]) ** 2)
            elif metric == 'mae':
                dist = np.mean(np.abs(images[i] - images[j]))
            else:
                raise ValueError(f"Unknown metric: {metric}")

            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric

    return distance_matrix


def hierarchical_clustering(distance_matrix: np.ndarray,
                           n_clusters: int = 2,
                           method: str = 'average') -> Dict:
    """
    Perform hierarchical clustering on distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Symmetric distance matrix of shape (n_samples, n_samples)
    n_clusters : int, optional
        Number of clusters to form (default: 2)
    method : str, optional
        Linkage method: 'single', 'complete', 'average', 'ward'
        Default: 'average' (as in PDF)

    Returns
    -------
    clustering_results : dict
        Dictionary containing:
        - 'linkage_matrix': Linkage matrix for dendrogram
        - 'labels': Cluster labels for each sample
        - 'n_clusters': Number of clusters

    Notes
    -----
    The PDF uses hierarchical clustering to separate pre-contrast and
    post-contrast images based on intensity similarity.
    """
    # Convert distance matrix to condensed form for scipy
    condensed_dist = squareform(distance_matrix)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method=method)

    # Cut dendrogram to form n_clusters
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    # Convert labels from 1-based to 0-based
    labels = labels - 1

    results = {
        'linkage_matrix': linkage_matrix,
        'labels': labels,
        'n_clusters': n_clusters
    }

    return results


def compute_image_gradient(image: np.ndarray, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradient using Gaussian derivatives.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D)
    sigma : float, optional
        Standard deviation of Gaussian filter for smoothing (default: 1.0)

    Returns
    -------
    grad_y : np.ndarray
        Gradient in y direction (rows)
    grad_x : np.ndarray
        Gradient in x direction (columns)

    Notes
    -----
    Gradient is computed after Gaussian smoothing to reduce noise sensitivity.
    This is essential for the Demons algorithm stability.
    """
    # Smooth image first
    smoothed = filters.gaussian(image, sigma=sigma, preserve_range=True)

    # Compute gradients using Sobel operator
    grad_y = ndimage.sobel(smoothed, axis=0)
    grad_x = ndimage.sobel(smoothed, axis=1)

    return grad_y, grad_x


def demons_step(moving: np.ndarray,
                fixed: np.ndarray,
                displacement_field: np.ndarray,
                alpha: float = 2.5,
                sigma_diffusion: float = 1.0) -> np.ndarray:
    """
    Perform one iteration of the Demons registration algorithm.

    Parameters
    ----------
    moving : np.ndarray
        Moving image F(x) to be registered (2D)
    fixed : np.ndarray
        Fixed (reference) image R(x) (2D)
    displacement_field : np.ndarray
        Current displacement field of shape (2, height, width)
        [0] = y-displacement, [1] = x-displacement
    alpha : float, optional
        Regularization parameter to prevent division by zero (default: 2.5)
        Controls the magnitude of displacement updates
    sigma_diffusion : float, optional
        Standard deviation for Gaussian smoothing of displacement field (default: 1.0)
        Acts as diffusion regularization

    Returns
    -------
    new_displacement_field : np.ndarray
        Updated displacement field of shape (2, height, width)

    Notes
    -----
    The Demons algorithm update rule (from PDF):

        U(n) = (F(n) - R) * grad_R / (||grad_R||^2 + alpha^2 * (F(n) - R)^2)

    Where:
    - F(n) is the moving image warped by current displacement
    - R is the fixed (reference) image
    - grad_R is the gradient of the fixed image
    - alpha is a regularization parameter

    The displacement field is then smoothed (diffusion) to ensure smoothness.

    This is analogous to Maxwell's demons: each pixel acts as a "demon" that
    pushes the moving image towards the fixed image along the intensity gradient.
    """
    # Warp moving image with current displacement field
    warped_moving = warp_image(moving, displacement_field)

    # Compute gradient of fixed image
    grad_y, grad_x = compute_image_gradient(fixed, sigma=1.0)

    # Compute intensity difference
    diff = warped_moving - fixed

    # Compute denominator: ||grad_R||^2 + alpha^2 * (F - R)^2
    grad_magnitude_sq = grad_y**2 + grad_x**2
    denominator = grad_magnitude_sq + alpha**2 * diff**2

    # Avoid division by zero
    denominator = np.maximum(denominator, 1e-10)

    # Compute displacement update: (F - R) * grad_R / denominator
    update_y = diff * grad_y / denominator
    update_x = diff * grad_x / denominator

    # Update displacement field
    new_disp_y = displacement_field[0] + update_y
    new_disp_x = displacement_field[1] + update_x

    # Apply Gaussian smoothing (diffusion regularization)
    if sigma_diffusion > 0:
        new_disp_y = filters.gaussian(new_disp_y, sigma=sigma_diffusion, preserve_range=True)
        new_disp_x = filters.gaussian(new_disp_x, sigma=sigma_diffusion, preserve_range=True)

    new_displacement_field = np.stack([new_disp_y, new_disp_x], axis=0)

    return new_displacement_field


def warp_image(image: np.ndarray, displacement_field: np.ndarray) -> np.ndarray:
    """
    Warp image using displacement field.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D)
    displacement_field : np.ndarray
        Displacement field of shape (2, height, width)
        [0] = y-displacement, [1] = x-displacement

    Returns
    -------
    warped : np.ndarray
        Warped image

    Notes
    -----
    Uses bilinear interpolation via scipy.ndimage.map_coordinates.
    The displacement field defines how each pixel in the output image
    maps to a location in the input image.
    """
    height, width = image.shape

    # Create coordinate grid
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Apply displacement
    new_y = y_coords + displacement_field[0]
    new_x = x_coords + displacement_field[1]

    # Warp image using map_coordinates (bilinear interpolation)
    coords = np.array([new_y, new_x])
    warped = ndimage.map_coordinates(image, coords, order=1, mode='nearest')

    return warped


def demons_registration(moving: np.ndarray,
                       fixed: np.ndarray,
                       n_iterations: int = 100,
                       alpha: float = 2.5,
                       sigma_diffusion: float = 1.0,
                       tolerance: float = 1e-4,
                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Register moving image to fixed image using Demons algorithm.

    Parameters
    ----------
    moving : np.ndarray
        Moving image to be registered (2D)
    fixed : np.ndarray
        Fixed (reference) image (2D)
    n_iterations : int, optional
        Maximum number of iterations (default: 100)
    alpha : float, optional
        Regularization parameter (default: 2.5)
    sigma_diffusion : float, optional
        Gaussian smoothing sigma for displacement field (default: 1.0)
    tolerance : float, optional
        Convergence tolerance on mean squared error (default: 1e-4)
    verbose : bool, optional
        Print progress information (default: True)

    Returns
    -------
    displacement_field : np.ndarray
        Final displacement field of shape (2, height, width)
    registered : np.ndarray
        Registered (warped) moving image

    Notes
    -----
    The algorithm iteratively:
    1. Warps the moving image with current displacement
    2. Computes the update from intensity difference and gradient
    3. Updates the displacement field
    4. Smooths the displacement field (diffusion)
    5. Checks for convergence

    Convergence is reached when the change in MSE between iterations
    falls below the tolerance.
    """
    # Normalize images to [0, 1]
    moving_norm = normalize_image(moving)
    fixed_norm = normalize_image(fixed)

    # Initialize displacement field to zero
    height, width = fixed.shape
    displacement_field = np.zeros((2, height, width), dtype=np.float32)

    # Track MSE for convergence
    prev_mse = float('inf')

    for iteration in range(n_iterations):
        # Perform one Demons step
        displacement_field = demons_step(
            moving_norm,
            fixed_norm,
            displacement_field,
            alpha=alpha,
            sigma_diffusion=sigma_diffusion
        )

        # Compute current MSE
        warped = warp_image(moving_norm, displacement_field)
        mse = np.mean((warped - fixed_norm) ** 2)

        # Check convergence
        mse_change = abs(prev_mse - mse)

        if verbose and (iteration % 10 == 0 or iteration == n_iterations - 1):
            print(f"  Iteration {iteration+1}/{n_iterations}: MSE = {mse:.6f}, Change = {mse_change:.6f}")

        if mse_change < tolerance:
            if verbose:
                print(f"  Converged at iteration {iteration+1}")
            break

        prev_mse = mse

    # Warp moving image with final displacement field
    registered = warp_image(moving, displacement_field)

    return displacement_field, registered


def multi_scale_demons(moving: np.ndarray,
                       fixed: np.ndarray,
                       scales: List[float] = [4, 2, 1],
                       n_iterations: int = 50,
                       alpha: float = 2.5,
                       sigma_diffusion: float = 1.0,
                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-scale (pyramid) Demons registration.

    Parameters
    ----------
    moving : np.ndarray
        Moving image to be registered (2D)
    fixed : np.ndarray
        Fixed (reference) image (2D)
    scales : list of float, optional
        List of downsampling factors from coarse to fine (default: [4, 2, 1])
        1 = original resolution
    n_iterations : int, optional
        Number of iterations per scale (default: 50)
    alpha : float, optional
        Regularization parameter (default: 2.5)
    sigma_diffusion : float, optional
        Gaussian smoothing sigma for displacement field (default: 1.0)
    verbose : bool, optional
        Print progress information (default: True)

    Returns
    -------
    displacement_field : np.ndarray
        Final displacement field at original resolution
    registered : np.ndarray
        Registered (warped) moving image

    Notes
    -----
    Multi-scale approach improves:
    1. Convergence speed (coarse scales capture large displacements)
    2. Robustness (less prone to local minima)
    3. Computational efficiency

    The displacement field is propagated from coarse to fine scales,
    with upsampling and refinement at each level.
    """
    height, width = fixed.shape
    displacement_field = None

    for scale_idx, scale in enumerate(scales):
        if verbose:
            print(f"\nScale {scale_idx+1}/{len(scales)}: downsampling factor = {scale}")

        # Downsample images
        if scale > 1:
            zoom_factor = 1.0 / scale
            moving_scaled = ndimage.zoom(moving, zoom_factor, order=1)
            fixed_scaled = ndimage.zoom(fixed, zoom_factor, order=1)
        else:
            moving_scaled = moving
            fixed_scaled = fixed

        # Upsample displacement field from previous scale
        if displacement_field is not None and scale > 1:
            # Upsample displacement field
            h_scaled, w_scaled = fixed_scaled.shape
            upsample_factor_y = h_scaled / displacement_field.shape[1]
            upsample_factor_x = w_scaled / displacement_field.shape[2]

            disp_y_upsampled = ndimage.zoom(displacement_field[0], upsample_factor_y, order=1)
            disp_x_upsampled = ndimage.zoom(displacement_field[1], upsample_factor_x, order=1)

            # Scale displacement magnitudes
            disp_y_upsampled *= upsample_factor_y
            disp_x_upsampled *= upsample_factor_x

            displacement_field = np.stack([disp_y_upsampled, disp_x_upsampled], axis=0)
        elif displacement_field is None:
            # Initialize at first scale
            h_scaled, w_scaled = fixed_scaled.shape
            displacement_field = np.zeros((2, h_scaled, w_scaled), dtype=np.float32)

        # Register at current scale
        displacement_field, _ = demons_registration(
            moving_scaled,
            fixed_scaled,
            n_iterations=n_iterations,
            alpha=alpha,
            sigma_diffusion=sigma_diffusion,
            verbose=verbose
        )

    # Warp moving image at original resolution with final displacement field
    registered = warp_image(moving, displacement_field)

    return displacement_field, registered


def apply_displacement_to_series(images: np.ndarray,
                                 displacement_field: np.ndarray) -> np.ndarray:
    """
    Apply displacement field to a series of images.

    Parameters
    ----------
    images : np.ndarray
        Array of images of shape (n_images, height, width)
    displacement_field : np.ndarray
        Displacement field of shape (2, height, width)

    Returns
    -------
    registered_series : np.ndarray
        Registered images of shape (n_images, height, width)
    """
    n_images = images.shape[0]
    registered_series = np.zeros_like(images)

    for i in range(n_images):
        registered_series[i] = warp_image(images[i], displacement_field)

    return registered_series


def extract_perfusion_curve(images: np.ndarray,
                           mask: Optional[np.ndarray] = None,
                           roi_coords: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Extract perfusion curve (mean intensity over time) from a region.

    Parameters
    ----------
    images : np.ndarray
        Temporal series of images (n_frames, height, width)
    mask : np.ndarray, optional
        Binary mask defining the region of interest (height, width)
        If None, use roi_coords
    roi_coords : tuple of int, optional
        ROI as (y_min, y_max, x_min, x_max)
        Only used if mask is None

    Returns
    -------
    curve : np.ndarray
        Perfusion curve (mean intensity at each time point)
        Shape: (n_frames,)

    Notes
    -----
    For renal perfusion, the curve shows:
    - Baseline (pre-contrast)
    - Rapid enhancement (arterial phase)
    - Gradual washout (venous/delayed phase)
    """
    n_frames = images.shape[0]
    curve = np.zeros(n_frames, dtype=np.float32)

    if mask is not None:
        # Use mask
        for i in range(n_frames):
            curve[i] = np.mean(images[i][mask > 0])
    elif roi_coords is not None:
        # Use rectangular ROI
        y_min, y_max, x_min, x_max = roi_coords
        for i in range(n_frames):
            curve[i] = np.mean(images[i][y_min:y_max, x_min:x_max])
    else:
        # Use whole image
        for i in range(n_frames):
            curve[i] = np.mean(images[i])

    return curve


if __name__ == "__main__":
    print("utils.py - Demons Registration and Hierarchical Clustering")
    print("This module provides functions for temporal series registration.")
    print("Use temporal_registration.py to run the complete pipeline.")
