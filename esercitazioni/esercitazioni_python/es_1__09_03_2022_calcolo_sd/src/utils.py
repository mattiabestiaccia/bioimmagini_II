"""
Utility functions for MRI noise analysis.

This module provides common functions for:
- Standard deviation map calculation
- Histogram analysis
- Statistical estimators
- Rayleigh correction
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


def compute_sd_map(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Compute standard deviation map using a local sliding window.

    Equivalent to MATLAB's stdfilt function.

    Parameters
    ----------
    image : np.ndarray
        Input image
    kernel_size : int, optional
        Size of the square kernel window (default: 5)

    Returns
    -------
    np.ndarray
        Standard deviation map with same shape as input

    Examples
    --------
    >>> img = np.random.randn(100, 100)
    >>> sd_map = compute_sd_map(img, kernel_size=5)
    """
    def local_std(values):
        return np.std(values)

    # Create footprint (kernel)
    footprint = np.ones((kernel_size, kernel_size))

    # Apply generic filter with std function
    sd_map = ndimage.generic_filter(
        image,
        local_std,
        footprint=footprint,
        mode='constant',
        cval=0.0
    )

    return sd_map


def estimate_sigma_from_histogram(values: np.ndarray, bins: int = 100) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate sigma from histogram maximum.

    Parameters
    ----------
    values : np.ndarray
        Array of values (e.g., flattened SD map)
    bins : int, optional
        Number of histogram bins (default: 100)

    Returns
    -------
    sigma_max : float
        Sigma estimated from histogram maximum
    hist : np.ndarray
        Histogram counts
    bin_centers : np.ndarray
        Bin center values

    Examples
    --------
    >>> data = np.random.randn(1000) * 5 + 10
    >>> sigma, hist, bins = estimate_sigma_from_histogram(data)
    """
    hist, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find maximum of histogram
    max_idx = np.argmax(hist)
    sigma_max = bin_centers[max_idx]

    return sigma_max, hist, bin_centers


def rayleigh_correction_factor() -> float:
    """
    Return the Rayleigh correction factor for noise estimation in MRI background.

    In MRI, the background noise follows a Rayleigh distribution rather than
    Gaussian. The correction factor is sqrt(2/(4-pi)) ≈ 1.526

    Returns
    -------
    float
        Correction factor (1.526)

    References
    ----------
    Constantinides, C. D., Atalar, E., & McVeigh, E. R. (1997).
    Signal-to-noise measurements in magnitude images from NMR phased arrays.
    """
    return np.sqrt(2.0 / (4.0 - np.pi))


def apply_rayleigh_correction(sd_background: float) -> float:
    """
    Apply Rayleigh correction to background standard deviation.

    Parameters
    ----------
    sd_background : float
        Standard deviation measured in background region

    Returns
    -------
    float
        Corrected standard deviation

    Examples
    --------
    >>> sd_bk = 10.5
    >>> sd_corrected = apply_rayleigh_correction(sd_bk)
    >>> print(f"Corrected: {sd_corrected:.2f}")
    """
    return sd_background * rayleigh_correction_factor()


def compute_statistics(values: np.ndarray) -> dict:
    """
    Compute comprehensive statistics on array values.

    Parameters
    ----------
    values : np.ndarray
        Input array

    Returns
    -------
    dict
        Dictionary containing mean, median, std, min, max

    Examples
    --------
    >>> data = np.random.randn(1000)
    >>> stats = compute_statistics(data)
    >>> print(f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    """
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values)
    }


def exclude_zero_padding(image: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Get indices of non-zero pixels to exclude zero padding.

    Parameters
    ----------
    image : np.ndarray
        Input image
    threshold : float, optional
        Threshold value (default: 0.0)

    Returns
    -------
    np.ndarray
        Boolean mask of pixels above threshold

    Examples
    --------
    >>> img = np.array([[0, 0, 1], [2, 3, 0]])
    >>> mask = exclude_zero_padding(img)
    >>> non_zero_values = img[mask]
    """
    return image > threshold


def create_synthetic_image(dim: int = 512, sigma_noise: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic image with multiple intensity patterns and Gaussian noise.

    Replicates the synthetic image from Calcolo_SD.m

    Parameters
    ----------
    dim : int, optional
        Image dimension (dim x dim) (default: 512)
    sigma_noise : float, optional
        Standard deviation of Gaussian noise to add (default: 5.0)

    Returns
    -------
    image_clean : np.ndarray
        Clean image without noise
    image_noisy : np.ndarray
        Image with added Gaussian noise

    Examples
    --------
    >>> clean, noisy = create_synthetic_image(dim=256, sigma_noise=3.0)
    >>> print(f"Shape: {noisy.shape}, Noise std: {np.std(noisy - clean):.2f}")
    """
    # Create base image
    image = np.ones((dim, dim)) * 50.0

    # Add patterns with different intensities
    image[50:101, 50:101] = 120
    image[101:181, 101:451] = 200
    image[200:501, 200:351] = 90
    image[230:271, 230:271] = 250
    image[5:401, 450:501] = 150

    # Add Gaussian noise
    noise = np.random.normal(0, sigma_noise, (dim, dim))
    image_noisy = image + noise

    return image, image_noisy
