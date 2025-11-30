#!/usr/bin/env python3
"""
utils.py - T2* Parametric Mapping and Curve Fitting

This module implements T2* relaxometry for iron overload quantification
using multi-echo Gradient Echo (GRE) MRI sequences.

Esercitazione 9: Mappe Parametriche T2*
Dataset: 2 patients with multi-echo GRE (10 echoes each)

Author: Biomedical Imaging Course
Date: 2025
"""

import numpy as np
import pydicom
from pathlib import Path
from scipy.optimize import curve_fit, least_squares
from typing import Tuple, List, Dict, Optional
import warnings


def load_multiecho_series(dicom_dir: str) -> Tuple[np.ndarray, np.ndarray, List[pydicom.Dataset]]:
    """
    Load multi-echo DICOM series for T2* mapping.

    Parameters
    ----------
    dicom_dir : str
        Directory containing multi-echo DICOM files

    Returns
    -------
    volume : np.ndarray
        3D array of shape (n_echoes, height, width)
    echo_times : np.ndarray
        Array of echo times in milliseconds (n_echoes,)
    datasets : list of pydicom.Dataset
        List of DICOM datasets for metadata access

    Notes
    -----
    Multi-echo sequences acquire multiple images at different TE values
    to sample the T2* decay curve. Typical TE range: 1-20 ms for cardiac/liver.
    """
    dicom_path = Path(dicom_dir)

    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    # Get all DICOM files
    dicom_files = sorted(dicom_path.glob("*.dcm"))

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    n_echoes = len(dicom_files)
    print(f"Found {n_echoes} echo images")

    # Read first file to get dimensions
    ds0 = pydicom.dcmread(dicom_files[0])
    height, width = ds0.pixel_array.shape

    # Allocate volume and echo times array
    volume = np.zeros((n_echoes, height, width), dtype=np.float32)
    echo_times = np.zeros(n_echoes, dtype=np.float32)
    datasets = []

    # Load all echoes
    for i, dcm_file in enumerate(dicom_files):
        ds = pydicom.dcmread(dcm_file)
        volume[i] = ds.pixel_array.astype(np.float32)

        # Extract echo time (EchoTime tag)
        if hasattr(ds, 'EchoTime'):
            echo_times[i] = float(ds.EchoTime)
        else:
            warnings.warn(f"EchoTime not found in {dcm_file.name}, using index")
            echo_times[i] = i * 2.0  # Fallback: assume 2ms spacing

        datasets.append(ds)

    # Sort by echo time
    sort_indices = np.argsort(echo_times)
    volume = volume[sort_indices]
    echo_times = echo_times[sort_indices]
    datasets = [datasets[i] for i in sort_indices]

    print(f"Echo times (ms): {echo_times}")
    print(f"Volume shape: {volume.shape}")

    return volume, echo_times, datasets


def model_s_exp(te: np.ndarray, S0: float, R2star: float) -> np.ndarray:
    """
    Single-exponential (S-EXP) model for T2* decay.

    Parameters
    ----------
    te : np.ndarray
        Echo times in milliseconds
    S0 : float
        Signal at TE=0 (initial magnetization)
    R2star : float
        R2* relaxation rate (1/T2*) in 1/ms

    Returns
    -------
    signal : np.ndarray
        Predicted signal at each TE

    Notes
    -----
    Model: S(TE) = S0 * exp(-TE * R2*)

    Equivalent to: S(TE) = S0 * exp(-TE / T2*)
    where T2* = 1 / R2*

    This model has 2 parameters and is appropriate when signal >> noise
    at all echo times.
    """
    return S0 * np.exp(-te * R2star)


def model_c_exp(te: np.ndarray, S0: float, R2star: float, C: float) -> np.ndarray:
    """
    Single-exponential plus constant (C-EXP) model for T2* decay.

    Parameters
    ----------
    te : np.ndarray
        Echo times in milliseconds
    S0 : float
        Signal at TE=0 (initial magnetization)
    R2star : float
        R2* relaxation rate (1/T2*) in 1/ms
    C : float
        Constant offset (noise floor + blood signal plateau)

    Returns
    -------
    signal : np.ndarray
        Predicted signal at each TE

    Notes
    -----
    Model: S(TE) = S0 * exp(-TE * R2*) + C

    This model has 3 parameters and accounts for:
    - Rician noise floor (non-zero mean after magnitude operation)
    - Contribution from tissues with very long T2* (e.g., oxygenated blood)
    - Partial volume effects (PVE) with multiple tissue types

    More robust for low SNR and severe iron overload (very fast decay).
    """
    return S0 * np.exp(-te * R2star) + C


def fit_t2star_pixel(signal: np.ndarray,
                    echo_times: np.ndarray,
                    model: str = 'c-exp',
                    initial_guess: Optional[np.ndarray] = None,
                    bounds: Optional[Tuple] = None) -> Tuple[np.ndarray, float]:
    """
    Fit T2* model to signal decay curve for a single pixel.

    Parameters
    ----------
    signal : np.ndarray
        Signal intensities at each echo time (n_echoes,)
    echo_times : np.ndarray
        Echo times in milliseconds (n_echoes,)
    model : str, optional
        Model type: 's-exp' (2 params) or 'c-exp' (3 params, default)
    initial_guess : np.ndarray, optional
        Initial parameter guess. If None, auto-estimated
    bounds : tuple, optional
        (lower_bounds, upper_bounds) for parameters
        If None, use default bounds

    Returns
    -------
    params : np.ndarray
        Fitted parameters:
        - s-exp: [S0, R2star]
        - c-exp: [S0, R2star, C]
    rmse : float
        Root mean squared error of fit

    Notes
    -----
    Uses scipy.optimize.curve_fit with Levenberg-Marquardt algorithm.

    For robust fitting:
    - Initial guess estimated from data
    - Bounds constrain parameters to physical ranges
    - RMSE quantifies goodness-of-fit
    """
    # Validate inputs
    if len(signal) != len(echo_times):
        raise ValueError("Signal and echo_times must have same length")

    if np.any(signal < 0):
        warnings.warn("Negative signal values detected, clipping to 0")
        signal = np.maximum(signal, 0)

    # Auto-estimate initial guess if not provided
    if initial_guess is None:
        S0_est = np.max(signal)

        # Estimate R2* from log-linear fit (if possible)
        if np.all(signal > 0):
            # Log-linear: ln(S) = ln(S0) - R2* * TE
            log_signal = np.log(signal + 1e-10)

            # Simple linear regression
            A = np.vstack([echo_times, np.ones(len(echo_times))]).T
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, log_signal, rcond=None)
                R2star_est = -coeffs[0]  # Negative slope
                R2star_est = np.clip(R2star_est, 0.01, 2.0)  # Reasonable range
            except:
                R2star_est = 0.05  # Default: T2* ~20ms
        else:
            R2star_est = 0.05

        if model == 's-exp':
            initial_guess = np.array([S0_est, R2star_est])
        else:  # c-exp
            C_est = np.min(signal)
            initial_guess = np.array([S0_est, R2star_est, C_est])

    # Set bounds if not provided
    if bounds is None:
        if model == 's-exp':
            # S0: [0, 2*max_signal], R2*: [0.01, 2.0] (T2*: 0.5-100ms)
            lower_bounds = [0, 0.01]
            upper_bounds = [2 * np.max(signal), 2.0]
        else:  # c-exp
            # S0: [0, 2*max], R2*: [0.01, 2.0], C: [0, max]
            lower_bounds = [0, 0.01, 0]
            upper_bounds = [2 * np.max(signal), 2.0, np.max(signal)]

        bounds = (lower_bounds, upper_bounds)

    # Select model function
    if model == 's-exp':
        model_func = model_s_exp
    elif model == 'c-exp':
        model_func = model_c_exp
    else:
        raise ValueError(f"Unknown model: {model}. Use 's-exp' or 'c-exp'")

    # Perform curve fitting
    try:
        params, _ = curve_fit(
            model_func,
            echo_times,
            signal,
            p0=initial_guess,
            bounds=bounds,
            method='trf',  # Trust Region Reflective
            maxfev=1000
        )

        # Compute RMSE
        fitted_signal = model_func(echo_times, *params)
        rmse = np.sqrt(np.mean((signal - fitted_signal) ** 2))

    except RuntimeError:
        # Fitting failed, return initial guess with high RMSE
        warnings.warn("Fitting failed, returning initial guess")
        params = initial_guess
        rmse = np.inf

    return params, rmse


def create_t2star_map(volume: np.ndarray,
                     echo_times: np.ndarray,
                     model: str = 'c-exp',
                     mask: Optional[np.ndarray] = None,
                     threshold: float = 10.0,
                     verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create T2* parametric map from multi-echo volume.

    Parameters
    ----------
    volume : np.ndarray
        Multi-echo volume (n_echoes, height, width)
    echo_times : np.ndarray
        Echo times in milliseconds (n_echoes,)
    model : str, optional
        Model type: 's-exp' or 'c-exp' (default)
    mask : np.ndarray, optional
        Binary mask (height, width). If None, threshold-based mask used
    threshold : float, optional
        Signal threshold for auto-masking (default: 10.0)
        Pixels with max signal < threshold are skipped
    verbose : bool, optional
        Print progress information (default: True)

    Returns
    -------
    t2star_map : np.ndarray
        T2* map in milliseconds (height, width)
    rmse_map : np.ndarray
        RMSE map (fitting error) (height, width)
    r2star_map : np.ndarray
        R2* map in 1/ms (height, width)

    Notes
    -----
    This function fits the T2* decay model to each pixel's signal-time curve.

    Computational cost: O(height * width * n_echoes * n_iterations)
    For 256x256 image with 10 echoes: ~60k fitting operations
    Typical runtime: 1-5 minutes (Python), can be reduced with ROI masking

    Output maps:
    - T2* map: Tissue-specific relaxation time (ms)
    - RMSE map: Goodness-of-fit (lower = better model fit)
    - R2* map: Relaxation rate (1/ms), inverse of T2*
    """
    n_echoes, height, width = volume.shape

    if verbose:
        print(f"\nCreating T2* map using {model.upper()} model")
        print(f"Image size: {height} x {width}")
        print(f"Number of echoes: {n_echoes}")

    # Create auto-mask if not provided
    if mask is None:
        # Use maximum signal across echoes
        max_signal = np.max(volume, axis=0)
        mask = max_signal > threshold

        if verbose:
            n_masked = np.sum(mask)
            n_total = height * width
            print(f"Auto-mask: {n_masked}/{n_total} pixels ({n_masked/n_total*100:.1f}%)")

    # Initialize output maps
    r2star_map = np.zeros((height, width), dtype=np.float32)
    t2star_map = np.zeros((height, width), dtype=np.float32)
    rmse_map = np.zeros((height, width), dtype=np.float32)

    # Count for progress
    n_pixels = np.sum(mask)
    processed = 0

    # Iterate over pixels
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue

            # Extract signal-time curve for this pixel
            signal = volume[:, y, x]

            # Fit T2* model
            params, rmse = fit_t2star_pixel(signal, echo_times, model=model)

            # Extract R2* (second parameter in both models)
            R2star = params[1]

            # Convert to T2* (ms)
            if R2star > 0:
                T2star = 1.0 / R2star
            else:
                T2star = 0.0

            # Store results
            r2star_map[y, x] = R2star
            t2star_map[y, x] = T2star
            rmse_map[y, x] = rmse

            # Progress update
            processed += 1
            if verbose and processed % 1000 == 0:
                print(f"  Processed {processed}/{n_pixels} pixels ({processed/n_pixels*100:.1f}%)")

    if verbose:
        print(f"  Completed: {processed}/{n_pixels} pixels")

        # Statistics (only on masked region)
        t2star_masked = t2star_map[mask]
        print(f"\nT2* statistics (masked region):")
        print(f"  Mean: {np.mean(t2star_masked):.2f} ms")
        print(f"  Median: {np.median(t2star_masked):.2f} ms")
        print(f"  Std: {np.std(t2star_masked):.2f} ms")
        print(f"  Range: [{np.min(t2star_masked):.2f}, {np.max(t2star_masked):.2f}] ms")

    return t2star_map, rmse_map, r2star_map


def compute_roi_statistics(map: np.ndarray, roi_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics in a region of interest (ROI).

    Parameters
    ----------
    map : np.ndarray
        Parametric map (e.g., T2* map)
    roi_mask : np.ndarray
        Binary ROI mask (same shape as map)

    Returns
    -------
    stats : dict
        Dictionary with statistics:
        - 'mean': Mean value in ROI
        - 'median': Median value in ROI
        - 'std': Standard deviation in ROI
        - 'min': Minimum value in ROI
        - 'max': Maximum value in ROI
        - 'n_pixels': Number of pixels in ROI

    Notes
    -----
    Used for clinical measurements (e.g., mean T2* in cardiac septum or liver).
    """
    roi_values = map[roi_mask > 0]

    if len(roi_values) == 0:
        warnings.warn("Empty ROI, returning NaN statistics")
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'n_pixels': 0
        }

    stats = {
        'mean': float(np.mean(roi_values)),
        'median': float(np.median(roi_values)),
        'std': float(np.std(roi_values)),
        'min': float(np.min(roi_values)),
        'max': float(np.max(roi_values)),
        'n_pixels': len(roi_values)
    }

    return stats


def normalize_rmse_map(rmse_map: np.ndarray, signal_map: np.ndarray) -> np.ndarray:
    """
    Normalize RMSE map by signal intensity (percentage error).

    Parameters
    ----------
    rmse_map : np.ndarray
        Absolute RMSE map
    signal_map : np.ndarray
        Reference signal map (e.g., S0 or max signal)

    Returns
    -------
    rmse_percent : np.ndarray
        RMSE as percentage of signal (%)

    Notes
    -----
    Percentage RMSE = (RMSE / Signal) * 100

    This normalization allows comparison of fitting quality across
    regions with different signal intensities.
    """
    # Avoid division by zero
    signal_map_safe = np.maximum(signal_map, 1e-10)

    rmse_percent = (rmse_map / signal_map_safe) * 100.0

    return rmse_percent


def estimate_iron_concentration(t2star_ms: float, organ: str = 'liver') -> float:
    """
    Estimate iron concentration from T2* value using calibration curves.

    Parameters
    ----------
    t2star_ms : float
        T2* value in milliseconds
    organ : str, optional
        Organ type: 'liver' or 'heart' (default: 'liver')

    Returns
    -------
    iron_concentration : float
        Estimated iron concentration in mg Fe/g dry weight (liver)
        or arbitrary units (heart)

    Notes
    -----
    Calibration curves from literature:

    Liver (Wood et al. 2005):
    LIC (mg Fe/g) = 0.0254 + 0.202 / T2* (ms)

    Heart: No validated calibration, but clinical thresholds:
    - T2* > 20 ms: Normal
    - T2* 10-20 ms: Mild overload
    - T2* 6-10 ms: Moderate overload
    - T2* < 6 ms: Severe overload (high risk)

    References
    ----------
    Wood JC, et al. (2005) "MRI R2 and R2* mapping accurately estimates
    hepatic iron concentration in transfusion-dependent thalassemia..."
    Blood 106(4):1460-5
    """
    if organ == 'liver':
        # Wood et al. 2005 calibration
        if t2star_ms > 0:
            lic = 0.0254 + 0.202 / t2star_ms
        else:
            lic = np.inf
        return lic
    elif organ == 'heart':
        # No validated calibration, return T2* as-is
        # Clinical interpretation based on thresholds above
        return t2star_ms
    else:
        raise ValueError(f"Unknown organ: {organ}. Use 'liver' or 'heart'")


if __name__ == "__main__":
    print("utils.py - T2* Parametric Mapping and Curve Fitting")
    print("This module provides functions for T2* relaxometry.")
    print("Use t2star_mapping.py to run the complete pipeline.")
