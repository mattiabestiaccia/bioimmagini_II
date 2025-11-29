#!/usr/bin/env python3
"""
utils.py - Cardiac Function Analysis with Active Contours

This module implements cardiac function analysis for left ventricle (LV)
using Active Contours (Chan-Vese) segmentation on cardiac MRI cine images.

Esercitazione 4: Segmentazione Funzione Cardiaca
Dataset: Cardiac MRI (15 slices x 30 temporal frames = 450 DICOM images)

Author: Biomedical Imaging Course
Date: 2025
"""

import numpy as np
import pydicom
from pathlib import Path
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from skimage import filters, morphology
from scipy import ndimage
from typing import Tuple, List, Dict, Optional
import warnings


def load_cardiac_4d(dicom_dir: str) -> Tuple[np.ndarray, List[pydicom.Dataset], Dict]:
    """
    Load 4D cardiac MRI dataset (3D+T: slices + temporal frames).

    Parameters
    ----------
    dicom_dir : str
        Directory containing DICOM files

    Returns
    -------
    volume_4d : np.ndarray
        4D array of shape (n_frames, n_slices, height, width)
    datasets : list of pydicom.Dataset
        List of DICOM datasets
    metadata : dict
        Dictionary with metadata:
        - 'n_frames': Number of temporal frames
        - 'n_slices': Number of slices
        - 'pixel_spacing': (dy, dx) in mm
        - 'slice_thickness': dz in mm
        - 'trigger_times': Trigger times for each frame (ms)

    Notes
    -----
    The DICOM series is organized as:
    - Total images = n_slices * n_frames
    - Images with same ImagePositionPatient belong to same slice
    - Images with same TriggerTime belong to same temporal frame

    Alternative: Use "Cardiac Number of Images" field (30 in this dataset)
    to determine n_frames, then group by ImagePositionPatient.
    """
    dicom_path = Path(dicom_dir)

    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    # Load all DICOM files
    dicom_files = sorted(dicom_path.glob("*.dcm"))

    if not dicom_files:
        # Try without extension
        dicom_files = sorted([f for f in dicom_path.iterdir() if f.is_file()])

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    print(f"Found {len(dicom_files)} DICOM files")

    # Read all datasets
    datasets = []
    for dcm_file in dicom_files:
        ds = pydicom.dcmread(dcm_file)
        datasets.append(ds)

    # Determine number of frames (from first dataset)
    ds0 = datasets[0]
    if hasattr(ds0, 'CardiacNumberOfImages'):
        n_frames = int(ds0.CardiacNumberOfImages)
    else:
        # Fallback: count unique TriggerTime values
        trigger_times = set()
        for ds in datasets:
            if hasattr(ds, 'TriggerTime'):
                trigger_times.add(float(ds.TriggerTime))
        n_frames = len(trigger_times)

    print(f"Number of temporal frames: {n_frames}")

    # Group images by ImagePositionPatient to find slices
    from collections import defaultdict
    position_groups = defaultdict(list)

    for i, ds in enumerate(datasets):
        if hasattr(ds, 'ImagePositionPatient'):
            # Use Z-coordinate (3rd element) as key
            z_pos = round(float(ds.ImagePositionPatient[2]), 2)
            position_groups[z_pos].append((i, ds))

    # Sort slices by Z position
    sorted_positions = sorted(position_groups.keys())
    n_slices = len(sorted_positions)

    print(f"Number of slices: {n_slices}")

    # Get image dimensions from first dataset
    height, width = ds0.pixel_array.shape

    # Allocate 4D volume
    volume_4d = np.zeros((n_frames, n_slices, height, width), dtype=np.float32)

    # Fill volume_4d
    # For each slice, sort images by TriggerTime
    for slice_idx, z_pos in enumerate(sorted_positions):
        slice_datasets = position_groups[z_pos]

        # Sort by TriggerTime
        slice_datasets_sorted = sorted(slice_datasets,
                                       key=lambda x: float(x[1].TriggerTime) if hasattr(x[1], 'TriggerTime') else 0)

        for frame_idx, (orig_idx, ds) in enumerate(slice_datasets_sorted[:n_frames]):
            volume_4d[frame_idx, slice_idx] = ds.pixel_array.astype(np.float32)

    # Extract metadata
    pixel_spacing = (float(ds0.PixelSpacing[0]), float(ds0.PixelSpacing[1]))  # (dy, dx)

    if hasattr(ds0, 'SliceThickness'):
        slice_thickness = float(ds0.SliceThickness)
    else:
        # Estimate from ImagePositionPatient
        if n_slices > 1:
            slice_thickness = abs(sorted_positions[1] - sorted_positions[0])
        else:
            slice_thickness = 1.0
            warnings.warn("Could not determine slice thickness, using 1.0 mm")

    # Extract trigger times
    trigger_times = []
    for frame_idx in range(n_frames):
        # Get trigger time from first slice of this frame
        for z_pos in sorted_positions:
            slice_datasets = position_groups[z_pos]
            slice_datasets_sorted = sorted(slice_datasets,
                                          key=lambda x: float(x[1].TriggerTime) if hasattr(x[1], 'TriggerTime') else 0)
            if frame_idx < len(slice_datasets_sorted):
                ds = slice_datasets_sorted[frame_idx][1]
                if hasattr(ds, 'TriggerTime'):
                    trigger_times.append(float(ds.TriggerTime))
                    break

    metadata = {
        'n_frames': n_frames,
        'n_slices': n_slices,
        'pixel_spacing': pixel_spacing,
        'slice_thickness': slice_thickness,
        'trigger_times': trigger_times if trigger_times else None
    }

    print(f"Loaded 4D volume: {volume_4d.shape}")
    print(f"Pixel spacing: {pixel_spacing[0]:.2f} x {pixel_spacing[1]:.2f} mm")
    print(f"Slice thickness: {slice_thickness:.2f} mm")

    return volume_4d, datasets, metadata


def find_cardiac_phases(volume_4d: np.ndarray,
                       trigger_times: Optional[List[float]] = None,
                       target_diastolic_time: float = 693.0,
                       target_systolic_time: float = 288.0) -> Tuple[int, int]:
    """
    Find diastolic and systolic phases from 4D cardiac volume.

    Parameters
    ----------
    volume_4d : np.ndarray
        4D cardiac volume (n_frames, n_slices, height, width)
    trigger_times : list of float, optional
        Trigger times for each frame (ms)
        If None, find phases by volume estimation
    target_diastolic_time : float, optional
        Target diastolic trigger time in ms (default: 693, frame 28)
    target_systolic_time : float, optional
        Target systolic trigger time in ms (default: 288, frame 12)

    Returns
    -------
    diastolic_frame : int
        Index of diastolic frame (maximum volume)
    systolic_frame : int
        Index of systolic frame (minimum volume)

    Notes
    -----
    Diastole: Maximum ventricular volume (relaxation, filling)
    Systole: Minimum ventricular volume (contraction, ejection)

    If trigger_times available, match to target times from report.
    Otherwise, estimate from intensity in central region (LV cavity).
    """
    n_frames = volume_4d.shape[0]

    if trigger_times is not None and len(trigger_times) == n_frames:
        # Find frames closest to target times
        trigger_times_array = np.array(trigger_times)

        diastolic_frame = np.argmin(np.abs(trigger_times_array - target_diastolic_time))
        systolic_frame = np.argmin(np.abs(trigger_times_array - target_systolic_time))

        print(f"Found frames by trigger time:")
        print(f"  Diastolic: frame {diastolic_frame} (trigger time: {trigger_times[diastolic_frame]:.1f} ms)")
        print(f"  Systolic: frame {systolic_frame} (trigger time: {trigger_times[systolic_frame]:.1f} ms)")
    else:
        # Estimate by central region intensity (LV cavity is bright in T1-weighted)
        # Higher intensity in diastole (larger cavity)
        print("Trigger times not available, estimating phases by intensity...")

        center_intensities = []
        for frame_idx in range(n_frames):
            # Use middle slices (5-10)
            mid_slices = volume_4d[frame_idx, 5:11]

            # Central region (approximate LV location)
            h, w = mid_slices.shape[1], mid_slices.shape[2]
            center_region = mid_slices[:, h//3:2*h//3, w//3:2*w//3]

            # Mean intensity in central region
            center_intensities.append(np.mean(center_region))

        # Diastole: maximum intensity (larger cavity)
        # Systole: minimum intensity (smaller cavity)
        diastolic_frame = np.argmax(center_intensities)
        systolic_frame = np.argmin(center_intensities)

        print(f"Estimated frames by intensity:")
        print(f"  Diastolic: frame {diastolic_frame}")
        print(f"  Systolic: frame {systolic_frame}")

    return diastolic_frame, systolic_frame


def create_circular_seed(image_shape: Tuple[int, int],
                        center: Optional[Tuple[int, int]] = None,
                        radius: int = 30) -> np.ndarray:
    """
    Create circular seed mask for active contour initialization.

    Parameters
    ----------
    image_shape : tuple of int
        Image shape (height, width)
    center : tuple of int, optional
        Seed center (y, x). If None, use image center
    radius : int, optional
        Seed radius in pixels (default: 30)

    Returns
    -------
    seed_mask : np.ndarray
        Binary seed mask (height, width)
    """
    height, width = image_shape

    if center is None:
        center = (height // 2, width // 2)

    cy, cx = center

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Distance from center
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)

    # Create circular mask
    seed_mask = (dist <= radius).astype(np.uint8)

    return seed_mask


def segment_lv_active_contour(image: np.ndarray,
                              seed_mask: np.ndarray,
                              n_iterations: int = 100,
                              smoothing: float = 2.0,
                              lambda1: float = 1.0,
                              lambda2: float = 1.0) -> np.ndarray:
    """
    Segment left ventricle using Chan-Vese active contours.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D)
    seed_mask : np.ndarray
        Initial contour as binary mask
    n_iterations : int, optional
        Number of iterations (default: 100)
    smoothing : float, optional
        Smoothing factor (higher = smoother contours) (default: 2.0)
    lambda1 : float, optional
        Weight for fitting inside contour (default: 1.0)
    lambda2 : float, optional
        Weight for fitting outside contour (default: 1.0)

    Returns
    -------
    segmentation_mask : np.ndarray
        Binary segmentation mask

    Notes
    -----
    This uses morphological_chan_vese from scikit-image, which implements
    the Chan-Vese model optimized for convergence speed and accuracy.

    The algorithm minimizes:
    E = lambda1 * (inside - c1)^2 + lambda2 * (outside - c2)^2 + mu * |contour_length|

    Where c1, c2 are mean intensities inside/outside contour.

    In MATLAB this corresponds to:
    activecontour(image, seed_mask, n_iterations, 'Chan-Vese', ...
                  'SmoothFactor', smoothing, 'ContractionBias', 0)
    """
    # Normalize image to [0, 1]
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)

    # Apply morphological Chan-Vese
    segmentation_mask = morphological_chan_vese(
        image_norm,
        num_iter=n_iterations,
        init_level_set=seed_mask,
        smoothing=int(smoothing),
        lambda1=lambda1,
        lambda2=lambda2
    )

    return segmentation_mask.astype(np.uint8)


def refine_segmentation(mask: np.ndarray,
                       min_area: int = 100,
                       fill_holes: bool = True,
                       smooth_iterations: int = 2) -> np.ndarray:
    """
    Refine segmentation mask by removing small components and filling holes.

    Parameters
    ----------
    mask : np.ndarray
        Binary segmentation mask
    min_area : int, optional
        Minimum area for connected components (default: 100 pixels)
    fill_holes : bool, optional
        Fill holes in segmentation (default: True)
    smooth_iterations : int, optional
        Number of morphological smoothing iterations (default: 2)

    Returns
    -------
    refined_mask : np.ndarray
        Refined binary mask
    """
    refined_mask = mask.copy()

    # Remove small connected components
    labeled, num_features = ndimage.label(refined_mask)
    component_sizes = ndimage.sum(refined_mask, labeled, range(1, num_features + 1))

    # Keep only components larger than min_area
    for i, size in enumerate(component_sizes, start=1):
        if size < min_area:
            refined_mask[labeled == i] = 0

    # Fill holes
    if fill_holes:
        refined_mask = ndimage.binary_fill_holes(refined_mask).astype(np.uint8)

    # Morphological smoothing (closing then opening)
    if smooth_iterations > 0:
        struct = morphology.disk(1)
        for _ in range(smooth_iterations):
            refined_mask = ndimage.binary_closing(refined_mask, structure=struct).astype(np.uint8)
            refined_mask = ndimage.binary_opening(refined_mask, structure=struct).astype(np.uint8)

    return refined_mask


def compute_volume_from_masks(masks: np.ndarray,
                              pixel_spacing: Tuple[float, float],
                              slice_thickness: float) -> float:
    """
    Compute ventricular volume from segmentation masks.

    Parameters
    ----------
    masks : np.ndarray
        3D array of binary masks (n_slices, height, width)
    pixel_spacing : tuple of float
        Pixel spacing (dy, dx) in mm
    slice_thickness : float
        Slice thickness (dz) in mm

    Returns
    -------
    volume : float
        Ventricular volume in mL

    Notes
    -----
    Volume calculation from PDF:
    V = sum(A_i) * dx * dy * dz

    Where A_i is the endocardial area on slice i.
    Result is converted from mm^3 to mL (1 mL = 1000 mm^3).
    """
    # Sum areas across slices
    total_area_pixels = np.sum(masks)

    # Convert to mm^2
    dy, dx = pixel_spacing
    pixel_area_mm2 = dx * dy
    total_area_mm2 = total_area_pixels * pixel_area_mm2

    # Multiply by slice thickness to get volume in mm^3
    volume_mm3 = total_area_mm2 * slice_thickness

    # Convert to mL
    volume_ml = volume_mm3 / 1000.0

    return volume_ml


def calculate_bsa(weight_kg: float, height_cm: float, method: str = 'mosteller') -> float:
    """
    Calculate Body Surface Area (BSA).

    Parameters
    ----------
    weight_kg : float
        Weight in kg
    height_cm : float
        Height in cm
    method : str, optional
        Formula to use: 'mosteller' (default), 'dubois', 'haycock'

    Returns
    -------
    bsa : float
        Body surface area in m^2

    Notes
    -----
    Mosteller formula (used in report):
    BSA = sqrt((height_cm * weight_kg) / 3600)

    DuBois formula:
    BSA = 0.007184 * height_cm^0.725 * weight_kg^0.425

    Haycock formula:
    BSA = 0.024265 * height_cm^0.3964 * weight_kg^0.5378
    """
    if method == 'mosteller':
        bsa = np.sqrt((height_cm * weight_kg) / 3600.0)
    elif method == 'dubois':
        bsa = 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)
    elif method == 'haycock':
        bsa = 0.024265 * (height_cm ** 0.3964) * (weight_kg ** 0.5378)
    else:
        raise ValueError(f"Unknown BSA method: {method}")

    return bsa


def calculate_cardiac_parameters(edlv: float,
                                 eslv: float,
                                 heart_rate: float,
                                 bsa: float) -> Dict[str, float]:
    """
    Calculate cardiac function parameters.

    Parameters
    ----------
    edlv : float
        End-diastolic left ventricular volume (mL)
    eslv : float
        End-systolic left ventricular volume (mL)
    heart_rate : float
        Heart rate (beats per minute)
    bsa : float
        Body surface area (m^2)

    Returns
    -------
    parameters : dict
        Dictionary with cardiac parameters:
        - 'stroke_volume': SV = EDLV - ESLV (mL)
        - 'ejection_fraction': EF = (EDLV - ESLV) / EDLV (%)
        - 'cardiac_output': CO = SV * HR (L/min)
        - 'edlv_indexed': EDLV / BSA (mL/m^2)
        - 'eslv_indexed': ESLV / BSA (mL/m^2)
        - 'sv_indexed': SV / BSA (mL/m^2)

    Notes
    -----
    From PDF report:
    - Stroke Volume (SV) = EDLV - ESLV
    - Ejection Fraction (EF) = (EDLV - ESLV) / EDLV * 100
    - Cardiac Output (Gittata Cardiaca) = SV * HR / 1000 (L/min)
    - Indexed values normalize for body size (BSA)
    """
    # Stroke volume
    stroke_volume = edlv - eslv

    # Ejection fraction (percentage)
    ejection_fraction = (stroke_volume / edlv) * 100.0 if edlv > 0 else 0.0

    # Cardiac output (L/min)
    cardiac_output = (stroke_volume * heart_rate) / 1000.0

    # Indexed values (normalized by BSA)
    edlv_indexed = edlv / bsa if bsa > 0 else 0.0
    eslv_indexed = eslv / bsa if bsa > 0 else 0.0
    sv_indexed = stroke_volume / bsa if bsa > 0 else 0.0

    parameters = {
        'stroke_volume': stroke_volume,
        'ejection_fraction': ejection_fraction,
        'cardiac_output': cardiac_output,
        'edlv_indexed': edlv_indexed,
        'eslv_indexed': eslv_indexed,
        'sv_indexed': sv_indexed
    }

    return parameters


def generate_cardiac_report(edlv: float,
                           eslv: float,
                           weight_kg: float,
                           height_cm: float,
                           heart_rate: float,
                           diastolic_frame: int,
                           systolic_frame: int,
                           diastolic_time: Optional[float] = None,
                           systolic_time: Optional[float] = None) -> str:
    """
    Generate cardiac function report similar to PDF report.

    Parameters
    ----------
    edlv : float
        End-diastolic LV volume (mL)
    eslv : float
        End-systolic LV volume (mL)
    weight_kg : float
        Patient weight (kg)
    height_cm : float
        Patient height (cm)
    heart_rate : float
        Heart rate (bpm)
    diastolic_frame : int
        Diastolic frame index
    systolic_frame : int
        Systolic frame index
    diastolic_time : float, optional
        Diastolic trigger time (ms)
    systolic_time : float, optional
        Systolic trigger time (ms)

    Returns
    -------
    report : str
        Formatted cardiac function report
    """
    # Calculate BSA
    bsa = calculate_bsa(weight_kg, height_cm, method='mosteller')
    bmi = weight_kg / ((height_cm / 100.0) ** 2)

    # Calculate cardiac parameters
    params = calculate_cardiac_parameters(edlv, eslv, heart_rate, bsa)

    # Format report
    report = f"""
{'='*70}
CARDIAC FUNCTION ANALYSIS - LEFT VENTRICLE
{'='*70}

CARDIAC PHASES:
  Diastolic Phase: Frame {diastolic_frame}""" + (f" ({diastolic_time:.1f} ms)" if diastolic_time else "") + f"""
  Systolic Phase:  Frame {systolic_frame}""" + (f" ({systolic_time:.1f} ms)" if systolic_time else "") + f"""

VENTRICULAR VOLUMES:
  ED Volume (LV):     {edlv:.0f} mL
  ES Volume (LV):     {eslv:.0f} mL
  Stroke Volume (LV): {params['stroke_volume']:.0f} mL

PATIENT DATA:
  Weight:              {weight_kg:.0f} kg
  Height:              {height_cm:.0f} cm
  BMI:                 {bmi:.2f}
  Heart Rate:          {heart_rate:.0f} bpm
  Body Surface Area:   {bsa:.5f} m^2

INDEXED VALUES (normalized by BSA):
  ED Volume / BSA:     {params['edlv_indexed']:.0f} mL/m^2
  ES Volume / BSA:     {params['eslv_indexed']:.0f} mL/m^2
  Stroke Volume / BSA: {params['sv_indexed']:.0f} mL/m^2

CARDIAC FUNCTION:
  Cardiac Output (LV): {params['cardiac_output']:.5f} L/min
  Ejection Fraction:   {params['ejection_fraction']:.0f} %

{'='*70}

REFERENCE RANGES (normal adult):
  EDLV:   70-180 mL (male), 60-140 mL (female)
  ESLV:   25-70 mL (male), 20-55 mL (female)
  EF:     55-70%
  SV:     60-100 mL
  CO:     4-8 L/min

{'='*70}
"""

    return report


if __name__ == "__main__":
    print("utils.py - Cardiac Function Analysis with Active Contours")
    print("This module provides functions for LV segmentation and parameter calculation.")
    print("Use cardiac_function_analysis.py to run the complete pipeline.")
