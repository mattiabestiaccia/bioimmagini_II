"""
Utility per caricamento e gestione di immagini DICOM CT.

Gestisce la conversione corretta dei valori HU usando Rescale Intercept e Slope.
Integra il modulo dicom_import centralizzato con fallback locale.
"""

import numpy as np
import pydicom
from pathlib import Path
from typing import Tuple, Dict, Any
from scipy import ndimage
import sys

# Try to import centralized dicom_import module
try:
    project_root = Path(__file__).parent.parent.parent.parent.parent
    src_path = project_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        from dicom_import import read_dicom_series, extract_metadata
        DICOM_IMPORT_AVAILABLE = True
    else:
        DICOM_IMPORT_AVAILABLE = False
except ImportError:
    DICOM_IMPORT_AVAILABLE = False

from .exceptions import DICOMReadError, IsotropyError, ValidationError
from .types import Volume3D, DicomMetadata, VoxelSpacing

# Funzioni principali
def load_dicom_volume(series_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Carica un volume 3D DICOM da una directory.

    Applica correttamente Rescale Intercept e Slope per ottenere valori HU.
    Usa il modulo dicom_import centralizzato se disponibile, altrimenti fallback locale.

    Parameters
    ----------
    series_path : str
        Path alla directory contenente i file DICOM

    Returns
    -------
    volume : np.ndarray
        Volume 3D con valori HU corretti (dtype: float64)
    metadata : dict
        Metadata DICOM dalla prima slice (PixelSpacing, SliceThickness, etc.)

    Notes
    -----
    La conversione HU è:
        HU = RescaleIntercept + RescaleSlope * PixelValue

    Il volume è ordinato per Instance Number crescente.
    """
    # Try centralized module first
    if DICOM_IMPORT_AVAILABLE:
        try:
            return _load_dicom_volume_centralized(series_path)
        except Exception as e:
            print(f"Centralized DICOM loading failed: {e}")
            print("Falling back to local implementation...")

    # Fallback to local implementation
    return _load_dicom_volume_local(series_path)


def _load_dicom_volume_centralized(series_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load DICOM volume using centralized dicom_import module."""
    series_dir = Path(series_path)

    # Use centralized read_dicom_series
    volume, datasets = read_dicom_series(series_dir, sort_by_position=True)

    if len(datasets) == 0:
        raise DICOMReadError(str(series_dir), "No DICOM files found")

    # Extract metadata from first slice
    first_ds = datasets[0]

    metadata = {
        'PixelSpacing': [float(x) for x in first_ds.PixelSpacing] if hasattr(first_ds, 'PixelSpacing') else [1.0, 1.0],
        'SliceThickness': float(first_ds.SliceThickness) if hasattr(first_ds, 'SliceThickness') else None,
        'RescaleIntercept': float(first_ds.RescaleIntercept) if hasattr(first_ds, 'RescaleIntercept') else 0.0,
        'RescaleSlope': float(first_ds.RescaleSlope) if hasattr(first_ds, 'RescaleSlope') else 1.0,
        'RescaleType': first_ds.RescaleType if hasattr(first_ds, 'RescaleType') else None,
        'WindowCenter': first_ds.WindowCenter if hasattr(first_ds, 'WindowCenter') else None,
        'WindowWidth': first_ds.WindowWidth if hasattr(first_ds, 'WindowWidth') else None,
        'Rows': int(first_ds.Rows),
        'Columns': int(first_ds.Columns),
        'NumSlices': len(datasets),
    }

    # Calculate SliceThickness from SliceLocation if needed
    if metadata['SliceThickness'] is None and len(datasets) > 1:
        if hasattr(datasets[0], 'SliceLocation') and hasattr(datasets[1], 'SliceLocation'):
            metadata['SliceThickness'] = abs(
                float(datasets[1].SliceLocation) - float(datasets[0].SliceLocation)
            )

    print(f"Volume caricato (centralized): {volume.shape}")
    print(f"Spacing: {metadata['PixelSpacing']} mm (x,y), {metadata['SliceThickness']} mm (z)")
    print(f"Range HU: [{volume.min():.1f}, {volume.max():.1f}]")

    # Transpose from (Z, Y, X) to (Y, X, Z) to match local implementation and make_isotropic expectation
    volume = volume.transpose(1, 2, 0)
    print(f"Volume trasposto per compatibilità: {volume.shape}")

    return volume.astype(np.float64), metadata


def _load_dicom_volume_local(series_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load DICOM volume using local implementation (fallback)."""
    series_dir = Path(series_path)

    # Trova tutti i file DICOM
    dicom_files = sorted(series_dir.glob("*.dcm"))

    if not dicom_files:
        raise DICOMReadError(str(series_path), "No DICOM files found")

    # Leggi tutti i DICOM e ordina per Instance Number
    slices = []
    for dcm_file in dicom_files:
        ds = pydicom.dcmread(str(dcm_file))
        slices.append(ds)

    # Ordina per Instance Number (o SliceLocation se disponibile)
    if hasattr(slices[0], 'InstanceNumber'):
        slices.sort(key=lambda x: int(x.InstanceNumber))
    elif hasattr(slices[0], 'SliceLocation'):
        slices.sort(key=lambda x: float(x.SliceLocation))

    # Estrai metadata dalla prima slice
    first_slice = slices[0]

    metadata = {
        'PixelSpacing': [float(x) for x in first_slice.PixelSpacing],
        'SliceThickness': float(first_slice.SliceThickness) if hasattr(first_slice, 'SliceThickness') else None,
        'RescaleIntercept': float(first_slice.RescaleIntercept) if hasattr(first_slice, 'RescaleIntercept') else 0.0,
        'RescaleSlope': float(first_slice.RescaleSlope) if hasattr(first_slice, 'RescaleSlope') else 1.0,
        'RescaleType': first_slice.RescaleType if hasattr(first_slice, 'RescaleType') else None,
        'WindowCenter': first_slice.WindowCenter if hasattr(first_slice, 'WindowCenter') else None,
        'WindowWidth': first_slice.WindowWidth if hasattr(first_slice, 'WindowWidth') else None,
        'Rows': int(first_slice.Rows),
        'Columns': int(first_slice.Columns),
        'NumSlices': len(slices),
    }

    # Calcola SpacingBetweenSlices se non c'è SliceThickness
    if metadata['SliceThickness'] is None and len(slices) > 1:
        if hasattr(slices[0], 'SliceLocation') and hasattr(slices[1], 'SliceLocation'):
            metadata['SliceThickness'] = abs(
                float(slices[1].SliceLocation) - float(slices[0].SliceLocation)
            )

    # Crea volume 3D
    volume = np.zeros(
        (metadata['Rows'], metadata['Columns'], len(slices)),
        dtype=np.float64
    )

    # Carica ogni slice e applica rescaling
    intercept = metadata['RescaleIntercept']
    slope = metadata['RescaleSlope']

    for i, slice_ds in enumerate(slices):
        # Leggi pixel data
        pixel_array = slice_ds.pixel_array.astype(np.float64)

        # Applica rescaling: HU = intercept + slope * pixel_value
        hu_values = intercept + slope * pixel_array

        volume[:, :, i] = hu_values

    print(f"Volume caricato (local): {volume.shape}")
    print(f"Spacing: {metadata['PixelSpacing']} mm (x,y), {metadata['SliceThickness']} mm (z)")
    print(f"Range HU: [{volume.min():.1f}, {volume.max():.1f}]")
    print(f"Rescale: intercept={intercept}, slope={slope}, type={metadata['RescaleType']}")

    return volume, metadata


def make_isotropic(volume: np.ndarray, metadata: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    """
    Interpola un volume 3D per renderlo isotropo.

    Usa interpolazione trilineare (order=1 in scipy.ndimage.zoom).

    Parameters
    ----------
    volume : np.ndarray
        Volume 3D originale (anisotropo)
    metadata : dict
        Metadata con PixelSpacing e SliceThickness

    Returns
    -------
    iso_volume : np.ndarray
        Volume interpolato isotropo
    iso_spacing : float
        Spacing isotropo risultante (mm)

    Notes
    -----
    L'interpolazione viene fatta per ottenere voxel cubici.
    Lo spacing di riferimento è il più piccolo tra x, y, z per non perdere risoluzione.
    """
    # Estrai spacing originale
    pixel_spacing = metadata['PixelSpacing']  # [row_spacing, col_spacing] in mm
    slice_thickness = metadata['SliceThickness']  # in mm

    # In DICOM, PixelSpacing è [row, col] che corrisponde a [y, x]
    spacing_x = pixel_spacing[1]  # Column spacing
    spacing_y = pixel_spacing[0]  # Row spacing
    spacing_z = slice_thickness

    original_spacing = np.array([spacing_y, spacing_x, spacing_z])

    # Usa lo spacing minimo come riferimento (per non perdere risoluzione)
    target_spacing = original_spacing.min()

    # Calcola i fattori di zoom
    zoom_factors = original_spacing / target_spacing

    print(f"\nInterpolazione per volume isotropo:")
    print(f"  Spacing originale: {original_spacing} mm")
    print(f"  Spacing target (isotropo): {target_spacing} mm")
    print(f"  Zoom factors: {zoom_factors}")
    print(f"  Shape originale: {volume.shape}")

    # Applica interpolazione trilineare
    # order=1 è bilineare/trilineare, equivalente a MATLAB interp3 con 'linear'
    iso_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')

    print(f"  Shape isotropo: {iso_volume.shape}")
    print(f"  Range HU dopo interpolazione: [{iso_volume.min():.1f}, {iso_volume.max():.1f}]")

    return iso_volume, target_spacing


def check_isotropy(metadata: Dict[str, Any]) -> bool:
    """
    Verifica se un volume è già isotropo.

    Parameters
    ----------
    metadata : dict
        Metadata DICOM con spacing information

    Returns
    -------
    is_isotropic : bool
        True se il volume è isotropo (con tolleranza di 0.01 mm)
    """
    pixel_spacing = metadata['PixelSpacing']
    slice_thickness = metadata['SliceThickness']

    spacing_x = pixel_spacing[1]
    spacing_y = pixel_spacing[0]
    spacing_z = slice_thickness

    # Tolleranza per confronto float
    tolerance = 0.01

    is_iso = (
        abs(spacing_x - spacing_y) < tolerance and
        abs(spacing_x - spacing_z) < tolerance and
        abs(spacing_y - spacing_z) < tolerance
    )

    return is_iso
