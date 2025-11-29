"""
Funzioni per la validazione di dati DICOM.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from pydicom.dataset import FileDataset


class DicomValidationError(Exception):
    """Eccezione sollevata quando la validazione DICOM fallisce."""
    pass


def validate_dicom_dataset(ds: FileDataset, strict: bool = False) -> Dict[str, any]:
    """
    Valida un dataset DICOM.

    Args:
        ds: Dataset DICOM da validare
        strict: Se True, solleva eccezioni per errori critici

    Returns:
        Dizionario con i risultati della validazione

    Raises:
        DicomValidationError: Se strict=True e la validazione fallisce
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'checks': {}
    }

    results['checks']['has_patient_id'] = _validate_patient_id(ds, results)
    results['checks']['has_series_uid'] = _validate_series_uid(ds, results)
    results['checks']['has_pixel_data'] = _validate_pixel_data(ds, results)
    results['checks']['has_dimensions'] = _validate_dimensions(ds, results)
    results['checks']['has_spacing'] = _validate_spacing(ds, results)

    if results['errors']:
        results['valid'] = False

        if strict:
            error_msg = '; '.join(results['errors'])
            raise DicomValidationError(f"Validazione DICOM fallita: {error_msg}")

    return results


def _validate_patient_id(ds: FileDataset, results: Dict) -> bool:
    """Valida la presenza del PatientID."""
    if not hasattr(ds, 'PatientID') or not ds.PatientID:
        results['warnings'].append("PatientID mancante o vuoto")
        return False
    return True


def _validate_series_uid(ds: FileDataset, results: Dict) -> bool:
    """Valida la presenza del SeriesInstanceUID."""
    if not hasattr(ds, 'SeriesInstanceUID') or not ds.SeriesInstanceUID:
        results['errors'].append("SeriesInstanceUID mancante (obbligatorio)")
        return False
    return True


def _validate_pixel_data(ds: FileDataset, results: Dict) -> bool:
    """Valida la presenza dei dati pixel."""
    if not hasattr(ds, 'pixel_array'):
        results['errors'].append("Dati pixel mancanti")
        return False

    try:
        pixel_array = ds.pixel_array
        if pixel_array.size == 0:
            results['errors'].append("Array pixel vuoto")
            return False
    except Exception as e:
        results['errors'].append(f"Errore lettura pixel data: {str(e)}")
        return False

    return True


def _validate_dimensions(ds: FileDataset, results: Dict) -> bool:
    """Valida le dimensioni dell'immagine."""
    if not hasattr(ds, 'Rows') or not hasattr(ds, 'Columns'):
        results['errors'].append("Dimensioni immagine (Rows/Columns) mancanti")
        return False

    if ds.Rows <= 0 or ds.Columns <= 0:
        results['errors'].append(f"Dimensioni invalide: {ds.Rows}x{ds.Columns}")
        return False

    return True


def _validate_spacing(ds: FileDataset, results: Dict) -> bool:
    """Valida il pixel spacing."""
    if not hasattr(ds, 'PixelSpacing'):
        results['warnings'].append("PixelSpacing mancante")
        return False

    try:
        spacing = [float(x) for x in ds.PixelSpacing]
        if any(s <= 0 for s in spacing):
            results['warnings'].append(f"PixelSpacing invalido: {spacing}")
            return False
    except:
        results['warnings'].append("Errore nel parsing di PixelSpacing")
        return False

    return True


def validate_pixel_data_range(
    pixel_array: np.ndarray,
    expected_min: Optional[float] = None,
    expected_max: Optional[float] = None
) -> Dict[str, any]:
    """
    Valida il range dei valori pixel.

    Args:
        pixel_array: Array numpy con i dati pixel
        expected_min: Valore minimo atteso (opzionale)
        expected_max: Valore massimo atteso (opzionale)

    Returns:
        Dizionario con i risultati della validazione
    """
    results = {
        'valid': True,
        'warnings': [],
        'stats': {
            'min': float(np.min(pixel_array)),
            'max': float(np.max(pixel_array)),
            'mean': float(np.mean(pixel_array)),
            'std': float(np.std(pixel_array))
        }
    }

    if expected_min is not None and results['stats']['min'] < expected_min:
        results['valid'] = False
        results['warnings'].append(
            f"Valore minimo {results['stats']['min']} sotto il limite {expected_min}"
        )

    if expected_max is not None and results['stats']['max'] > expected_max:
        results['valid'] = False
        results['warnings'].append(
            f"Valore massimo {results['stats']['max']} sopra il limite {expected_max}"
        )

    if np.any(np.isnan(pixel_array)):
        results['valid'] = False
        results['warnings'].append("Presenti valori NaN nei dati pixel")

    if np.any(np.isinf(pixel_array)):
        results['valid'] = False
        results['warnings'].append("Presenti valori infiniti nei dati pixel")

    return results


def validate_series_geometry(datasets: List[FileDataset]) -> Dict[str, any]:
    """
    Valida la geometria di una serie DICOM.

    Args:
        datasets: Lista di dataset DICOM della serie

    Returns:
        Dizionario con i risultati della validazione
    """
    results = {
        'valid': True,
        'warnings': [],
        'checks': {}
    }

    if len(datasets) < 2:
        results['warnings'].append("Serie con meno di 2 slice")
        return results

    results['checks']['uniform_spacing'] = _check_uniform_slice_spacing(datasets, results)
    results['checks']['no_gaps'] = _check_no_gaps(datasets, results)
    results['checks']['no_overlaps'] = _check_no_overlaps(datasets, results)

    if any(not check for check in results['checks'].values()):
        results['valid'] = False

    return results


def _check_uniform_slice_spacing(datasets: List[FileDataset], results: Dict) -> bool:
    """Verifica che lo spacing tra le slice sia uniforme."""
    positions = []

    for ds in datasets:
        if hasattr(ds, 'ImagePositionPatient'):
            positions.append(float(ds.ImagePositionPatient[2]))
        elif hasattr(ds, 'SliceLocation'):
            positions.append(float(ds.SliceLocation))
        else:
            results['warnings'].append("Impossibile verificare spacing (posizioni mancanti)")
            return False

    positions = sorted(positions)
    spacings = np.diff(positions)

    if len(spacings) > 0:
        mean_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)

        if std_spacing / mean_spacing > 0.01:
            results['warnings'].append(
                f"Spacing non uniforme (std/mean = {std_spacing/mean_spacing:.3f})"
            )
            return False

    return True


def _check_no_gaps(datasets: List[FileDataset], results: Dict) -> bool:
    """Verifica che non ci siano gap tra le slice."""
    positions = []

    for ds in datasets:
        if hasattr(ds, 'ImagePositionPatient'):
            positions.append(float(ds.ImagePositionPatient[2]))
        elif hasattr(ds, 'SliceLocation'):
            positions.append(float(ds.SliceLocation))
        else:
            return True

    positions = sorted(positions)
    spacings = np.diff(positions)

    if len(spacings) > 1:
        mean_spacing = np.mean(spacings)
        max_spacing = np.max(spacings)

        if max_spacing > mean_spacing * 1.5:
            results['warnings'].append(
                f"Possibile gap tra slice (max spacing = {max_spacing:.2f} mm)"
            )
            return False

    return True


def _check_no_overlaps(datasets: List[FileDataset], results: Dict) -> bool:
    """Verifica che non ci siano overlap tra le slice."""
    positions = []

    for ds in datasets:
        if hasattr(ds, 'ImagePositionPatient'):
            positions.append(float(ds.ImagePositionPatient[2]))
        elif hasattr(ds, 'SliceLocation'):
            positions.append(float(ds.SliceLocation))
        else:
            return True

    positions = sorted(positions)

    for i in range(len(positions) - 1):
        if abs(positions[i] - positions[i+1]) < 1e-6:
            results['warnings'].append("Possibile overlap tra slice (posizioni duplicate)")
            return False

    return True


def validate_modality(ds: FileDataset, expected_modality: str) -> bool:
    """
    Valida che il dataset abbia la modalità attesa.

    Args:
        ds: Dataset DICOM
        expected_modality: Modalità attesa (es. 'MR', 'CT', etc.)

    Returns:
        True se la modalità corrisponde
    """
    if not hasattr(ds, 'Modality'):
        return False

    return ds.Modality == expected_modality


def get_validation_summary(validation_results: Dict) -> str:
    """
    Crea un summary testuale dei risultati di validazione.

    Args:
        validation_results: Dizionario con i risultati della validazione

    Returns:
        Stringa con il summary
    """
    lines = []

    lines.append(f"Validazione: {'PASSED' if validation_results['valid'] else 'FAILED'}")

    if validation_results.get('errors'):
        lines.append("\nErrori:")
        for error in validation_results['errors']:
            lines.append(f"  - {error}")

    if validation_results.get('warnings'):
        lines.append("\nWarning:")
        for warning in validation_results['warnings']:
            lines.append(f"  - {warning}")

    if 'checks' in validation_results:
        lines.append("\nCheck:")
        for check_name, passed in validation_results['checks'].items():
            status = '✓' if passed else '✗'
            lines.append(f"  {status} {check_name}")

    return '\n'.join(lines)
