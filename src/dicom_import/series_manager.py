"""
Funzioni per la gestione di serie multiple DICOM.
"""

from pathlib import Path
from typing import Dict, List, Union, Optional
from collections import defaultdict
import pydicom
from pydicom.dataset import FileDataset
import numpy as np


def group_dicom_files(
    directory: Union[str, Path],
    recursive: bool = False
) -> Dict[str, List[Path]]:
    """
    Raggruppa i file DICOM per SeriesInstanceUID.

    Args:
        directory: Directory contenente i file DICOM
        recursive: Se True, cerca ricorsivamente nelle sottodirectory

    Returns:
        Dizionario con SeriesInstanceUID come chiave e lista di Path come valore
    """
    directory = Path(directory)
    series_dict = defaultdict(list)

    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    for file_path in directory.glob(pattern):
        if file_path.is_file():
            try:
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                series_uid = ds.SeriesInstanceUID
                series_dict[series_uid].append(file_path)
            except:
                continue

    return dict(series_dict)


def sort_series_by_position(datasets: List[FileDataset]) -> List[int]:
    """
    Ordina i dataset di una serie per posizione spaziale.

    Args:
        datasets: Lista di dataset DICOM

    Returns:
        Lista di indici ordinati
    """
    positions = []

    for ds in datasets:
        pos = _get_slice_position(ds)
        positions.append(pos)

    return np.argsort(positions).tolist()


def _get_slice_position(ds: FileDataset) -> float:
    """
    Ottiene la posizione della slice da un dataset DICOM.

    Args:
        ds: Dataset DICOM

    Returns:
        Posizione della slice
    """
    if hasattr(ds, 'ImagePositionPatient'):
        return float(ds.ImagePositionPatient[2])
    elif hasattr(ds, 'SliceLocation'):
        return float(ds.SliceLocation)
    elif hasattr(ds, 'InstanceNumber'):
        return float(ds.InstanceNumber)
    else:
        return 0.0


def get_series_info_summary(directory: Union[str, Path]) -> List[Dict]:
    """
    Crea un summary di tutte le serie in una directory.

    Args:
        directory: Directory contenente i file DICOM

    Returns:
        Lista di dizionari con informazioni su ogni serie
    """
    series_groups = group_dicom_files(directory, recursive=False)
    summaries = []

    for series_uid, file_paths in series_groups.items():
        try:
            first_ds = pydicom.dcmread(str(file_paths[0]), stop_before_pixels=True)

            summary = {
                'series_uid': series_uid,
                'num_files': len(file_paths),
                'series_number': getattr(first_ds, 'SeriesNumber', None),
                'series_description': getattr(first_ds, 'SeriesDescription', None),
                'modality': getattr(first_ds, 'Modality', None),
                'rows': getattr(first_ds, 'Rows', None),
                'columns': getattr(first_ds, 'Columns', None),
            }

            if hasattr(first_ds, 'PixelSpacing'):
                summary['pixel_spacing'] = [float(x) for x in first_ds.PixelSpacing]

            if hasattr(first_ds, 'SliceThickness'):
                summary['slice_thickness'] = float(first_ds.SliceThickness)

            summaries.append(summary)

        except Exception as e:
            continue

    return summaries


def validate_series_consistency(datasets: List[FileDataset]) -> Dict[str, bool]:
    """
    Valida la consistenza di una serie DICOM.

    Args:
        datasets: Lista di dataset DICOM della serie

    Returns:
        Dizionario con i risultati della validazione
    """
    if not datasets:
        return {'valid': False, 'reason': 'Nessun dataset fornito'}

    if len(datasets) < 2:
        return {'valid': True, 'warnings': ['Serie con una sola slice']}

    validation = {
        'valid': True,
        'warnings': [],
        'checks': {}
    }

    first_ds = datasets[0]

    validation['checks']['same_series_uid'] = _check_same_series_uid(datasets)
    validation['checks']['consistent_dimensions'] = _check_consistent_dimensions(datasets)
    validation['checks']['consistent_spacing'] = _check_consistent_spacing(datasets)
    validation['checks']['consistent_orientation'] = _check_consistent_orientation(datasets)

    for check_name, check_result in validation['checks'].items():
        if not check_result:
            validation['valid'] = False
            validation['warnings'].append(f"Check fallito: {check_name}")

    return validation


def _check_same_series_uid(datasets: List[FileDataset]) -> bool:
    """Verifica che tutti i dataset abbiano lo stesso SeriesInstanceUID."""
    if not hasattr(datasets[0], 'SeriesInstanceUID'):
        return False

    series_uid = datasets[0].SeriesInstanceUID
    return all(hasattr(ds, 'SeriesInstanceUID') and ds.SeriesInstanceUID == series_uid
               for ds in datasets)


def _check_consistent_dimensions(datasets: List[FileDataset]) -> bool:
    """Verifica che tutti i dataset abbiano le stesse dimensioni."""
    if not (hasattr(datasets[0], 'Rows') and hasattr(datasets[0], 'Columns')):
        return False

    rows = datasets[0].Rows
    columns = datasets[0].Columns

    return all(
        hasattr(ds, 'Rows') and hasattr(ds, 'Columns') and
        ds.Rows == rows and ds.Columns == columns
        for ds in datasets
    )


def _check_consistent_spacing(datasets: List[FileDataset]) -> bool:
    """Verifica che tutti i dataset abbiano lo stesso pixel spacing."""
    if not hasattr(datasets[0], 'PixelSpacing'):
        return True

    spacing = datasets[0].PixelSpacing

    for ds in datasets:
        if not hasattr(ds, 'PixelSpacing'):
            return False
        if not np.allclose(ds.PixelSpacing, spacing, rtol=1e-5):
            return False

    return True


def _check_consistent_orientation(datasets: List[FileDataset]) -> bool:
    """Verifica che tutti i dataset abbiano la stessa orientazione."""
    if not hasattr(datasets[0], 'ImageOrientationPatient'):
        return True

    orientation = datasets[0].ImageOrientationPatient

    for ds in datasets:
        if not hasattr(ds, 'ImageOrientationPatient'):
            return False
        if not np.allclose(ds.ImageOrientationPatient, orientation, rtol=1e-5):
            return False

    return True


def find_series_by_description(
    directory: Union[str, Path],
    description_pattern: str,
    case_sensitive: bool = False
) -> List[str]:
    """
    Trova serie DICOM che matchano un pattern nella descrizione.

    Args:
        directory: Directory contenente i file DICOM
        description_pattern: Pattern da cercare nella descrizione
        case_sensitive: Se True, ricerca case-sensitive

    Returns:
        Lista di SeriesInstanceUID che matchano il pattern
    """
    series_groups = group_dicom_files(directory, recursive=False)
    matching_series = []

    if not case_sensitive:
        description_pattern = description_pattern.lower()

    for series_uid, file_paths in series_groups.items():
        try:
            ds = pydicom.dcmread(str(file_paths[0]), stop_before_pixels=True)

            if hasattr(ds, 'SeriesDescription'):
                description = ds.SeriesDescription
                if not case_sensitive:
                    description = description.lower()

                if description_pattern in description:
                    matching_series.append(series_uid)

        except:
            continue

    return matching_series
