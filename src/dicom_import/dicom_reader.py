"""
Funzioni per la lettura di file e serie DICOM.
"""

import os
from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np
import pydicom
from pydicom.dataset import FileDataset


def read_dicom_file(file_path: Union[str, Path]) -> Tuple[np.ndarray, FileDataset]:
    """
    Legge un singolo file DICOM.

    Args:
        file_path: Percorso del file DICOM da leggere

    Returns:
        Tupla contenente:
        - Array numpy con i dati pixel
        - Dataset pydicom con tutti i metadata

    Raises:
        FileNotFoundError: Se il file non esiste
        pydicom.errors.InvalidDicomError: Se il file non è un DICOM valido
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File DICOM non trovato: {file_path}")

    try:
        ds = pydicom.dcmread(str(file_path))

        if not hasattr(ds, 'pixel_array'):
            raise ValueError(f"Il file {file_path} non contiene dati pixel")

        pixel_data = ds.pixel_array

        return pixel_data, ds

    except Exception as e:
        raise RuntimeError(f"Errore nella lettura del file DICOM {file_path}: {str(e)}")


def read_dicom_series(
    directory: Union[str, Path],
    series_uid: Optional[str] = None,
    sort_by_position: bool = True
) -> Tuple[np.ndarray, List[FileDataset]]:
    """
    Legge una serie completa di file DICOM da una directory.

    Args:
        directory: Directory contenente i file DICOM
        series_uid: UID della serie da leggere. Se None, legge la prima serie trovata
        sort_by_position: Se True, ordina le slice per posizione spaziale

    Returns:
        Tupla contenente:
        - Array numpy 3D con la serie completa (slice, rows, cols)
        - Lista di dataset pydicom per ogni slice

    Raises:
        FileNotFoundError: Se la directory non esiste
        ValueError: Se non vengono trovati file DICOM validi
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory non trovata: {directory}")

    dicom_files = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            try:
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                if series_uid is None or ds.SeriesInstanceUID == series_uid:
                    dicom_files.append(file_path)
            except:
                continue

    if not dicom_files:
        raise ValueError(f"Nessun file DICOM trovato in {directory}")

    datasets = []
    pixel_arrays = []

    for file_path in dicom_files:
        pixel_data, ds = read_dicom_file(file_path)
        datasets.append(ds)
        pixel_arrays.append(pixel_data)

    if sort_by_position:
        sorted_indices = _sort_by_slice_position(datasets)
        datasets = [datasets[i] for i in sorted_indices]
        pixel_arrays = [pixel_arrays[i] for i in sorted_indices]

    volume = np.stack(pixel_arrays, axis=0)

    return volume, datasets


def _sort_by_slice_position(datasets: List[FileDataset]) -> List[int]:
    """
    Ordina i dataset DICOM per posizione della slice.

    Args:
        datasets: Lista di dataset DICOM

    Returns:
        Lista di indici ordinati
    """
    positions = []

    for ds in datasets:
        if hasattr(ds, 'ImagePositionPatient'):
            z_pos = float(ds.ImagePositionPatient[2])
        elif hasattr(ds, 'SliceLocation'):
            z_pos = float(ds.SliceLocation)
        elif hasattr(ds, 'InstanceNumber'):
            z_pos = float(ds.InstanceNumber)
        else:
            z_pos = 0.0

        positions.append(z_pos)

    return np.argsort(positions).tolist()


def get_dicom_files_in_directory(
    directory: Union[str, Path],
    recursive: bool = False
) -> List[Path]:
    """
    Trova tutti i file DICOM in una directory.

    Args:
        directory: Directory da scansionare
        recursive: Se True, cerca ricorsivamente nelle sottodirectory

    Returns:
        Lista di Path ai file DICOM trovati
    """
    directory = Path(directory)
    dicom_files = []

    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    for file_path in directory.glob(pattern):
        if file_path.is_file():
            try:
                pydicom.dcmread(str(file_path), stop_before_pixels=True)
                dicom_files.append(file_path)
            except:
                continue

    return dicom_files
