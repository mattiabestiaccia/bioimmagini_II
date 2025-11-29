"""
Modulo per l'importazione e gestione di dati DICOM.

Questo modulo fornisce funzionalità per:
- Lettura di file DICOM singoli e serie multiple
- Estrazione di metadata da file DICOM
- Validazione e normalizzazione dei dati
- Gestione di dataset multi-serie
"""

from .dicom_reader import read_dicom_file, read_dicom_series
from .metadata_extractor import extract_metadata, extract_patient_info
from .series_manager import group_dicom_files, sort_series_by_position

__all__ = [
    'read_dicom_file',
    'read_dicom_series',
    'extract_metadata',
    'extract_patient_info',
    'group_dicom_files',
    'sort_series_by_position',
]

__version__ = '1.0.0'
