"""
Funzioni per l'estrazione di metadata da file DICOM.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import pydicom
from pydicom.dataset import FileDataset


def extract_metadata(ds: FileDataset) -> Dict[str, Any]:
    """
    Estrae tutti i metadata rilevanti da un dataset DICOM.

    Args:
        ds: Dataset DICOM da cui estrarre i metadata

    Returns:
        Dizionario con i metadata estratti
    """
    metadata = {}

    metadata['patient'] = extract_patient_info(ds)
    metadata['study'] = extract_study_info(ds)
    metadata['series'] = extract_series_info(ds)
    metadata['image'] = extract_image_info(ds)
    metadata['acquisition'] = extract_acquisition_info(ds)

    return metadata


def extract_patient_info(ds: FileDataset) -> Dict[str, Any]:
    """
    Estrae informazioni sul paziente.

    Args:
        ds: Dataset DICOM

    Returns:
        Dizionario con le informazioni del paziente
    """
    patient_info = {}

    fields = {
        'patient_id': 'PatientID',
        'patient_name': 'PatientName',
        'patient_birth_date': 'PatientBirthDate',
        'patient_sex': 'PatientSex',
        'patient_age': 'PatientAge',
        'patient_weight': 'PatientWeight',
    }

    for key, dicom_tag in fields.items():
        patient_info[key] = _safe_get_attribute(ds, dicom_tag)

    return patient_info


def extract_study_info(ds: FileDataset) -> Dict[str, Any]:
    """
    Estrae informazioni sullo studio.

    Args:
        ds: Dataset DICOM

    Returns:
        Dizionario con le informazioni dello studio
    """
    study_info = {}

    fields = {
        'study_instance_uid': 'StudyInstanceUID',
        'study_date': 'StudyDate',
        'study_time': 'StudyTime',
        'study_description': 'StudyDescription',
        'study_id': 'StudyID',
        'accession_number': 'AccessionNumber',
        'referring_physician': 'ReferringPhysicianName',
    }

    for key, dicom_tag in fields.items():
        study_info[key] = _safe_get_attribute(ds, dicom_tag)

    if study_info['study_date'] and study_info['study_time']:
        study_info['study_datetime'] = _parse_dicom_datetime(
            study_info['study_date'],
            study_info['study_time']
        )

    return study_info


def extract_series_info(ds: FileDataset) -> Dict[str, Any]:
    """
    Estrae informazioni sulla serie.

    Args:
        ds: Dataset DICOM

    Returns:
        Dizionario con le informazioni della serie
    """
    series_info = {}

    fields = {
        'series_instance_uid': 'SeriesInstanceUID',
        'series_number': 'SeriesNumber',
        'series_description': 'SeriesDescription',
        'series_date': 'SeriesDate',
        'series_time': 'SeriesTime',
        'modality': 'Modality',
        'body_part_examined': 'BodyPartExamined',
        'protocol_name': 'ProtocolName',
        'operators_name': 'OperatorsName',
    }

    for key, dicom_tag in fields.items():
        series_info[key] = _safe_get_attribute(ds, dicom_tag)

    return series_info


def extract_image_info(ds: FileDataset) -> Dict[str, Any]:
    """
    Estrae informazioni sull'immagine.

    Args:
        ds: Dataset DICOM

    Returns:
        Dizionario con le informazioni dell'immagine
    """
    image_info = {}

    fields = {
        'sop_instance_uid': 'SOPInstanceUID',
        'instance_number': 'InstanceNumber',
        'rows': 'Rows',
        'columns': 'Columns',
        'pixel_spacing': 'PixelSpacing',
        'slice_thickness': 'SliceThickness',
        'slice_location': 'SliceLocation',
        'image_position_patient': 'ImagePositionPatient',
        'image_orientation_patient': 'ImageOrientationPatient',
        'bits_allocated': 'BitsAllocated',
        'bits_stored': 'BitsStored',
        'samples_per_pixel': 'SamplesPerPixel',
        'photometric_interpretation': 'PhotometricInterpretation',
    }

    for key, dicom_tag in fields.items():
        image_info[key] = _safe_get_attribute(ds, dicom_tag)

    if image_info['pixel_spacing']:
        image_info['pixel_spacing'] = [float(x) for x in image_info['pixel_spacing']]

    if image_info['image_position_patient']:
        image_info['image_position_patient'] = [
            float(x) for x in image_info['image_position_patient']
        ]

    if image_info['image_orientation_patient']:
        image_info['image_orientation_patient'] = [
            float(x) for x in image_info['image_orientation_patient']
        ]

    return image_info


def extract_acquisition_info(ds: FileDataset) -> Dict[str, Any]:
    """
    Estrae informazioni sull'acquisizione.

    Args:
        ds: Dataset DICOM

    Returns:
        Dizionario con le informazioni dell'acquisizione
    """
    acquisition_info = {}

    fields = {
        'manufacturer': 'Manufacturer',
        'manufacturer_model': 'ManufacturerModelName',
        'station_name': 'StationName',
        'software_version': 'SoftwareVersions',
        'magnetic_field_strength': 'MagneticFieldStrength',
        'scanning_sequence': 'ScanningSequence',
        'sequence_variant': 'SequenceVariant',
        'scan_options': 'ScanOptions',
        'mr_acquisition_type': 'MRAcquisitionType',
        'repetition_time': 'RepetitionTime',
        'echo_time': 'EchoTime',
        'flip_angle': 'FlipAngle',
    }

    for key, dicom_tag in fields.items():
        acquisition_info[key] = _safe_get_attribute(ds, dicom_tag)

    return acquisition_info


def _safe_get_attribute(ds: FileDataset, attribute: str, default: Any = None) -> Any:
    """
    Ottiene un attributo da un dataset DICOM in modo sicuro.

    Args:
        ds: Dataset DICOM
        attribute: Nome dell'attributo da ottenere
        default: Valore di default se l'attributo non esiste

    Returns:
        Valore dell'attributo o default
    """
    try:
        value = getattr(ds, attribute, default)
        if value is not None and value != '':
            return value
        return default
    except:
        return default


def _parse_dicom_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """
    Parsifica data e ora DICOM in un oggetto datetime.

    Args:
        date_str: Data in formato DICOM (YYYYMMDD)
        time_str: Ora in formato DICOM (HHMMSS.ffffff)

    Returns:
        Oggetto datetime o None se il parsing fallisce
    """
    try:
        date_str = str(date_str).split('.')[0]
        time_str = str(time_str).split('.')[0]

        dt_str = f"{date_str}{time_str}"
        return datetime.strptime(dt_str, '%Y%m%d%H%M%S')
    except:
        return None


def get_series_summary(datasets: List[FileDataset]) -> Dict[str, Any]:
    """
    Crea un summary di una serie DICOM.

    Args:
        datasets: Lista di dataset DICOM della serie

    Returns:
        Dizionario con il summary della serie
    """
    if not datasets:
        return {}

    first_ds = datasets[0]

    summary = {
        'num_slices': len(datasets),
        'series_uid': _safe_get_attribute(first_ds, 'SeriesInstanceUID'),
        'series_description': _safe_get_attribute(first_ds, 'SeriesDescription'),
        'modality': _safe_get_attribute(first_ds, 'Modality'),
        'patient_id': _safe_get_attribute(first_ds, 'PatientID'),
    }

    if hasattr(first_ds, 'Rows') and hasattr(first_ds, 'Columns'):
        summary['matrix_size'] = (first_ds.Rows, first_ds.Columns)

    if hasattr(first_ds, 'PixelSpacing'):
        summary['pixel_spacing'] = [float(x) for x in first_ds.PixelSpacing]

    if hasattr(first_ds, 'SliceThickness'):
        summary['slice_thickness'] = float(first_ds.SliceThickness)

    return summary
