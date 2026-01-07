#!/usr/bin/env python3
"""
generate_synthetic_data.py - Generate Synthetic Multi-Echo Data for T2* Mapping

Creates realistic synthetic multi-echo GRE data for testing the T2* mapping pipeline.
Simulates two patients:
- PAZIENTE1: Iron overload (thalassemia major)
- PAZIENTE2: Normal control

Author: Biomedical Imaging Course
Date: 2025
"""

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
from pathlib import Path
from datetime import datetime
from typing import Tuple
import struct


def create_phantom_image(size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a synthetic cardiac phantom with anatomical regions.

    Returns masks for: heart_septum, liver, muscle, background
    """
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size // 2, size // 2

    # Heart region (elliptical)
    heart_mask = ((x - center_x) / 60)**2 + ((y - center_y - 20) / 45)**2 < 1

    # Cardiac septum (inner part of heart)
    septum_mask = ((x - center_x) / 30)**2 + ((y - center_y - 20) / 20)**2 < 1
    septum_mask = septum_mask & heart_mask

    # Liver region (right side, lower)
    liver_mask = ((x - center_x - 80) / 50)**2 + ((y - center_y + 40) / 35)**2 < 1

    # Skeletal muscle (left side)
    muscle_mask = ((x - center_x + 90) / 40)**2 + ((y - center_y + 30) / 60)**2 < 1

    # Background (everything else that has some signal)
    all_tissue = heart_mask | liver_mask | muscle_mask
    background_mask = ~all_tissue

    return septum_mask, liver_mask, muscle_mask, heart_mask


def generate_multiecho_volume(
    size: int = 256,
    n_echoes: int = 10,
    echo_times: np.ndarray = None,
    t2star_septum: float = 22.0,  # ms
    t2star_liver: float = 26.0,   # ms
    t2star_muscle: float = 30.0,  # ms
    t2star_blood: float = 200.0,  # ms (long T2* for blood)
    s0_septum: float = 180.0,
    s0_liver: float = 200.0,
    s0_muscle: float = 150.0,
    noise_level: float = 5.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic multi-echo volume with known T2* values.

    Parameters
    ----------
    size : int
        Image size (square)
    n_echoes : int
        Number of echo times
    echo_times : np.ndarray, optional
        Echo times in ms. If None, uses typical values
    t2star_septum : float
        T2* of cardiac septum in ms
    t2star_liver : float
        T2* of liver in ms
    t2star_muscle : float
        T2* of skeletal muscle in ms
    t2star_blood : float
        T2* of blood in ms
    s0_* : float
        Initial signal intensity for each tissue
    noise_level : float
        Standard deviation of Rician noise
    seed : int
        Random seed for reproducibility

    Returns
    -------
    volume : np.ndarray
        Shape (n_echoes, size, size)
    echo_times : np.ndarray
        Echo times in ms
    """
    np.random.seed(seed)

    if echo_times is None:
        # Typical echo times for T2* mapping (spacing ~2.2 ms)
        echo_times = np.array([2.0, 4.2, 6.4, 8.6, 10.8, 13.0, 15.2, 17.4, 19.6, 21.8])

    n_echoes = len(echo_times)

    # Create anatomical masks
    septum_mask, liver_mask, muscle_mask, heart_mask = create_phantom_image(size)

    # Blood is heart region minus septum
    blood_mask = heart_mask & ~septum_mask

    # Initialize volume
    volume = np.zeros((n_echoes, size, size), dtype=np.float32)

    # Generate signal decay for each tissue
    for i, te in enumerate(echo_times):
        # Septum: S = S0 * exp(-TE/T2*)
        volume[i][septum_mask] = s0_septum * np.exp(-te / t2star_septum)

        # Liver
        volume[i][liver_mask] = s0_liver * np.exp(-te / t2star_liver)

        # Muscle
        volume[i][muscle_mask] = s0_muscle * np.exp(-te / t2star_muscle)

        # Blood (very slow decay)
        volume[i][blood_mask] = 100 * np.exp(-te / t2star_blood)

    # Add Rician noise (magnitude of complex Gaussian)
    real_noise = np.random.normal(0, noise_level, volume.shape)
    imag_noise = np.random.normal(0, noise_level, volume.shape)
    volume = np.sqrt((volume + real_noise)**2 + imag_noise**2)

    return volume.astype(np.float32), echo_times


def create_dicom_file(
    pixel_data: np.ndarray,
    echo_time: float,
    patient_name: str,
    series_description: str,
    instance_number: int,
    output_path: Path,
    global_max: float = None
) -> None:
    """
    Create a minimal DICOM file from pixel data.

    Parameters
    ----------
    pixel_data : np.ndarray
        2D image array
    echo_time : float
        Echo time in ms
    patient_name : str
        Patient name for DICOM header
    series_description : str
        Series description
    instance_number : int
        Instance number (echo index + 1)
    output_path : Path
        Output file path
    global_max : float, optional
        Global maximum for normalization across all echoes.
        If None, uses local max (NOT recommended for multi-echo).
    """
    # File meta information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
    file_meta.ImplementationClassUID = generate_uid()

    # Create the FileDataset
    ds = FileDataset(
        str(output_path),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128
    )

    # Patient Module
    ds.PatientName = patient_name
    ds.PatientID = patient_name.replace(' ', '_')
    ds.PatientBirthDate = '19800101'
    ds.PatientSex = 'O'

    # General Study Module
    ds.StudyInstanceUID = generate_uid()
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.now().strftime('%H%M%S')
    ds.StudyID = '1'
    ds.AccessionNumber = ''

    # General Series Module
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 1
    ds.SeriesDescription = series_description
    ds.Modality = 'MR'

    # General Equipment Module
    ds.Manufacturer = 'Synthetic'
    ds.InstitutionName = 'Biomedical Imaging Course'

    # Image Pixel Module
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.Rows = pixel_data.shape[0]
    ds.Columns = pixel_data.shape[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned

    # Convert to uint16 using global max for consistent scaling across echoes
    scale_max = global_max if global_max is not None else np.max(pixel_data)
    if scale_max > 0:
        # Use 60000 instead of 65535 to leave headroom
        pixel_data_scaled = np.clip(pixel_data / scale_max * 60000, 0, 65535).astype(np.uint16)
    else:
        pixel_data_scaled = np.zeros_like(pixel_data, dtype=np.uint16)

    ds.PixelData = pixel_data_scaled.tobytes()

    # MR Image Module
    ds.EchoTime = str(echo_time)
    ds.RepetitionTime = '15'
    ds.FlipAngle = '20'
    ds.MagneticFieldStrength = '1.5'
    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'M']

    # Frame of Reference Module
    ds.FrameOfReferenceUID = generate_uid()

    # SOP Common Module
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber = instance_number

    # Save
    ds.save_as(str(output_path))


def generate_patient_data(
    output_dir: Path,
    patient_name: str,
    t2star_septum: float,
    t2star_liver: float,
    t2star_muscle: float,
    size: int = 256,
    seed: int = None
) -> None:
    """
    Generate complete multi-echo DICOM series for a patient.

    Parameters
    ----------
    output_dir : Path
        Output directory for DICOM files
    patient_name : str
        Patient identifier
    t2star_septum : float
        T2* of cardiac septum in ms
    t2star_liver : float
        T2* of liver in ms
    t2star_muscle : float
        T2* of skeletal muscle in ms
    size : int
        Image size
    seed : int, optional
        Random seed
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating data for {patient_name}:")
    print(f"  T2* septum: {t2star_septum:.1f} ms")
    print(f"  T2* liver: {t2star_liver:.1f} ms")
    print(f"  T2* muscle: {t2star_muscle:.1f} ms")

    # Generate volume
    volume, echo_times = generate_multiecho_volume(
        size=size,
        t2star_septum=t2star_septum,
        t2star_liver=t2star_liver,
        t2star_muscle=t2star_muscle,
        seed=seed
    )

    # Calculate global maximum across ALL echoes for consistent scaling
    global_max = np.max(volume)
    print(f"  Global signal max: {global_max:.1f}")

    # Save each echo as DICOM with consistent scaling
    for i, te in enumerate(echo_times):
        filename = f"IM_{i+1:04d}.dcm"
        output_path = output_dir / filename

        create_dicom_file(
            pixel_data=volume[i],
            echo_time=te,
            patient_name=patient_name,
            series_description=f"T2* Multi-Echo GRE",
            instance_number=i + 1,
            output_path=output_path,
            global_max=global_max
        )

    print(f"  Created {len(echo_times)} DICOM files in {output_dir}")
    print(f"  Echo times: {echo_times.tolist()} ms")


def main():
    """Generate synthetic data for both patients."""
    base_dir = Path(__file__).parent.parent / "data" / "DICOM"

    print("=" * 70)
    print("GENERATING SYNTHETIC MULTI-ECHO DATA FOR T2* MAPPING")
    print("=" * 70)

    # PAZIENTE1: Severe iron overload (thalassemia)
    # Very short T2* due to iron deposition
    generate_patient_data(
        output_dir=base_dir / "PAZIENTE1",
        patient_name="PAZIENTE1",
        t2star_septum=2.0,    # Severe cardiac overload
        t2star_liver=0.8,     # Very severe hepatic overload
        t2star_muscle=25.0,   # Muscle relatively spared
        seed=42
    )

    # PAZIENTE2: Normal control
    # Normal T2* values
    generate_patient_data(
        output_dir=base_dir / "PAZIENTE2",
        patient_name="PAZIENTE2",
        t2star_septum=22.0,   # Normal cardiac T2*
        t2star_liver=26.0,    # Normal hepatic T2*
        t2star_muscle=30.0,   # Normal muscle T2*
        seed=123
    )

    print("\n" + "=" * 70)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nData saved to: {base_dir}")
    print("\nExpected T2* values:")
    print("  PAZIENTE1 (iron overload):")
    print("    - Cardiac septum: ~2 ms (severe)")
    print("    - Liver: <1 ms (very severe)")
    print("    - Muscle: ~25 ms (normal)")
    print("  PAZIENTE2 (normal):")
    print("    - Cardiac septum: ~22 ms")
    print("    - Liver: ~26 ms")
    print("    - Muscle: ~30 ms")


if __name__ == "__main__":
    main()
