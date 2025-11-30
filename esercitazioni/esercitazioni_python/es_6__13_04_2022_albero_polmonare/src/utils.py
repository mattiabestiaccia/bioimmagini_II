"""
Utilita' per analisi albero bronchiale da CT toracica ad alta risoluzione.

Questo modulo implementa le funzioni per:
- Caricamento volumi 3D DICOM CT con correzione Hounsfield Units
- Interpolazione isotropa per voxel cubici
- Region growing 3D per segmentazione lume bronchiale
- Skeletonization (centerline extraction) con 3D medial axis
- Sphere method per misurazione diametro lume
- Estrazione percorso lungo albero bronchiale
- Visualizzazione 3D e grafici diametro vs distanza

Pipeline:
    1. Caricamento CT 3D (148 slice) con Hounsfield Units
    2. Interpolazione isotropa
    3. Region growing da seed in trachea
    4. Filtraggio maschera (riempimento buchi)
    5. Skeletonization 3D
    6. Estrazione centerline da endpoint a trachea
    7. Sphere method per diametro lungo centerline

Valori attesi:
    - Diametro trachea: 15-18 mm
    - Diametro bronchi primari: 10-12 mm

References:
    - Cancer Imaging Archive (LIDC-IDRI dataset)
    - Hounsfield Units: aria = -1000 HU, acqua = 0 HU
    - Sphere method per vessel diameter quantification

Author: Generated with Claude Code
Date: 2025-11-20
"""

import numpy as np
import pydicom
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from skimage.morphology import skeletonize_3d, ball, binary_closing, binary_dilation
from skimage.measure import label
import warnings


def load_ct_volume_hounsfield(
    dicom_dir: Path,
    expected_slices: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Carica volume 3D CT con corretta conversione a Hounsfield Units.

    I DICOM CT memorizzano valori grezzi che devono essere convertiti a HU usando:
    HU = pixel_value * RescaleSlope + RescaleIntercept

    Args:
        dicom_dir: Directory contenente file DICOM
        expected_slices: Numero atteso di slice (opzionale)

    Returns:
        volume: Array 3D (slices, height, width) in Hounsfield Units
        metadata: Dizionario con informazioni DICOM

    Raises:
        ValueError: Se i file DICOM non hanno RescaleSlope/Intercept
    """
    # Trova tutti i file DICOM
    dicom_files = sorted(Path(dicom_dir).glob("*.dcm"))

    if len(dicom_files) == 0:
        # Prova a cercare in subdirectory
        subdirs = [d for d in Path(dicom_dir).iterdir() if d.is_dir() and not d.name.startswith('_')]
        if subdirs:
            dicom_files = sorted(subdirs[0].glob("*"))
            dicom_files = [f for f in dicom_files if f.is_file() and not f.name.startswith('.')]

    if expected_slices and len(dicom_files) != expected_slices:
        warnings.warn(
            f"Trovate {len(dicom_files)} slice invece di {expected_slices}",
            UserWarning
        )

    # Carica primo file per metadata
    first_ds = pydicom.dcmread(dicom_files[0], force=True)
    rows, cols = first_ds.Rows, first_ds.Columns

    # Verifica presenza RescaleSlope e RescaleIntercept
    if not hasattr(first_ds, 'RescaleSlope') or not hasattr(first_ds, 'RescaleIntercept'):
        raise ValueError("File DICOM non contiene RescaleSlope/RescaleIntercept per conversione HU")

    # Inizializza volume 3D
    volume = np.zeros((len(dicom_files), rows, cols), dtype=np.float32)
    slice_locations = []

    # Carica tutte le slice con conversione HU
    for i, dcm_path in enumerate(dicom_files):
        ds = pydicom.dcmread(dcm_path, force=True)

        # Converti a Hounsfield Units
        pixel_array = ds.pixel_array.astype(np.float32)
        hu_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        volume[i] = hu_array

        # Estrai posizione slice
        if hasattr(ds, 'SliceLocation'):
            slice_locations.append(ds.SliceLocation)
        elif hasattr(ds, 'ImagePositionPatient'):
            slice_locations.append(ds.ImagePositionPatient[2])
        else:
            slice_locations.append(i)

    # Ordina slice per posizione (cranio-caudale)
    if slice_locations:
        sort_indices = np.argsort(slice_locations)
        volume = volume[sort_indices]

    # Estrai metadata
    metadata = {
        'pixel_spacing': [float(first_ds.PixelSpacing[0]), float(first_ds.PixelSpacing[1])],
        'slice_thickness': float(first_ds.SliceThickness) if hasattr(first_ds, 'SliceThickness') else 1.0,
        'rows': rows,
        'cols': cols,
        'n_slices': len(dicom_files),
        'rescale_slope': first_ds.RescaleSlope,
        'rescale_intercept': first_ds.RescaleIntercept,
        'slice_locations': slice_locations
    }

    return volume, metadata


def check_isotropic(metadata: Dict[str, any], tolerance: float = 0.01) -> bool:
    """
    Verifica se il volume e' isotropo (voxel cubici).

    Args:
        metadata: Dizionario metadata da load_ct_volume_hounsfield
        tolerance: Tolleranza relativa per considerare dimensioni uguali

    Returns:
        True se isotropo, False altrimenti
    """
    px_spacing = metadata['pixel_spacing'][0]
    py_spacing = metadata['pixel_spacing'][1]
    pz_spacing = metadata['slice_thickness']

    max_spacing = max(px_spacing, py_spacing, pz_spacing)

    is_isotropic = (
        abs(px_spacing - py_spacing) / max_spacing < tolerance and
        abs(px_spacing - pz_spacing) / max_spacing < tolerance and
        abs(py_spacing - pz_spacing) / max_spacing < tolerance
    )

    return is_isotropic


def interpolate_to_isotropic(
    volume: np.ndarray,
    metadata: Dict[str, any],
    target_spacing: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Interpola volume a voxel isotropi (cubici).

    Mantiene costante il FOV (Field of View), interpolando a risoluzione massima
    (tipicamente pixel spacing).

    Args:
        volume: Volume 3D non isotropo
        metadata: Metadata con spacing
        target_spacing: Spacing target (se None, usa min(pixel_spacing))

    Returns:
        volume_iso: Volume interpolato con voxel cubici
        metadata_iso: Metadata aggiornato
    """
    px, py = metadata['pixel_spacing']
    pz = metadata['slice_thickness']

    # Target spacing = massima risoluzione (min spacing)
    if target_spacing is None:
        target_spacing = min(px, py, pz)

    # Dimensioni originali
    nz, ny, nx = volume.shape

    # FOV originale
    fov_x = nx * px
    fov_y = ny * py
    fov_z = nz * pz

    # Nuove dimensioni per mantenere FOV
    new_nx = int(np.round(fov_x / target_spacing))
    new_ny = int(np.round(fov_y / target_spacing))
    new_nz = int(np.round(fov_z / target_spacing))

    print(f"Interpolazione isotropa:")
    print(f"  Spacing originale: [{px:.3f}, {py:.3f}, {pz:.3f}] mm")
    print(f"  Shape originale: {volume.shape}")
    print(f"  Target spacing: {target_spacing:.3f} mm")
    print(f"  Shape interpolata: ({new_nz}, {new_ny}, {new_nx})")

    # Crea griglia originale
    z_orig = np.arange(nz) * pz
    y_orig = np.arange(ny) * py
    x_orig = np.arange(nx) * px

    # Crea griglia target
    z_new = np.linspace(0, fov_z, new_nz, endpoint=False)
    y_new = np.linspace(0, fov_y, new_ny, endpoint=False)
    x_new = np.linspace(0, fov_x, new_nx, endpoint=False)

    # Interpolatore
    interpolator = RegularGridInterpolator(
        (z_orig, y_orig, x_orig),
        volume,
        method='linear',
        bounds_error=False,
        fill_value=volume.min()
    )

    # Griglia meshgrid per nuovi punti
    z_grid, y_grid, x_grid = np.meshgrid(z_new, y_new, x_new, indexing='ij')
    points = np.stack([z_grid.ravel(), y_grid.ravel(), x_grid.ravel()], axis=-1)

    # Interpola
    volume_iso = interpolator(points).reshape(new_nz, new_ny, new_nx)

    # Aggiorna metadata
    metadata_iso = metadata.copy()
    metadata_iso['pixel_spacing'] = [target_spacing, target_spacing]
    metadata_iso['slice_thickness'] = target_spacing
    metadata_iso['n_slices'] = new_nz
    metadata_iso['rows'] = new_ny
    metadata_iso['cols'] = new_nx
    metadata_iso['is_isotropic'] = True

    return volume_iso, metadata_iso


def region_growing_3d(
    volume: np.ndarray,
    seed: Tuple[int, int, int],
    tolerance: float = 100.0,
    connectivity: int = 26,
    max_iterations: Optional[int] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Region growing 3D per segmentazione albero bronchiale.

    Algoritmo iterativo che espande una regione dal seed includendo voxel vicini
    con intensita' simile (entro tolerance dalla media della regione segmentata).

    Args:
        volume: Volume 3D in Hounsfield Units
        seed: Coordinate seed (z, y, x) nella trachea
        tolerance: Tolleranza HU (default 100 per aria ~-1000 HU)
        connectivity: 6 (faces), 18 (edges), o 26 (vertices)
        max_iterations: Numero massimo iterazioni (None = illimitate)
        verbose: Stampa progresso

    Returns:
        mask: Maschera binaria 3D della regione segmentata
    """
    mask = np.zeros(volume.shape, dtype=bool)
    mask[seed] = True

    # Elemento strutturante per connettivita'
    if connectivity == 6:
        struct = ndimage.generate_binary_structure(3, 1)  # Only faces
    elif connectivity == 18:
        struct = ndimage.generate_binary_structure(3, 2)  # Faces + edges
    else:  # 26
        struct = ndimage.generate_binary_structure(3, 3)  # Faces + edges + vertices
        # Per 26-connectivity uso cube(3)
        struct = np.ones((3, 3, 3), dtype=bool)

    iteration = 0
    prev_count = 0

    while True:
        iteration += 1

        # Calcola media intensita' regione corrente
        segmented_values = volume[mask]
        mean_val = np.mean(segmented_values)

        # Trova vicini
        dilated = ndimage.binary_dilation(mask, structure=struct)
        neighbors_mask = np.logical_and(dilated, ~mask)
        neighbor_coords = np.argwhere(neighbors_mask)

        if len(neighbor_coords) == 0:
            break

        # Valuta vicini
        neighbor_values = volume[neighbors_mask]
        within_tolerance = np.abs(neighbor_values - mean_val) <= tolerance

        # Aggiungi vicini che soddisfano criterio
        coords_to_add = neighbor_coords[within_tolerance]
        for coord in coords_to_add:
            mask[tuple(coord)] = True

        current_count = np.sum(mask)

        if verbose and iteration % 10 == 0:
            print(f"  Iteration {iteration}: {current_count} voxel, mean HU = {mean_val:.1f}")

        # Condizioni di stop
        if current_count == prev_count:
            break

        if max_iterations and iteration >= max_iterations:
            break

        prev_count = current_count

    if verbose:
        print(f"Region growing completato: {iteration} iterazioni, {np.sum(mask)} voxel")

    return mask.astype(np.uint8)


def fill_holes_3d(mask: np.ndarray, method: str = 'closing') -> np.ndarray:
    """
    Riempie buchi nella maschera 3D.

    Args:
        mask: Maschera binaria 3D
        method: 'closing', 'label', o 'fill'
            - 'closing': Chiusura morfologica
            - 'label': Labeling background + rimozione componenti interne
            - 'fill': ndimage.binary_fill_holes

    Returns:
        mask_filled: Maschera con buchi riempiti
    """
    if method == 'closing':
        struct = ball(3)
        mask_filled = binary_closing(mask, struct)
        return mask_filled.astype(np.uint8)

    elif method == 'label':
        # Labeling del background
        background = ~mask.astype(bool)
        labeled_bg = label(background, connectivity=3)

        # Identifica background esterno (componente piu' grande)
        bg_sizes = np.bincount(labeled_bg.ravel())
        bg_sizes[0] = 0  # Ignora label 0
        largest_bg_label = np.argmax(bg_sizes)

        # Tutto cio' che non e' background esterno diventa foreground
        external_bg = (labeled_bg == largest_bg_label)
        mask_filled = ~external_bg

        return mask_filled.astype(np.uint8)

    else:  # 'fill'
        mask_filled = ndimage.binary_fill_holes(mask)
        return mask_filled.astype(np.uint8)


def skeletonize_3d_clean(
    mask: np.ndarray,
    min_branch_length: int = 5,
    verbose: bool = True
) -> np.ndarray:
    """
    Skeletonization 3D con pulizia rami corti.

    Estrae la centerline (medial axis) dell'albero bronchiale.

    Args:
        mask: Maschera binaria 3D
        min_branch_length: Lunghezza minima ramo (in voxel)
        verbose: Stampa info

    Returns:
        skeleton: Maschera binaria scheletro 3D
    """
    if verbose:
        print("Skeletonization 3D in corso (puo' richiedere alcuni minuti)...")

    # Skeletonize
    skeleton = skeletonize_3d(mask.astype(bool))
    skeleton = skeleton.astype(np.uint8)

    if verbose:
        n_skeleton = np.sum(skeleton)
        print(f"  Scheletro grezzo: {n_skeleton} voxel")

    # Pulizia rami corti tramite labeling
    if min_branch_length > 1:
        labeled_skel = label(skeleton, connectivity=3)

        # Rimuovi componenti piccole
        for region_label in range(1, labeled_skel.max() + 1):
            region_size = np.sum(labeled_skel == region_label)
            if region_size < min_branch_length:
                skeleton[labeled_skel == region_label] = 0

        if verbose:
            n_cleaned = np.sum(skeleton)
            print(f"  Scheletro pulito: {n_cleaned} voxel")

    return skeleton


def find_skeleton_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """
    Trova endpoints (punti terminali) dello scheletro 3D.

    Un endpoint e' un voxel con esattamente 1 vicino nello scheletro.

    Args:
        skeleton: Maschera binaria scheletro

    Returns:
        endpoints: Array Nx3 con coordinate (z, y, x) degli endpoints
    """
    # Conta vicini per ogni voxel dello scheletro
    struct = np.ones((3, 3, 3), dtype=bool)
    struct[1, 1, 1] = False  # Escludi centro

    neighbor_count = ndimage.convolve(skeleton.astype(int), struct.astype(int), mode='constant', cval=0)

    # Endpoints hanno esattamente 1 vicino
    endpoints_mask = np.logical_and(skeleton > 0, neighbor_count == 1)
    endpoints = np.argwhere(endpoints_mask)

    return endpoints


def extract_centerline_path(
    skeleton: np.ndarray,
    start_endpoint: Tuple[int, int, int],
    direction: str = 'descending_z'
) -> List[Tuple[int, int, int]]:
    """
    Estrae percorso lungo centerline da un endpoint.

    Usa strategia greedy: da ogni punto, seleziona il vicino piu' vicino
    con coordinata z decrescente (verso trachea).

    Args:
        skeleton: Maschera binaria scheletro
        start_endpoint: Coordinate (z, y, x) punto di partenza
        direction: 'descending_z' (verso trachea) o 'ascending_z'

    Returns:
        path: Lista di coordinate (z, y, x) lungo centerline
    """
    path = [start_endpoint]
    visited = set([start_endpoint])

    current = start_endpoint
    skeleton_coords = set(map(tuple, np.argwhere(skeleton > 0)))

    while True:
        z, y, x = current

        # Trova vicini nello scheletro (26-connectivity)
        neighbors = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue

                    neighbor = (z + dz, y + dy, x + dx)

                    if (neighbor in skeleton_coords and
                        neighbor not in visited):
                        neighbors.append(neighbor)

        if len(neighbors) == 0:
            break

        # Seleziona vicino con z minore (descending) o maggiore (ascending)
        if direction == 'descending_z':
            next_point = min(neighbors, key=lambda p: (p[0], np.linalg.norm(np.array(p) - np.array(current))))
        else:
            next_point = max(neighbors, key=lambda p: (p[0], -np.linalg.norm(np.array(p) - np.array(current))))

        # Filtro: accetta solo se z <= current_z (per descending)
        if direction == 'descending_z' and next_point[0] > current[0]:
            # Permetti piccole salite (biforcazioni)
            if next_point[0] - current[0] > 2:
                break

        path.append(next_point)
        visited.add(next_point)
        current = next_point

    return path


def sphere_method_diameter(
    mask: np.ndarray,
    centerline_point: Tuple[int, int, int],
    max_radius: int = 30,
    spacing: float = 1.0
) -> float:
    """
    Calcola diametro lume con sphere method.

    Trova il massimo raggio di una sfera centrata su centerline_point
    che sia completamente contenuta nella maschera.

    Args:
        mask: Maschera binaria lume
        centerline_point: Punto centerline (z, y, x)
        max_radius: Raggio massimo da testare (voxel)
        spacing: Spacing voxel in mm (per conversione)

    Returns:
        diameter_mm: Diametro in mm (2 * raggio massimo inscrivibile)
    """
    z0, y0, x0 = centerline_point

    for radius in range(1, max_radius + 1):
        # Crea sfera
        sphere = ball(radius)

        # Centro sfera
        sz, sy, sx = sphere.shape
        cz, cy, cx = sz // 2, sy // 2, sx // 2

        # Coordinate sfera nel volume
        z_start = z0 - cz
        z_end = z0 - cz + sz
        y_start = y0 - cy
        y_end = y0 - cy + sy
        x_start = x0 - cx
        x_end = x0 - cx + sx

        # Verifica bounds
        if (z_start < 0 or z_end > mask.shape[0] or
            y_start < 0 or y_end > mask.shape[1] or
            x_start < 0 or x_end > mask.shape[2]):
            # Sfera esce dal volume
            radius -= 1
            break

        # Estrai regione volume
        region = mask[z_start:z_end, y_start:y_end, x_start:x_end]

        # Verifica se sfera e' inscritta (tutti voxel sphere devono essere in mask)
        if not np.all(region[sphere]):
            # Sfera tocca bordo
            radius -= 1
            break

    # Converti a mm
    diameter_mm = 2 * radius * spacing

    return diameter_mm


def measure_diameter_along_path(
    mask: np.ndarray,
    centerline_path: List[Tuple[int, int, int]],
    spacing: float = 1.0,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Misura diametro lungo centerline path.

    Args:
        mask: Maschera binaria lume
        centerline_path: Lista coordinate centerline
        spacing: Spacing voxel in mm
        verbose: Stampa progresso

    Returns:
        distances_mm: Array distanze cumulative lungo path (mm)
        diameters_mm: Array diametri corrispondenti (mm)
    """
    n_points = len(centerline_path)
    diameters_mm = np.zeros(n_points)
    distances_mm = np.zeros(n_points)

    for i, point in enumerate(centerline_path):
        # Misura diametro
        diameter = sphere_method_diameter(mask, point, max_radius=30, spacing=spacing)
        diameters_mm[i] = diameter

        # Calcola distanza cumulativa
        if i > 0:
            prev_point = centerline_path[i - 1]
            delta = np.array(point) - np.array(prev_point)
            dist_voxel = np.linalg.norm(delta)
            distances_mm[i] = distances_mm[i - 1] + dist_voxel * spacing

        if verbose and i % 20 == 0:
            print(f"  Punto {i}/{n_points}: distanza = {distances_mm[i]:.1f} mm, diametro = {diameter:.2f} mm")

    return distances_mm, diameters_mm


def smooth_diameters(diameters: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smoothing diametri con media mobile.

    Args:
        diameters: Array diametri grezzi
        window_size: Dimensione finestra

    Returns:
        diameters_smooth: Array diametri smoothed
    """
    kernel = np.ones(window_size) / window_size
    diameters_smooth = np.convolve(diameters, kernel, mode='same')
    return diameters_smooth
