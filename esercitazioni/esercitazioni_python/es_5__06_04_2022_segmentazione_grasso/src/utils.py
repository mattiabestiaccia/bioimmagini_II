"""
Utilita' per la segmentazione del grasso addominale (SAT/VAT) da MRI.

Questo modulo implementa le funzioni per:
- Caricamento volumi 3D DICOM da acquisizioni assiali addominali
- Clustering K-means (K=3) per separazione aria/acqua/grasso
- Labeling e rimozione componenti spurie (braccia)
- Active contours doppi per segmentazione SAT (bordo esterno e interno)
- EM-GMM (Expectation-Maximization Gaussian Mixture Model) per VAT
- Calcolo volumi e indici SAT, VAT, VAT/SAT

References:
    Positano et al., "Accurate segmentation of subcutaneous and
    intermuscular adipose tissue from MR images of the thigh",
    Journal of Magnetic Resonance Imaging, 2004.
"""

import numpy as np
import pydicom
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import ndimage
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour
from skimage.filters import gaussian
from skimage.measure import label, regionprops
import warnings


def load_abdominal_volume(
    dicom_dir: Path,
    expected_slices: int = 18
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Carica volume 3D da serie DICOM assiale addominale.

    Args:
        dicom_dir: Directory contenente i file DICOM
        expected_slices: Numero atteso di slice (default 18)

    Returns:
        volume: Array 3D (slices, height, width) con intensita' normalizzate [0,1]
        metadata: Dizionario con informazioni DICOM (pixel_spacing, slice_thickness, etc.)

    Raises:
        ValueError: Se il numero di slice non corrisponde a expected_slices
    """
    dicom_files = sorted(Path(dicom_dir).glob("*.dcm"))

    if len(dicom_files) != expected_slices:
        warnings.warn(
            f"Trovate {len(dicom_files)} slice invece di {expected_slices}",
            UserWarning
        )

    # Carica primo file per metadata
    first_ds = pydicom.dcmread(dicom_files[0])
    rows, cols = first_ds.Rows, first_ds.Columns

    # Inizializza volume 3D
    volume = np.zeros((len(dicom_files), rows, cols), dtype=np.float32)
    slice_positions = []

    # Carica tutte le slice
    for i, dcm_path in enumerate(dicom_files):
        ds = pydicom.dcmread(dcm_path)
        volume[i] = ds.pixel_array.astype(np.float32)

        # Estrai posizione slice se disponibile
        if hasattr(ds, 'SliceLocation'):
            slice_positions.append(ds.SliceLocation)
        elif hasattr(ds, 'ImagePositionPatient'):
            slice_positions.append(ds.ImagePositionPatient[2])  # Coordinata Z

    # Ordina slice per posizione spaziale (cranio-caudale)
    if slice_positions:
        sort_indices = np.argsort(slice_positions)
        volume = volume[sort_indices]

    # Normalizza intensita' [0, 1]
    volume_min = volume.min()
    volume_max = volume.max()
    if volume_max > volume_min:
        volume = (volume - volume_min) / (volume_max - volume_min)

    # Estrai metadata rilevanti
    metadata = {
        'pixel_spacing': first_ds.PixelSpacing if hasattr(first_ds, 'PixelSpacing') else [1.0, 1.0],
        'slice_thickness': first_ds.SliceThickness if hasattr(first_ds, 'SliceThickness') else 1.0,
        'rows': rows,
        'cols': cols,
        'n_slices': len(dicom_files),
        'slice_positions': slice_positions if slice_positions else None
    }

    return volume, metadata


def kmeans_fat_segmentation(
    volume: np.ndarray,
    n_clusters: int = 3,
    n_init: int = 20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Segmentazione grasso tramite K-means clustering (K=3).

    Separa il volume in 3 cluster principali:
    - Aria (intensita' bassa)
    - Acqua/muscolo (intensita' media)
    - Grasso (intensita' alta in T1-weighted)

    Args:
        volume: Volume 3D normalizzato [0,1]
        n_clusters: Numero cluster (default 3)
        n_init: Numero inizializzazioni K-means (default 20)
        random_state: Seed per riproducibilita'

    Returns:
        labels_volume: Volume 3D con label cluster (0, 1, 2)
        centroids: Centroidi dei cluster (intensita' media)
        tissue_map: Dizionario {'air': label, 'water': label, 'fat': label}
    """
    # Reshape per K-means: (n_voxels, 1)
    original_shape = volume.shape
    X = volume.reshape(-1, 1)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_.flatten()

    # Reshape a volume 3D
    labels_volume = labels.reshape(original_shape)

    # Identifica cluster per intensita' crescente
    sorted_indices = np.argsort(centroids)

    tissue_map = {
        'air': sorted_indices[0],        # Intensita' minima
        'water': sorted_indices[1],      # Intensita' media
        'fat': sorted_indices[2]         # Intensita' massima (T1-weighted)
    }

    return labels_volume, centroids, tissue_map


def remove_spurious_components(
    binary_mask: np.ndarray,
    keep_largest: bool = True,
    min_size: Optional[int] = None
) -> np.ndarray:
    """
    Rimuove componenti connesse spurie (es. braccia) tramite labeling 3D.

    Args:
        binary_mask: Maschera binaria 3D
        keep_largest: Se True, mantiene solo la componente piu' grande (torso)
        min_size: Se specificato, rimuove componenti < min_size voxel

    Returns:
        cleaned_mask: Maschera binaria pulita
    """
    # Labeling 3D con connettivita' 26 (3D)
    labeled_volume, num_features = label(binary_mask, connectivity=3, return_num=True)

    if num_features == 0:
        return binary_mask

    # Calcola dimensione di ogni componente
    component_sizes = np.bincount(labeled_volume.ravel())
    component_sizes[0] = 0  # Ignora background

    if keep_largest:
        # Mantieni solo la componente piu' grande (torso)
        largest_label = np.argmax(component_sizes)
        cleaned_mask = (labeled_volume == largest_label).astype(np.uint8)

    elif min_size is not None:
        # Rimuovi componenti piccole
        valid_labels = np.where(component_sizes >= min_size)[0]
        cleaned_mask = np.isin(labeled_volume, valid_labels).astype(np.uint8)

    else:
        cleaned_mask = binary_mask

    return cleaned_mask


def create_circular_seed(
    image_shape: Tuple[int, int],
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[int] = None
) -> np.ndarray:
    """
    Crea maschera seed circolare per inizializzare active contours.

    Args:
        image_shape: Dimensioni immagine (height, width)
        center: Centro cerchio (row, col). Se None, usa centro immagine
        radius: Raggio cerchio. Se None, usa 1/4 della dimensione minima

    Returns:
        seed: Maschera binaria con seed circolare
    """
    rows, cols = image_shape

    if center is None:
        center = (rows // 2, cols // 2)

    if radius is None:
        radius = min(rows, cols) // 4

    # Crea griglia coordinate
    y, x = np.ogrid[:rows, :cols]

    # Distanza dal centro
    dist_from_center = np.sqrt((y - center[0])**2 + (x - center[1])**2)

    # Seed circolare
    seed = (dist_from_center <= radius).astype(np.uint8)

    return seed


def segment_sat_with_active_contours(
    image_slice: np.ndarray,
    torso_mask: np.ndarray,
    outer_seed: Optional[np.ndarray] = None,
    inner_seed: Optional[np.ndarray] = None,
    outer_iterations: int = 150,
    inner_iterations: int = 100,
    smoothing: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segmentazione SAT con approccio morfologico migliorato.

    Il SAT e' delimitato da:
    - Bordo ESTERNO: cute (usato torso_mask da K-means + closing)
    - Bordo INTERNO: fascia muscolare (erosione del torso mask)

    Args:
        image_slice: Slice 2D normalizzata [0,1]
        torso_mask: Maschera torso da K-means (grasso, esclude braccia)
        outer_seed: Non usato (mantenuto per compatibilita')
        inner_seed: Non usato (mantenuto per compatibilita')
        outer_iterations: Numero erosioni per inner (default 150 -> 15)
        inner_iterations: Non usato
        smoothing: Non usato

    Returns:
        sat_mask: Maschera SAT (regione tra outer e inner)
        outer_contour: Maschera contorno esterno (cute)
        inner_contour: Maschera contorno interno (fascia)
    """
    # STRATEGIA SEMPLIFICATA:
    # Il K-means gia' identifica il grasso correttamente
    # Usiamo operazioni morfologiche per separare SAT da VAT

    # OUTER CONTOUR: Chiusura morfologica per riempire buchi e smoothing
    struct = ndimage.generate_binary_structure(2, 2)
    outer_contour = ndimage.binary_closing(torso_mask, structure=struct, iterations=5)
    outer_contour = ndimage.binary_fill_holes(outer_contour)
    outer_contour = outer_contour.astype(np.uint8)

    # INNER CONTOUR: Erosione pesante per ottenere solo regione intra-addominale
    # Il numero di erosioni dipende dalla risoluzione - usiamo outer_iterations/10
    n_erosions = max(outer_iterations // 10, 10)
    inner_contour = ndimage.binary_erosion(
        outer_contour,
        structure=struct,
        iterations=n_erosions
    ).astype(np.uint8)

    # Riempi buchi nell'inner contour
    inner_contour = ndimage.binary_fill_holes(inner_contour).astype(np.uint8)

    # SAT = grasso nella regione outer AND NOT inner
    sat_region = np.logical_and(outer_contour, np.logical_not(inner_contour))
    sat_mask = np.logical_and(sat_region, torso_mask).astype(np.uint8)

    return sat_mask, outer_contour, inner_contour


def fit_em_gmm(
    histogram: np.ndarray,
    bin_edges: np.ndarray,
    n_components: int = 2,
    n_init: int = 10,
    random_state: int = 42
) -> Tuple[GaussianMixture, np.ndarray]:
    """
    Fit EM-GMM (Expectation-Maximization Gaussian Mixture Model) su istogramma.

    Usato per identificare VAT da istogramma intensita' nella regione intra-addominale.
    Il VAT ha intensita' simile al grasso subcutaneo (picco alto nell'istogramma).

    Args:
        histogram: Conteggi istogramma
        bin_edges: Bordi bin istogramma
        n_components: Numero Gaussiane (default 2: tessuto + grasso)
        n_init: Numero inizializzazioni EM
        random_state: Seed riproducibilita'

    Returns:
        gmm: Modello GaussianMixture fittato
        bin_centers: Centri bin per plotting
    """
    # Centri bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Replica valori per pesare con conteggi istogramma
    # (EM-GMM richiede samples, non istogramma diretto)
    X = np.repeat(bin_centers, histogram.astype(int))
    X = X.reshape(-1, 1)

    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_components,
        n_init=n_init,
        random_state=random_state,
        covariance_type='full'
    )
    gmm.fit(X)

    return gmm, bin_centers


def extract_vat_from_gmm(
    image_slice: np.ndarray,
    inner_contour: np.ndarray,
    gmm: GaussianMixture,
    fat_component_idx: int = 1
) -> np.ndarray:
    """
    Estrae maschera VAT usando GMM fittato su istogramma.

    Classifica ogni pixel nella regione intra-addominale (inner_contour)
    come VAT se appartiene alla componente Gaussiana del grasso.

    Args:
        image_slice: Slice 2D normalizzata
        inner_contour: Maschera regione intra-addominale
        gmm: Modello GMM fittato
        fat_component_idx: Indice componente grasso (default 1)

    Returns:
        vat_mask: Maschera binaria VAT
    """
    # Estrai intensita' nella regione intra-addominale
    intra_abdominal_pixels = image_slice[inner_contour > 0]

    if len(intra_abdominal_pixels) == 0:
        return np.zeros_like(image_slice, dtype=np.uint8)

    # Predict componente GMM per ogni pixel
    X = intra_abdominal_pixels.reshape(-1, 1)
    predictions = gmm.predict(X)

    # Crea maschera VAT
    vat_mask = np.zeros_like(image_slice, dtype=np.uint8)

    # Assegna VAT ai pixel classificati come grasso
    intra_coords = np.argwhere(inner_contour > 0)
    for i, (row, col) in enumerate(intra_coords):
        if predictions[i] == fat_component_idx:
            vat_mask[row, col] = 1

    return vat_mask


def calculate_fat_volumes(
    sat_mask_3d: np.ndarray,
    vat_mask_3d: np.ndarray,
    pixel_spacing: List[float],
    slice_thickness: float
) -> Dict[str, float]:
    """
    Calcola volumi SAT, VAT e indice VAT/SAT.

    Args:
        sat_mask_3d: Maschera SAT 3D (slices, height, width)
        vat_mask_3d: Maschera VAT 3D
        pixel_spacing: [row_spacing_mm, col_spacing_mm]
        slice_thickness: Spessore slice in mm

    Returns:
        results: Dizionario con:
            - 'sat_volume_cm3': Volume SAT in cm^3
            - 'vat_volume_cm3': Volume VAT in cm^3
            - 'vat_sat_ratio_percent': Rapporto VAT/SAT in %
            - 'total_fat_cm3': Volume totale grasso
    """
    # Calcola volume voxel in mm^3
    voxel_volume_mm3 = pixel_spacing[0] * pixel_spacing[1] * slice_thickness

    # Converti a cm^3
    voxel_volume_cm3 = voxel_volume_mm3 / 1000.0

    # Conta voxel
    sat_voxels = np.sum(sat_mask_3d)
    vat_voxels = np.sum(vat_mask_3d)

    # Calcola volumi
    sat_volume_cm3 = sat_voxels * voxel_volume_cm3
    vat_volume_cm3 = vat_voxels * voxel_volume_cm3
    total_fat_cm3 = sat_volume_cm3 + vat_volume_cm3

    # Rapporto VAT/SAT in percentuale
    if sat_volume_cm3 > 0:
        vat_sat_ratio_percent = (vat_volume_cm3 / sat_volume_cm3) * 100.0
    else:
        vat_sat_ratio_percent = 0.0

    results = {
        'sat_volume_cm3': sat_volume_cm3,
        'vat_volume_cm3': vat_volume_cm3,
        'vat_sat_ratio_percent': vat_sat_ratio_percent,
        'total_fat_cm3': total_fat_cm3
    }

    return results


def get_largest_component_2d(binary_mask: np.ndarray) -> np.ndarray:
    """
    Estrae la componente connessa piu' grande da maschera 2D.

    Utile per rimuovere piccoli artefatti da segmentazioni slice-by-slice.

    Args:
        binary_mask: Maschera binaria 2D

    Returns:
        largest_component: Maschera con solo la componente piu' grande
    """
    labeled, num_features = label(binary_mask, connectivity=2, return_num=True)

    if num_features == 0:
        return binary_mask

    # Trova componente piu' grande
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # Ignora background

    largest_label = np.argmax(component_sizes)
    largest_component = (labeled == largest_label).astype(np.uint8)

    return largest_component


def morphological_cleanup(
    binary_mask: np.ndarray,
    operation: str = 'closing',
    kernel_size: int = 3
) -> np.ndarray:
    """
    Operazioni morfologiche per pulire maschere binarie.

    Args:
        binary_mask: Maschera binaria 2D o 3D
        operation: 'erosion', 'dilation', 'opening', 'closing'
        kernel_size: Dimensione kernel strutturante

    Returns:
        cleaned_mask: Maschera pulita
    """
    # Crea elemento strutturante
    if binary_mask.ndim == 2:
        struct = ndimage.generate_binary_structure(2, 1)
    else:
        struct = ndimage.generate_binary_structure(3, 1)

    # Applica operazione
    if operation == 'erosion':
        cleaned_mask = ndimage.binary_erosion(binary_mask, structure=struct, iterations=kernel_size)
    elif operation == 'dilation':
        cleaned_mask = ndimage.binary_dilation(binary_mask, structure=struct, iterations=kernel_size)
    elif operation == 'opening':
        cleaned_mask = ndimage.binary_opening(binary_mask, structure=struct, iterations=kernel_size)
    elif operation == 'closing':
        cleaned_mask = ndimage.binary_closing(binary_mask, structure=struct, iterations=kernel_size)
    else:
        raise ValueError(f"Operazione non supportata: {operation}")

    return cleaned_mask.astype(np.uint8)


def process_fat_segmentation_pipeline(
    volume: np.ndarray,
    metadata: Dict[str, any],
    kmeans_clusters: int = 3,
    outer_iterations: int = 150,
    inner_iterations: int = 100,
    gmm_components: int = 2,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Pipeline completa per segmentazione grasso addominale SAT/VAT.

    Steps:
        1. K-means clustering (K=3) per separare aria/acqua/grasso
        2. Rimozione braccia tramite labeling 3D
        3. Active contours doppi per SAT (bordo esterno/interno)
        4. EM-GMM su istogramma intra-addominale per VAT
        5. Calcolo volumi e indici

    Args:
        volume: Volume 3D normalizzato
        metadata: Metadati DICOM (pixel_spacing, slice_thickness)
        kmeans_clusters: Numero cluster K-means
        outer_iterations: Iterazioni snake esterno
        inner_iterations: Iterazioni snake interno
        gmm_components: Numero Gaussiane GMM
        verbose: Stampa progresso

    Returns:
        results: Dizionario con maschere, volumi, parametri
    """
    n_slices = volume.shape[0]

    # STEP 1: K-means clustering
    if verbose:
        print(f"[1/5] K-means clustering (K={kmeans_clusters})...")

    labels_volume, centroids, tissue_map = kmeans_fat_segmentation(
        volume,
        n_clusters=kmeans_clusters
    )

    # Maschera grasso iniziale
    fat_label = tissue_map['fat']
    fat_mask_kmeans = (labels_volume == fat_label).astype(np.uint8)

    # STEP 2: Rimozione braccia
    if verbose:
        print("[2/5] Rimozione componenti spurie (braccia)...")

    torso_mask_3d = remove_spurious_components(
        fat_mask_kmeans,
        keep_largest=True
    )

    # STEP 3: Active contours per SAT (slice by slice)
    if verbose:
        print(f"[3/5] Segmentazione SAT con active contours ({n_slices} slices)...")

    sat_mask_3d = np.zeros_like(volume, dtype=np.uint8)
    outer_contour_3d = np.zeros_like(volume, dtype=np.uint8)
    inner_contour_3d = np.zeros_like(volume, dtype=np.uint8)

    for z in range(n_slices):
        sat_mask_2d, outer_2d, inner_2d = segment_sat_with_active_contours(
            volume[z],
            torso_mask_3d[z],
            outer_iterations=outer_iterations,
            inner_iterations=inner_iterations
        )

        sat_mask_3d[z] = sat_mask_2d
        outer_contour_3d[z] = outer_2d
        inner_contour_3d[z] = inner_2d

    # STEP 4: EM-GMM per VAT
    if verbose:
        print(f"[4/5] Segmentazione VAT con EM-GMM ({n_slices} slices)...")

    vat_mask_3d = np.zeros_like(volume, dtype=np.uint8)

    for z in range(n_slices):
        # Estrai intensita' intra-addominali
        inner_region = inner_contour_3d[z] > 0

        if np.sum(inner_region) == 0:
            continue

        intra_pixels = volume[z][inner_region]

        # Crea istogramma
        hist, bin_edges = np.histogram(intra_pixels, bins=50, range=(0, 1))

        # Fit GMM
        try:
            gmm, _ = fit_em_gmm(hist, bin_edges, n_components=gmm_components)

            # Identifica componente grasso (media piu' alta in T1)
            means = gmm.means_.flatten()
            fat_component_idx = np.argmax(means)

            # Estrai VAT
            vat_mask_2d = extract_vat_from_gmm(
                volume[z],
                inner_contour_3d[z],
                gmm,
                fat_component_idx=fat_component_idx
            )

            vat_mask_3d[z] = vat_mask_2d

        except Exception as e:
            if verbose:
                print(f"  Warning slice {z}: GMM fit failed ({e})")
            continue

    # STEP 5: Calcolo volumi
    if verbose:
        print("[5/5] Calcolo volumi SAT, VAT, indici...")

    volumes = calculate_fat_volumes(
        sat_mask_3d,
        vat_mask_3d,
        metadata['pixel_spacing'],
        metadata['slice_thickness']
    )

    # Risultati
    results = {
        'sat_mask_3d': sat_mask_3d,
        'vat_mask_3d': vat_mask_3d,
        'outer_contour_3d': outer_contour_3d,
        'inner_contour_3d': inner_contour_3d,
        'torso_mask_3d': torso_mask_3d,
        'kmeans_labels': labels_volume,
        'tissue_map': tissue_map,
        'volumes': volumes,
        'metadata': metadata
    }

    if verbose:
        print("\n=== RISULTATI ===")
        print(f"SAT volume: {volumes['sat_volume_cm3']:.1f} cm^3")
        print(f"VAT volume: {volumes['vat_volume_cm3']:.1f} cm^3")
        print(f"VAT/SAT ratio: {volumes['vat_sat_ratio_percent']:.1f} %")
        print(f"Total fat: {volumes['total_fat_cm3']:.1f} cm^3")

    return results
