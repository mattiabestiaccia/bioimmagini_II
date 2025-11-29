"""
Funzioni utility per l'analisi di clustering su immagini di perfusione cardiaca MRI.

Questo modulo fornisce funzioni per:
- Caricamento dati DICOM sequenziali (2D+T)
- Calcolo del DICE index
- Post-processing delle maschere di segmentazione
- Visualizzazione risultati
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from scipy import ndimage
import scipy.io

# Import centralizzato del modulo DICOM
try:
    # Prova import relativo (quando es_3 è package installato)
    from dicom_import import read_dicom_series, extract_metadata
except ImportError:
    # Fallback: aggiungi src/ al path e importa
    project_root = Path(__file__).parent.parent.parent.parent.parent
    src_path = project_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        from dicom_import import read_dicom_series, extract_metadata
    else:
        # Se non disponibile, usa implementazione locale (backward compatibility)
        read_dicom_series = None
        extract_metadata = None

from .exceptions import (
    DataLoadError,
    DicomReadError,
    InvalidClusterCountError,
    ShapeMismatchError,
    ValidationError,
)
from .types import (
    BinaryMask,
    Centroids,
    ClusterLabels,
    ImageStack,
    TimeCurves,
    TriggerTimes,
)


logger = logging.getLogger(__name__)


def load_perfusion_series(
    dicom_dir: Path,
    n_frames: int | None = None
) -> tuple[ImageStack, TriggerTimes]:
    """
    Carica una serie temporale di immagini DICOM di perfusione.

    Usa il modulo centralizzato dicom_import se disponibile, altrimenti
    fallback all'implementazione locale.

    Parameters
    ----------
    dicom_dir : Path
        Directory contenente i file DICOM (I01, I02, ...)
    n_frames : Optional[int]
        Numero di frame da caricare. Se None, carica tutti i frame disponibili

    Returns
    -------
    image_stack : ImageStack
        Stack di immagini, shape (height, width, n_frames)
    trigger_times : TriggerTimes
        Array con i tempi di trigger per ogni frame, shape (n_frames,)

    Raises
    ------
    DataLoadError
        Se la directory non esiste o non contiene file DICOM
    DicomReadError
        Se i file DICOM non possono essere letti

    Examples
    --------
    >>> data_dir = Path("data/perfusione")
    >>> images, times = load_perfusion_series(data_dir)
    >>> print(f"Loaded {images.shape[2]} frames of size {images.shape[:2]}")
    """
    logger.info(f"Loading DICOM series from: {dicom_dir}")

    # Usa modulo centralizzato se disponibile
    if read_dicom_series is not None:
        return _load_perfusion_series_centralized(dicom_dir, n_frames)
    else:
        logger.warning("Using local DICOM loading (dicom_import module not available)")
        return _load_perfusion_series_local(dicom_dir, n_frames)


def _load_perfusion_series_centralized(
    dicom_dir: Path,
    n_frames: int | None = None
) -> tuple[ImageStack, TriggerTimes]:
    """
    Implementazione usando il modulo dicom_import centralizzato.
    """
    try:
        # Carica la serie completa usando il modulo centralizzato
        volume, datasets = read_dicom_series(
            directory=dicom_dir,
            series_uid=None,  # Carica la prima serie trovata
            sort_by_position=False  # Non ordinare per posizione spaziale (già ordinati per tempo)
        )

        # Limita al numero di frame richiesto
        if n_frames is not None and n_frames < volume.shape[0]:
            volume = volume[:n_frames, :, :]
            datasets = datasets[:n_frames]

        # Transposta per avere (H, W, T) invece di (T, H, W)
        image_stack = np.transpose(volume, (1, 2, 0)).astype(np.float32)

        # Estrai trigger times dai metadata
        trigger_times = np.zeros(len(datasets), dtype=np.float32)
        for i, ds in enumerate(datasets):
            if hasattr(ds, "TriggerTime"):
                trigger_times[i] = ds.TriggerTime / 1000.0  # ms -> s
            else:
                # Stima approssimativa
                if i == 0:
                    logger.warning("TriggerTime tag not found, using estimated times")
                trigger_times[i] = i * 0.8

        logger.info(f"Successfully loaded {image_stack.shape[2]} frames using centralized module")
        return image_stack, trigger_times

    except Exception as e:
        logger.error(f"Centralized loading failed: {e}, falling back to local implementation")
        return _load_perfusion_series_local(dicom_dir, n_frames)


def _load_perfusion_series_local(
    dicom_dir: Path,
    n_frames: int | None = None
) -> tuple[ImageStack, TriggerTimes]:
    """
    Implementazione locale (backward compatibility).
    """
    if not dicom_dir.exists():
        error_msg = f"DICOM directory does not exist: {dicom_dir}"
        logger.error(error_msg)
        raise DataLoadError(error_msg)

    dicom_files = sorted(dicom_dir.glob("I*"))

    if not dicom_files:
        error_msg = f"No DICOM files found in {dicom_dir}"
        logger.error(error_msg)
        raise DataLoadError(error_msg)

    if n_frames is not None:
        dicom_files = dicom_files[:n_frames]

    logger.info(f"Found {len(dicom_files)} DICOM files to load")

    # Carica il primo per ottenere le dimensioni
    try:
        first_dcm = pydicom.dcmread(dicom_files[0], force=True)
        height, width = first_dcm.pixel_array.shape
        n_frames_actual = len(dicom_files)
    except Exception as e:
        error_msg = "Failed to read first DICOM file"
        logger.error(error_msg)
        raise DicomReadError(error_msg, file_path=str(dicom_files[0]), original_error=e)

    # Pre-alloca array
    image_stack = np.zeros((height, width, n_frames_actual), dtype=np.float32)
    trigger_times = np.zeros(n_frames_actual, dtype=np.float32)

    # Carica tutte le immagini
    for i, dcm_path in enumerate(dicom_files):
        try:
            dcm = pydicom.dcmread(dcm_path, force=True)
            image_stack[:, :, i] = dcm.pixel_array.astype(np.float32)

            # Estrai trigger time se disponibile
            if hasattr(dcm, "TriggerTime"):
                trigger_times[i] = dcm.TriggerTime / 1000.0  # Converti ms -> s
            else:
                # Stima approssimativa: assume 1 frame per battito, ~800ms per battito
                if i == 0:
                    logger.warning("TriggerTime tag not found in DICOM files, using estimated times")
                trigger_times[i] = i * 0.8
        except Exception as e:
            error_msg = f"Failed to read DICOM file at frame {i}"
            logger.error(error_msg)
            raise DicomReadError(error_msg, file_path=str(dcm_path), original_error=e)

    logger.info(f"Successfully loaded {n_frames_actual} frames, shape: {image_stack.shape}")
    return image_stack, trigger_times


def load_gold_standard(mat_file: Path) -> dict[str, BinaryMask]:
    """
    Carica le maschere gold standard dal file MATLAB.

    Parameters
    ----------
    mat_file : Path
        Path al file GoldStandard.mat

    Returns
    -------
    masks : dict[str, BinaryMask]
        Dizionario con chiavi 'rv' (ventricolo destro), 'lv' (ventricolo sinistro),
        'myo' (miocardio), ogni elemento e' un array booleano shape (height, width)

    Raises
    ------
    DataLoadError
        Se il file non esiste o non puo' essere letto

    Examples
    --------
    >>> masks = load_gold_standard(Path("data/GoldStandard.mat"))
    >>> print(f"RV pixels: {masks['rv'].sum()}")
    """
    logger.info(f"Loading gold standard from: {mat_file}")

    if not mat_file.exists():
        error_msg = f"Gold standard file does not exist: {mat_file}"
        logger.error(error_msg)
        raise DataLoadError(error_msg)

    try:
        data = scipy.io.loadmat(mat_file)
    except Exception as e:
        error_msg = f"Failed to read MATLAB file: {mat_file}"
        logger.error(error_msg)
        raise DataLoadError(f"{error_msg} - {e}")

    try:
        masks = {
            "rv": data["DXmask"].astype(bool),  # Right Ventricle (Destro)
            "lv": data["SXmask"].astype(bool),  # Left Ventricle (Sinistro)
            "myo": data["MyoMask"].astype(bool)  # Myocardium
        }
    except KeyError as e:
        error_msg = "Required mask not found in gold standard file"
        logger.error(f"{error_msg}: {e}")
        raise DataLoadError(f"{error_msg}: {e}")

    logger.info(f"Successfully loaded {len(masks)} gold standard masks")
    return masks


def dice_coefficient(mask1: BinaryMask, mask2: BinaryMask) -> float:
    """
    Calcola il DICE coefficient tra due maschere binarie.

    Il DICE coefficient misura la sovrapposizione tra due maschere:
    DICE = 2 * |A ) B| / (|A| + |B|)

    Un valore di 1.0 indica perfetta sovrapposizione,
    un valore di 0.0 indica nessuna sovrapposizione.

    Parameters
    ----------
    mask1 : BinaryMask
        Prima maschera binaria
    mask2 : BinaryMask
        Seconda maschera binaria (stesse dimensioni di mask1)

    Returns
    -------
    dice : float
        DICE coefficient [0, 1]

    Raises
    ------
    ShapeMismatchError
        Se le due maschere hanno dimensioni diverse

    Examples
    --------
    >>> mask_a = np.array([[1, 1], [0, 0]], dtype=bool)
    >>> mask_b = np.array([[1, 0], [0, 0]], dtype=bool)
    >>> dice = dice_coefficient(mask_a, mask_b)
    >>> print(f"DICE = {dice:.3f}")  # 0.667
    """
    if mask1.shape != mask2.shape:
        error_msg = "Mask shapes do not match for DICE coefficient calculation"
        logger.error(f"{error_msg}: {mask1.shape} vs {mask2.shape}")
        raise ShapeMismatchError(
            error_msg,
            expected_shape=mask1.shape,
            actual_shape=mask2.shape
        )

    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()

    if mask1.sum() + mask2.sum() == 0:
        logger.debug("Both masks are empty, returning DICE = 1.0")
        return 1.0  # Entrambe vuote = perfetta corrispondenza

    dice = 2.0 * intersection / (mask1.sum() + mask2.sum())

    return float(dice)


def remove_small_regions(
    mask: BinaryMask,
    min_size: int = 50
) -> BinaryMask:
    """
    Rimuove regioni piccole dalla maschera binaria usando connected component labeling.

    Parameters
    ----------
    mask : BinaryMask
        Maschera binaria di input
    min_size : int
        Dimensione minima (in pixel) delle regioni da mantenere

    Returns
    -------
    cleaned_mask : BinaryMask
        Maschera pulita (stesse dimensioni di input)

    Examples
    --------
    >>> mask = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    >>> cleaned = remove_small_regions(mask, min_size=2)
    """
    mask_binary = mask.astype(bool)
    labeled, num_features = ndimage.label(mask_binary)

    # Calcola dimensione di ogni componente
    component_sizes = np.bincount(labeled.ravel())

    # Trova componenti da rimuovere (troppo piccole)
    small_components = np.where(component_sizes < min_size)[0]

    # Rimuovi componenti piccole
    remove_mask = np.isin(labeled, small_components)
    cleaned_mask = mask_binary.copy()
    cleaned_mask[remove_mask] = False

    return cleaned_mask


def keep_largest_component(mask: BinaryMask) -> BinaryMask:
    """
    Mantiene solo la componente connessa pi� grande nella maschera.

    Utile per eliminare regioni spurie mantenendo solo la regione principale.

    Parameters
    ----------
    mask : BinaryMask
        Maschera binaria di input

    Returns
    -------
    largest_mask : BinaryMask
        Maschera contenente solo la componente pi� grande

    Examples
    --------
    >>> mask = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool)
    >>> largest = keep_largest_component(mask)
    >>> print(largest.sum())  # 4 (la regione 2x2 a destra)
    """
    mask_binary = mask.astype(bool)
    labeled, num_features = ndimage.label(mask_binary)

    if num_features == 0:
        return mask_binary

    # Calcola dimensione di ogni componente (escludendo background = 0)
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # Ignora background

    # Trova componente pi� grande
    largest_component = component_sizes.argmax()

    # Crea maschera con solo la componente pi� grande
    largest_mask: BinaryMask = (labeled == largest_component).astype(np.bool_)

    return largest_mask


def extract_pixel_time_curves(
    image_stack: ImageStack,
    pixel_coords: list[tuple[int, int]]
) -> TimeCurves:
    """
    Estrae le curve intensit�/tempo per specifici pixel.

    Parameters
    ----------
    image_stack : ImageStack
        Stack di immagini, shape (height, width, n_frames)
    pixel_coords : list[tuple[int, int]]
        Lista di coordinate (row, col) dei pixel di interesse

    Returns
    -------
    curves : TimeCurves
        Array di curve, shape (n_pixels, n_frames)

    Examples
    --------
    >>> images = np.random.rand(256, 256, 79)
    >>> coords = [(100, 120), (150, 180)]
    >>> curves = extract_pixel_time_curves(images, coords)
    >>> print(curves.shape)  # (2, 79)
    """
    n_frames = image_stack.shape[2]
    n_pixels = len(pixel_coords)

    curves = np.zeros((n_pixels, n_frames), dtype=np.float32)

    for i, (row, col) in enumerate(pixel_coords):
        curves[i, :] = image_stack[row, col, :]

    return curves


def crop_to_roi(
    image: np.ndarray,
    roi: tuple[int, int, int, int]
) -> np.ndarray:
    """
    Ritaglia l'immagine alla ROI specificata.

    Parameters
    ----------
    image : np.ndarray
        Immagine 2D o 3D (con dimensione temporale)
    roi : tuple[int, int, int, int]
        ROI come (row_start, row_end, col_start, col_end)

    Returns
    -------
    cropped : np.ndarray
        Immagine ritagliata

    Raises
    ------
    ValidationError
        Se l'immagine ha dimensionalita' non supportata (non 2D o 3D)

    Examples
    --------
    >>> img = np.random.rand(256, 256, 79)
    >>> cropped = crop_to_roi(img, (50, 200, 50, 200))
    >>> print(cropped.shape)  # (150, 150, 79)
    """
    row_start, row_end, col_start, col_end = roi

    if image.ndim == 2:
        return image[row_start:row_end, col_start:col_end]
    if image.ndim == 3:
        return image[row_start:row_end, col_start:col_end, :]
    error_msg = f"Unsupported image dimensionality: {image.ndim} (expected 2D or 3D)"
    logger.error(error_msg)
    raise ValidationError(error_msg)


def identify_tissue_clusters(
    clustered_image: ClusterLabels,
    centroids: Centroids,
    n_clusters: int = 4
) -> dict[str, int]:
    """
    Identifica quale cluster corrisponde a quale tessuto basandosi sui centroidi.

    La funzione kmeans restituisce cluster in ordine casuale. Questa funzione
    identifica i cluster basandosi sulle caratteristiche delle curve:
    - Background: segnale costante basso
    - RV (ventricolo destro): picco precoce, alto
    - LV (ventricolo sinistro): picco intermedio
    - Myocardio: picco tardivo, moderato

    Parameters
    ----------
    clustered_image : ClusterLabels
        Immagine con etichette cluster, shape (height, width)
    centroids : Centroids
        Centroidi dei cluster, shape (n_clusters, n_features)
    n_clusters : int
        Numero di cluster (default: 4)

    Returns
    -------
    tissue_map : dict[str, int]
        Dizionario che mappa nome tessuto -> indice cluster
        Chiavi: 'background', 'rv', 'lv', 'myo'

    Raises
    ------
    InvalidClusterCountError
        Se n_clusters < 4 (minimo richiesto per identificare tutti i tessuti)

    Examples
    --------
    >>> labels = np.random.randint(0, 4, (256, 256))
    >>> cents = np.random.rand(4, 79)
    >>> tissue_map = identify_tissue_clusters(labels, cents)
    >>> print(tissue_map)  # {'background': 0, 'rv': 2, 'lv': 1, 'myo': 3}
    """
    # Validation
    min_clusters_required = 4
    if n_clusters < min_clusters_required:
        error_msg = "Insufficient clusters for tissue identification"
        logger.error(f"{error_msg}: need at least {min_clusters_required}, got {n_clusters}")
        raise InvalidClusterCountError(
            error_msg,
            n_clusters=n_clusters,
            min_required=min_clusters_required
        )

    logger.debug(f"Identifying tissue clusters from {n_clusters} clusters")

    # Calcola caratteristiche per ogni centroide
    n_frames = centroids.shape[1]
    features = {}

    for cluster_id in range(n_clusters):
        curve = centroids[cluster_id, :]

        # Caratteristiche
        baseline = np.mean(curve[:5])  # Media primi 5 frame
        peak_value = np.max(curve)
        peak_time = np.argmax(curve)
        contrast_increase = peak_value - baseline

        features[cluster_id] = {
            "baseline": baseline,
            "peak_value": peak_value,
            "peak_time": peak_time,
            "contrast_increase": contrast_increase,
            "mean": np.mean(curve)
        }

    # Identifica tessuti
    tissue_map = {}

    # Background: minor contrasto
    contrast_increases = [features[i]["contrast_increase"] for i in range(n_clusters)]
    tissue_map["background"] = int(np.argmin(contrast_increases))

    # RV: picco precoce e alto
    remaining = [i for i in range(n_clusters) if i != tissue_map["background"]]
    peak_times_remaining = [features[i]["peak_time"] for i in remaining]
    rv_candidate = remaining[np.argmin(peak_times_remaining)]
    tissue_map["rv"] = int(rv_candidate)

    # LV e Myocardio: tra i rimanenti, LV ha picco piu alto
    remaining = [i for i in remaining if i != rv_candidate]
    if len(remaining) == 2:
        peak_values = [features[i]["peak_value"] for i in remaining]
        if peak_values[0] > peak_values[1]:
            tissue_map["lv"] = int(remaining[0])
            tissue_map["myo"] = int(remaining[1])
        else:
            tissue_map["lv"] = int(remaining[1])
            tissue_map["myo"] = int(remaining[0])

    return tissue_map


def visualize_segmentation(
    image: np.ndarray,
    masks: dict[str, BinaryMask],
    title: str = "Segmentation Results",
    save_path: Path | None = None
) -> None:
    """
    Visualizza i risultati della segmentazione.

    Parameters
    ----------
    image : np.ndarray
        Immagine di riferimento (2D)
    masks : dict[str, BinaryMask]
        Dizionario con maschere per ogni tessuto
    title : str
        Titolo della figura
    save_path : Path | None
        Se specificato, salva la figura in questo path

    Examples
    --------
    >>> img = np.random.rand(256, 256)
    >>> masks = {'rv': rv_mask, 'lv': lv_mask, 'myo': myo_mask}
    >>> visualize_segmentation(img, masks, "K-means Segmentation")
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Immagine originale
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Overlay
    overlay = np.zeros((*image.shape, 3))
    overlay[..., 0] = image / image.max()  # Canale rosso: immagine
    overlay[..., 1] = image / image.max()  # Canale verde: immagine
    overlay[..., 2] = image / image.max()  # Canale blu: immagine

    # Colori per tessuti
    colors = {
        "rv": [0, 0, 1],    # Blu
        "lv": [1, 0, 0],    # Rosso
        "myo": [0, 1, 0]    # Verde
    }

    for tissue_name, mask in masks.items():
        if tissue_name in colors:
            color = colors[tissue_name]
            for c in range(3):
                overlay[mask, c] = color[c]

    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title("Overlay (RV=Blue, LV=Red, Myo=Green)")
    axes[0, 1].axis("off")

    # Maschere individuali
    mask_idx = 0
    positions = [(1, 0), (1, 1)]
    tissue_names_display = ["rv", "lv", "myo"]

    for tissue_name in tissue_names_display:
        if tissue_name in masks and mask_idx < 3:
            if mask_idx < 2:
                ax = axes[positions[mask_idx]]
                ax.imshow(masks[tissue_name], cmap="gray")
                ax.set_title(f"{tissue_name.upper()} Mask")
                ax.axis("off")
                mask_idx += 1

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    plt.show()


def plot_time_curves(
    curves: TimeCurves,
    trigger_times: TriggerTimes,
    labels: list[str],
    title: str = "Intensity/Time Curves",
    save_path: Path | None = None
) -> None:
    """
    Visualizza le curve intensit�/tempo per diversi tessuti.

    Parameters
    ----------
    curves : TimeCurves
        Array di curve, shape (n_curves, n_frames)
    trigger_times : TriggerTimes
        Tempi di trigger (secondi), shape (n_frames,)
    labels : list[str]
        Etichette per ogni curva
    title : str
        Titolo del grafico
    save_path : Path | None
        Se specificato, salva la figura in questo path

    Examples
    --------
    >>> curves = np.random.rand(4, 79)
    >>> times = np.arange(79) * 0.8
    >>> labels = ['RV', 'LV', 'Myocardium', 'Background']
    >>> plot_time_curves(curves, times, labels)
    """
    plt.figure(figsize=(12, 6))

    colors = ["blue", "red", "green", "gray"]

    for i, (curve, label) in enumerate(zip(curves, labels)):
        color = colors[i] if i < len(colors) else None
        plt.plot(trigger_times, curve, label=label, color=color, linewidth=2)

    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Signal Intensity", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    plt.show()
