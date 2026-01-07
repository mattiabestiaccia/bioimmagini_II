"""
Utilita' per registrazione immagini MRI con Algoritmi Genetici.

Pipeline:
    1. Caricamento immagini synthetic (BrainWeb MINC format)
    2. Estrazione slice 2D + zero padding
    3. Disallineamento random (roto-traslazione 2D)
    4. Registrazione con GA + Mutual Information
    5. Validazione con Bland-Altman plots

Algoritmo Genetico:
    - Ottimizzatore globale per evitare minimi locali
    - Fitness: -MI (massimizzazione MI = minimizzazione -MI)
    - Search space: tx,ty ∈ [-dim/10, +dim/10], angle ∈ [-60°, +60°]

Author: Generated with Claude Code
Date: 2025-11-20
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from scipy.ndimage import affine_transform, map_coordinates, shift, rotate
from scipy.optimize import differential_evolution
from sklearn.metrics import mutual_info_score
import warnings

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    warnings.warn("nibabel non installato. Installare con: pip install nibabel")


def load_minc_slice(
    minc_path: Path,
    slice_idx: int = 62
) -> np.ndarray:
    """
    Carica singola slice da file MINC (BrainWeb format).

    Args:
        minc_path: Path al file .mnc
        slice_idx: Indice slice da estrarre (default 62 = centrale)

    Returns:
        slice_2d: Array 2D slice estratta
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel richiesto per leggere MINC. Installare con: pip install nibabel")

    # Carica volume MINC
    img = nib.load(str(minc_path))
    volume = img.get_fdata()

    # Estrai slice (assumendo asse Z = 0)
    if volume.ndim == 3:
        slice_2d = volume[slice_idx, :, :]
    elif volume.ndim == 4:
        slice_2d = volume[slice_idx, :, :, 0]
    else:
        raise ValueError(f"Volume dimensione inattesa: {volume.shape}")

    return slice_2d.astype(np.float32)


def pad_to_square(image: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    """
    Zero padding per rendere immagine quadrata.

    Necessario per permettere roto-traslazioni senza perdere parti dell'immagine.

    Args:
        image: Immagine 2D
        pad_value: Valore padding (default 0)

    Returns:
        image_padded: Immagine quadrata con padding
    """
    h, w = image.shape
    max_dim = max(h, w)

    # Calcola padding
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2

    # Padding
    image_padded = np.pad(
        image,
        ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w)),
        mode='constant',
        constant_values=pad_value
    )

    return image_padded


def random_rigid_transform_2d(
    image_shape: Tuple[int, int],
    max_translation_fraction: float = 0.1,
    max_rotation_deg: float = 60.0,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Genera parametri random per trasformazione rigida 2D.

    Args:
        image_shape: Dimensioni immagine (h, w)
        max_translation_fraction: Frazione dimensione per max traslazione
        max_rotation_deg: Massima rotazione in gradi
        seed: Seed per riproducibilita'

    Returns:
        tx, ty, angle: Traslazione x, y (pixel) e rotazione (gradi)
    """
    if seed is not None:
        np.random.seed(seed)

    dim = max(image_shape)
    max_trans = dim * max_translation_fraction

    tx = np.random.uniform(-max_trans, max_trans)
    ty = np.random.uniform(-max_trans, max_trans)
    angle = np.random.uniform(-max_rotation_deg, max_rotation_deg)

    return tx, ty, angle


def apply_rigid_transform_2d(
    image: np.ndarray,
    tx: float,
    ty: float,
    angle_deg: float,
    order: int = 0
) -> np.ndarray:
    """
    Applica trasformazione rigida 2D (roto-traslazione).

    Ordine operazioni: prima rotazione (attorno al centro), poi traslazione.

    Args:
        image: Immagine 2D
        tx: Traslazione orizzontale in pixel (positivo = destra)
        ty: Traslazione verticale in pixel (positivo = basso)
        angle_deg: Rotazione in gradi (positivo = antiorario)
        order: Ordine interpolazione (0=NN, 1=linear, 3=cubic)

    Returns:
        image_transformed: Immagine trasformata
    """
    # Prima ruota attorno al centro
    image_rotated = rotate(
        image,
        angle_deg,
        reshape=False,
        order=order,
        mode='constant',
        cval=0.0
    )

    # Poi trasla: shift([row_shift, col_shift])
    # ty = shift verticale (righe), tx = shift orizzontale (colonne)
    image_transformed = shift(
        image_rotated,
        [ty, tx],
        order=order,
        mode='constant',
        cval=0.0
    )

    return image_transformed


def compute_mutual_information(
    image1: np.ndarray,
    image2: np.ndarray,
    bins: int = 256,
    normalized: bool = True,
    min_overlap_fraction: float = 0.3
) -> float:
    """
    Calcola Mutual Information tra due immagini.

    MI(I1, I2) = H(I1) + H(I2) - H(I1, I2)

    Dove H e' l'entropia.

    Args:
        image1: Prima immagine
        image2: Seconda immagine
        bins: Numero bin per istogramma (default 256)
        normalized: Se True, normalizza MI in [0,1]
        min_overlap_fraction: Frazione minima di sovrapposizione richiesta (default 0.3)

    Returns:
        mi: Mutual Information
    """
    # Maschera parti valide (entrambe non zero)
    mask = (image1 > 0) & (image2 > 0)

    # Conta pixel validi in ciascuna immagine
    valid_img1 = np.sum(image1 > 0)
    valid_img2 = np.sum(image2 > 0)
    overlap = np.sum(mask)

    # Penalizza se sovrapposizione troppo bassa
    # Usa il minimo tra le due immagini come riferimento
    if valid_img1 > 0 and valid_img2 > 0:
        overlap_fraction = overlap / min(valid_img1, valid_img2)
        if overlap_fraction < min_overlap_fraction:
            return 0.0

    if overlap == 0:
        return 0.0

    # Estrai valori validi
    img1_valid = image1[mask].ravel()
    img2_valid = image2[mask].ravel()

    # Calcola istogramma 2D
    hist_2d, _, _ = np.histogram2d(
        img1_valid,
        img2_valid,
        bins=bins,
        range=[[img1_valid.min(), img1_valid.max()],
               [img2_valid.min(), img2_valid.max()]]
    )

    # Normalizza istogramma
    pxy = hist_2d / np.sum(hist_2d)

    # Marginal distributions
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # Entropie
    # H(X) = -sum(p(x) * log(p(x)))
    px_py = px[:, None] * py[None, :]

    # Evita log(0)
    nonzero = pxy > 0

    # MI = sum(p(x,y) * log(p(x,y) / (p(x)*p(y))))
    mi = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))

    if normalized:
        # Normalized MI: MI / sqrt(H(X) * H(Y))
        hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
        hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
        if hx > 0 and hy > 0:
            mi = mi / np.sqrt(hx * hy)

    return mi


def fitness_function_mi(
    params: np.ndarray,
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    min_overlap_fraction: float = 0.5
) -> float:
    """
    Fitness function per GA: -MI (massimizzazione MI = minimizzazione -MI).

    Args:
        params: Array [tx, ty, angle] parametri trasformazione
        fixed_image: Immagine fissa di riferimento
        moving_image: Immagine mobile da registrare
        min_overlap_fraction: Minima sovrapposizione richiesta

    Returns:
        fitness: -MI (negativo per massimizzare MI), o penalita' alta se sovrapposizione insufficiente
    """
    tx, ty, angle = params

    # Applica trasformazione
    moving_transformed = apply_rigid_transform_2d(
        moving_image,
        tx, ty, angle,
        order=0  # Nearest neighbor come da specifiche
    )

    # Verifica sovrapposizione PRIMA del calcolo MI
    mask = (fixed_image > 0) & (moving_transformed > 0)
    valid_fixed = np.sum(fixed_image > 0)
    valid_moving = np.sum(moving_transformed > 0)
    overlap = np.sum(mask)

    # Penalizza FORTEMENTE se sovrapposizione insufficiente
    if valid_fixed > 0 and valid_moving > 0:
        overlap_fraction = overlap / min(valid_fixed, valid_moving)
        if overlap_fraction < min_overlap_fraction:
            # Ritorna penalita' proporzionale alla mancanza di overlap
            return 1.0 + (min_overlap_fraction - overlap_fraction)

    if overlap == 0:
        return 2.0

    # Calcola MI (non normalizzata per evitare instabilita')
    mi = compute_mutual_information(fixed_image, moving_transformed, bins=64, normalized=False)

    # Ritorna -MI per minimizzazione
    return -mi


def register_with_differential_evolution(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
    maxiter: int = 100,
    popsize: int = 15,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Registrazione con Differential Evolution (simile a GA).

    scipy non ha GA puro, ma Differential Evolution e' equivalente e spesso superiore.

    Args:
        fixed_image: Immagine fissa
        moving_image: Immagine mobile
        bounds: Search space [(tx_min, tx_max), (ty_min, ty_max), (angle_min, angle_max)]
        maxiter: Numero massimo generazioni
        popsize: Dimensione popolazione
        verbose: Stampa progresso

    Returns:
        result: Dizionario con parametri ottimali, MI, convergenza
    """
    # Bounds default se non specificati
    if bounds is None:
        dim = max(fixed_image.shape)
        max_trans = dim / 10
        bounds = [
            (-max_trans, max_trans),  # tx
            (-max_trans, max_trans),  # ty
            (-65.0, 65.0)             # angle (leggermente maggiore del max simulato)
        ]

    # Ottimizzazione con Differential Evolution
    result_de = differential_evolution(
        fitness_function_mi,
        bounds,
        args=(fixed_image, moving_image),
        maxiter=maxiter,
        popsize=popsize,
        strategy='best1bin',
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=None,
        disp=verbose,
        polish=True,  # Local refinement finale
        atol=1e-4,
        tol=0.01
    )

    # Parametri ottimali
    tx_opt, ty_opt, angle_opt = result_de.x

    # Immagine registrata
    moving_registered = apply_rigid_transform_2d(
        moving_image,
        tx_opt, ty_opt, angle_opt,
        order=0
    )

    # MI finale
    mi_final = compute_mutual_information(fixed_image, moving_registered)

    results = {
        'params': result_de.x,
        'tx': tx_opt,
        'ty': ty_opt,
        'angle': angle_opt,
        'mi_final': mi_final,
        'fitness': result_de.fun,
        'success': result_de.success,
        'nit': result_de.nit,
        'nfev': result_de.nfev,
        'moving_registered': moving_registered
    }

    return results


def bland_altman_stats(
    true_values: np.ndarray,
    estimated_values: np.ndarray
) -> Dict[str, float]:
    """
    Calcola statistiche per Bland-Altman plot.

    Args:
        true_values: Valori veri (ground truth)
        estimated_values: Valori stimati

    Returns:
        stats: Dizionario con mean_diff, std_diff, limits_of_agreement
    """
    diff = estimated_values - true_values
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # Limits of agreement: mean ± 1.96 * SD
    loa_lower = mean_diff - 1.96 * std_diff
    loa_upper = mean_diff + 1.96 * std_diff

    stats = {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'loa_lower': loa_lower,
        'loa_upper': loa_upper,
        'bias': mean_diff,
        'precision': std_diff
    }

    return stats
