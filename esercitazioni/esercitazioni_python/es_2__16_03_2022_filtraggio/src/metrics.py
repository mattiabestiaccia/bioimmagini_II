"""
Calcolo di metriche di qualità per immagini filtrate.

Include:
- SNR (Signal-to-Noise Ratio)
- Acutezza delle transizioni (Edge sharpness)
- Funzioni per ROI selection
"""

import numpy as np
from typing import Tuple, Optional
from scipy import ndimage


def calculate_snr(image: np.ndarray, roi_mask: np.ndarray) -> float:
    """
    Calcola il Signal-to-Noise Ratio (SNR) in una ROI.

    SNR = μ / σ

    dove:
    - μ è la media del segnale nella ROI
    - σ è la deviazione standard del rumore nella ROI

    Parameters
    ----------
    image : np.ndarray
        Immagine 2D o 3D
    roi_mask : np.ndarray
        Maschera booleana della stessa dimensione di image

    Returns
    -------
    snr : float
        Signal-to-Noise Ratio

    Notes
    -----
    Assumiamo che la ROI sia abbastanza grande e omogenea
    per stimare correttamente media e deviazione standard.
    """
    roi_values = image[roi_mask]

    # Calcola media (segnale)
    mean_signal = np.mean(roi_values)

    # Calcola deviazione standard (rumore)
    std_noise = np.std(roi_values, ddof=1)  # ddof=1 per stima campionaria

    # Calcola SNR
    if std_noise == 0:
        # Evita divisione per zero
        snr = np.inf
    else:
        snr = mean_signal / std_noise

    return snr


def calculate_cnr(
    image: np.ndarray,
    roi_signal: np.ndarray,
    roi_background: np.ndarray
) -> float:
    """
    Calcola il Contrast-to-Noise Ratio (CNR) tra due ROI.

    CNR = |μ_signal - μ_background| / σ_background

    Parameters
    ----------
    image : np.ndarray
        Immagine 2D o 3D
    roi_signal : np.ndarray
        Maschera booleana per la ROI del segnale
    roi_background : np.ndarray
        Maschera booleana per la ROI del background

    Returns
    -------
    cnr : float
        Contrast-to-Noise Ratio

    Notes
    -----
    Utile per valutare il contrasto tra due regioni (es. fantoccio vs sfondo).
    """
    signal_values = image[roi_signal]
    background_values = image[roi_background]

    mean_signal = np.mean(signal_values)
    mean_background = np.mean(background_values)
    std_background = np.std(background_values, ddof=1)

    if std_background == 0:
        cnr = np.inf
    else:
        cnr = abs(mean_signal - mean_background) / std_background

    return cnr


def create_circular_roi(
    image_shape: Tuple[int, int],
    center: Tuple[int, int],
    radius: int
) -> np.ndarray:
    """
    Crea una ROI circolare per immagini 2D.

    Parameters
    ----------
    image_shape : tuple of int
        Shape dell'immagine (rows, cols)
    center : tuple of int
        Centro della ROI (row, col)
    radius : int
        Raggio della ROI in pixel

    Returns
    -------
    roi_mask : np.ndarray (bool)
        Maschera booleana 2D con True dentro la ROI

    Examples
    --------
    >>> roi = create_circular_roi((512, 512), (256, 256), 50)
    >>> snr = calculate_snr(image, roi)
    """
    rows, cols = image_shape
    center_row, center_col = center

    # Crea griglia di coordinate
    y, x = np.ogrid[0:rows, 0:cols]

    # Calcola distanza dal centro
    dist_from_center = np.sqrt((y - center_row)**2 + (x - center_col)**2)

    # Crea maschera
    roi_mask = dist_from_center <= radius

    return roi_mask


def create_rectangular_roi(
    image_shape: Tuple[int, int],
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int]
) -> np.ndarray:
    """
    Crea una ROI rettangolare per immagini 2D.

    Parameters
    ----------
    image_shape : tuple of int
        Shape dell'immagine (rows, cols)
    top_left : tuple of int
        Angolo superiore sinistro (row, col)
    bottom_right : tuple of int
        Angolo inferiore destro (row, col)

    Returns
    -------
    roi_mask : np.ndarray (bool)
        Maschera booleana 2D con True dentro la ROI
    """
    rows, cols = image_shape
    roi_mask = np.zeros((rows, cols), dtype=bool)

    r1, c1 = top_left
    r2, c2 = bottom_right

    roi_mask[r1:r2, c1:c2] = True

    return roi_mask


def extract_profile(image: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> np.ndarray:
    """
    Estrae un profilo lineare da un'immagine 2D.

    Parameters
    ----------
    image : np.ndarray
        Immagine 2D
    start : tuple of int
        Punto iniziale (row, col)
    end : tuple of int
        Punto finale (row, col)

    Returns
    -------
    profile : np.ndarray
        Valori lungo il profilo

    Notes
    -----
    Usa interpolazione bilineare per estrarre valori tra pixel.
    """
    # Numero di punti lungo il profilo
    length = int(np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2))

    # Coordinate lungo il profilo
    rows = np.linspace(start[0], end[0], length)
    cols = np.linspace(start[1], end[1], length)

    # Estrai valori con interpolazione
    profile = ndimage.map_coordinates(image, [rows, cols], order=1, mode='nearest')

    return profile


def calculate_edge_sharpness(profile: np.ndarray, method: str = 'gradient') -> float:
    """
    Calcola l'acutezza di una transizione (edge) in un profilo.

    Parameters
    ----------
    profile : np.ndarray
        Profilo 1D attraverso una transizione
    method : str, default='gradient'
        Metodo per calcolare acutezza:
        - 'gradient': massimo gradiente
        - 'width': larghezza transizione (10%-90%)
        - 'slope': pendenza nella regione centrale

    Returns
    -------
    sharpness : float
        Metrica di acutezza (più alto = più acuto)

    Notes
    -----
    Per 'gradient': sharpness = max(|dI/dx|)
    Per 'width': sharpness = 1 / width_10_90
    Per 'slope': sharpness = (I_90 - I_10) / width
    """
    if method == 'gradient':
        # Calcola gradiente
        gradient = np.gradient(profile)
        # Acutezza = massimo valore assoluto del gradiente
        sharpness = np.max(np.abs(gradient))

    elif method == 'width':
        # Calcola larghezza transizione al 10%-90%
        pmin, pmax = profile.min(), profile.max()
        p10 = pmin + 0.1 * (pmax - pmin)
        p90 = pmin + 0.9 * (pmax - pmin)

        # Trova punti dove profile attraversa p10 e p90
        idx_10 = np.where(profile <= p10)[0]
        idx_90 = np.where(profile >= p90)[0]

        if len(idx_10) > 0 and len(idx_90) > 0:
            # Prendi gli indici più vicini alla transizione
            if idx_10[-1] < idx_90[0]:
                # Transizione crescente
                width = idx_90[0] - idx_10[-1]
            else:
                # Transizione decrescente
                width = idx_10[0] - idx_90[-1]

            if width > 0:
                sharpness = 1.0 / width
            else:
                sharpness = np.inf
        else:
            sharpness = 0.0

    elif method == 'slope':
        # Calcola pendenza nella regione centrale
        pmin, pmax = profile.min(), profile.max()
        p10 = pmin + 0.1 * (pmax - pmin)
        p90 = pmin + 0.9 * (pmax - pmin)

        delta_signal = abs(p90 - p10)

        # Trova larghezza
        idx_10 = np.where(abs(profile - p10) == np.min(abs(profile - p10)))[0]
        idx_90 = np.where(abs(profile - p90) == np.min(abs(profile - p90)))[0]

        if len(idx_10) > 0 and len(idx_90) > 0:
            width = abs(idx_90[0] - idx_10[0])
            if width > 0:
                sharpness = delta_signal / width
            else:
                sharpness = np.inf
        else:
            sharpness = 0.0

    else:
        raise ValueError(f"Metodo '{method}' non riconosciuto. Usa 'gradient', 'width', o 'slope'.")

    return sharpness


def calculate_multiple_metrics(
    image: np.ndarray,
    roi_mask: np.ndarray,
    profile_start: Tuple[int, int],
    profile_end: Tuple[int, int],
    sharpness_method: str = 'gradient'
) -> dict:
    """
    Calcola SNR e acutezza per un'immagine.

    Funzione di convenienza per calcolare tutte le metriche insieme.

    Parameters
    ----------
    image : np.ndarray
        Immagine 2D
    roi_mask : np.ndarray
        Maschera ROI per SNR
    profile_start : tuple of int
        Punto iniziale profilo per acutezza
    profile_end : tuple of int
        Punto finale profilo per acutezza
    sharpness_method : str, default='gradient'
        Metodo per calcolo acutezza

    Returns
    -------
    metrics : dict
        Dizionario con 'snr' e 'sharpness'
    """
    snr = calculate_snr(image, roi_mask)
    profile = extract_profile(image, profile_start, profile_end)
    sharpness = calculate_edge_sharpness(profile, method=sharpness_method)

    metrics = {
        'snr': snr,
        'sharpness': sharpness,
        'profile': profile
    }

    return metrics
