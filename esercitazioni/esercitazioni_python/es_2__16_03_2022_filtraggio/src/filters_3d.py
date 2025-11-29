"""
Implementazione di filtri 3D per immagini CT.

Include filtri:
- Media mobile 3D
- Gaussiano 3D
- Wiener adattivo 3D
- Funzioni di supporto (stdfilt3, varianza locale)
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter, generic_filter
from typing import Tuple


def moving_average_filter_3d(volume: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Applica un filtro a media mobile 3D.

    Equivalente a MATLAB: fspecial3('average', [k,k,k]) + imfilter
    o conv3 con kernel uniforme normalizzato.

    Parameters
    ----------
    volume : np.ndarray
        Volume 3D da filtrare
    kernel_size : int, default=7
        Dimensione del kernel cubico (kernel_size x kernel_size x kernel_size)

    Returns
    -------
    filtered : np.ndarray
        Volume filtrato

    Notes
    -----
    Il filtro a media mobile sostituisce ogni voxel con la media
    dei voxel nella sua vicinanza (kernel cubico).
    """
    # scipy.ndimage.uniform_filter è equivalente a un filtro a media mobile
    # mode='nearest' gestisce i bordi replicando i pixel ai bordi
    filtered = uniform_filter(volume, size=kernel_size, mode='nearest')

    return filtered


def gaussian_filter_3d(volume: np.ndarray, kernel_size: int = 7, sigma: float = 1.0) -> np.ndarray:
    """
    Applica un filtro Gaussiano 3D.

    Equivalente a MATLAB: imgaussfilt3(volume, sigma) o
    fspecial3('gaussian', [k,k,k], sigma) + imfilter.

    Parameters
    ----------
    volume : np.ndarray
        Volume 3D da filtrare
    kernel_size : int, default=7
        Dimensione del kernel (usato per determinare truncate)
    sigma : float, default=1.0
        Deviazione standard della Gaussiana

    Returns
    -------
    filtered : np.ndarray
        Volume filtrato

    Notes
    -----
    Il parametro truncate determina quanto il kernel Gaussiano viene troncato.
    truncate = (kernel_size - 1) / (2 * sigma) per ottenere la dimensione desiderata.
    """
    # Calcola truncate per ottenere un kernel della dimensione desiderata
    # La dimensione effettiva del kernel è: 2 * int(truncate * sigma) + 1
    # Quindi: truncate = (kernel_size - 1) / (2 * sigma)
    truncate = (kernel_size - 1) / (2.0 * sigma)

    filtered = ndimage.gaussian_filter(
        volume,
        sigma=sigma,
        mode='nearest',
        truncate=truncate
    )

    return filtered


def stdfilt3(volume: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Calcola la deviazione standard locale in 3D.

    Equivalente a MATLAB stdfilt ma in 3D.
    Calcola la deviazione standard in una finestra mobile cubica.

    Parameters
    ----------
    volume : np.ndarray
        Volume 3D di input
    kernel_size : int, default=7
        Dimensione del kernel cubico

    Returns
    -------
    std_map : np.ndarray
        Mappa 3D della deviazione standard locale

    Notes
    -----
    Implementazione efficiente usando:
    Var(X) = E[X²] - E[X]²

    Dove E[X] è la media locale (filtro uniforme).
    """
    # Calcola media locale: E[X]
    mean_local = uniform_filter(volume, size=kernel_size, mode='nearest')

    # Calcola media di X²: E[X²]
    mean_of_squares = uniform_filter(volume**2, size=kernel_size, mode='nearest')

    # Varianza locale: Var(X) = E[X²] - E[X]²
    variance_local = mean_of_squares - mean_local**2

    # Assicurati che la varianza sia non-negativa (errori numerici)
    variance_local = np.maximum(variance_local, 0.0)

    # Deviazione standard locale
    std_local = np.sqrt(variance_local)

    return std_local


def variance_map_3d(volume: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Calcola la varianza locale in 3D.

    Wrapper per stdfilt3 che restituisce varianza invece di std.

    Parameters
    ----------
    volume : np.ndarray
        Volume 3D di input
    kernel_size : int, default=7
        Dimensione del kernel cubico

    Returns
    -------
    var_map : np.ndarray
        Mappa 3D della varianza locale (σ²)
    """
    std_map = stdfilt3(volume, kernel_size)
    var_map = std_map ** 2
    return var_map


def wiener_filter_3d(
    volume: np.ndarray,
    kernel_size: int = 7,
    noise_variance: float = None,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Applica un filtro Wiener adattivo 3D.

    Implementazione della formula:
    I_W = I_MM + α(I_OR - I_MM)

    dove α = (I_VAR - σ²) / I_VAR  se I_VAR >= σ²
              0                      altrimenti

    Parameters
    ----------
    volume : np.ndarray
        Volume 3D originale (I_OR)
    kernel_size : int, default=7
        Dimensione del kernel cubico
    noise_variance : float, optional
        Varianza del rumore (σ²). Se None, viene stimata automaticamente.
    epsilon : float, default=1e-10
        Piccola costante per evitare divisione per zero

    Returns
    -------
    filtered : np.ndarray
        Volume filtrato con Wiener

    Notes
    -----
    Il filtro Wiener è adattivo:
    - Nelle aree omogenee (I_VAR ≈ σ²) applica media mobile
    - Nei contorni (I_VAR >> σ²) preserva i dettagli
    - Gestisce il caso I_VAR < σ² applicando media mobile

    La varianza del rumore σ² dovrebbe essere stimata in una ROI
    omogenea del fantoccio.
    """
    # Step 1: Calcola immagine filtrata a media mobile (I_MM)
    filtered_mean = moving_average_filter_3d(volume, kernel_size)

    # Step 2: Calcola mappa di varianza locale (I_VAR)
    variance_local = variance_map_3d(volume, kernel_size)

    # Step 3: Stima varianza del rumore se non fornita
    if noise_variance is None:
        # Stima automatica: usa la mediana della varianza locale
        # (la mediana è robusta agli outliers dei contorni)
        noise_variance = np.median(variance_local)
        print(f"Varianza rumore stimata automaticamente: {noise_variance:.4f}")

    # Step 4: Calcola coefficiente α
    # α = (I_VAR - σ²) / I_VAR  se I_VAR >= σ²
    # α = 0                      altrimenti

    # Aggiungi epsilon per evitare divisione per zero
    variance_local_safe = variance_local + epsilon

    # Calcola α con condizione
    alpha = np.zeros_like(variance_local)
    mask_valid = variance_local >= noise_variance

    alpha[mask_valid] = (
        (variance_local[mask_valid] - noise_variance) / variance_local_safe[mask_valid]
    )

    # Step 5: Applica filtro Wiener
    # I_W = I_MM + α(I_OR - I_MM)
    filtered_wiener = filtered_mean + alpha * (volume - filtered_mean)

    return filtered_wiener


def estimate_noise_variance(volume: np.ndarray, roi_mask: np.ndarray = None) -> float:
    """
    Stima la varianza del rumore in un volume 3D.

    Parameters
    ----------
    volume : np.ndarray
        Volume 3D
    roi_mask : np.ndarray, optional
        Maschera booleana 3D per selezionare la ROI.
        Se None, usa tutto il volume.

    Returns
    -------
    noise_var : float
        Varianza del rumore stimata

    Notes
    -----
    Se viene fornita una ROI, la varianza viene calcolata solo in quella regione.
    Questo è il metodo preferito per stimare il rumore in una regione omogenea.
    """
    if roi_mask is not None:
        roi_values = volume[roi_mask]
    else:
        roi_values = volume.flatten()

    noise_var = np.var(roi_values, ddof=1)  # ddof=1 per varianza campionaria
    return noise_var


def estimate_noise_std(volume: np.ndarray, roi_mask: np.ndarray = None) -> float:
    """
    Stima la deviazione standard del rumore in un volume 3D.

    Wrapper per estimate_noise_variance che restituisce std invece di varianza.

    Parameters
    ----------
    volume : np.ndarray
        Volume 3D
    roi_mask : np.ndarray, optional
        Maschera booleana 3D per selezionare la ROI

    Returns
    -------
    noise_std : float
        Deviazione standard del rumore stimata (σ)
    """
    noise_var = estimate_noise_variance(volume, roi_mask)
    noise_std = np.sqrt(noise_var)
    return noise_std
