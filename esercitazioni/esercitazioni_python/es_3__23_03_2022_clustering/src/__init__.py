"""
Esercitazione 3: K-means Clustering per Segmentazione Cardiaca MRI.

Questo package fornisce strumenti per la segmentazione automatica di strutture
cardiache (ventricolo destro, ventricolo sinistro, miocardio) su immagini MRI
di perfusione first-pass usando l'algoritmo K-means clustering.

Modules
-------
utils
    Funzioni utility per caricamento dati, DICE index, post-processing
kmeans_segmentation
    Script principale per segmentazione K-means
plot_time_curves
    Visualizzazione curve intensita/tempo
optimize_kmeans
    Ottimizzazione parametri K-means

Examples
--------
    # Import funzioni utility
    from utils import load_perfusion_series, dice_coefficient

    # Carica dati
    images, times = load_perfusion_series(Path("data/perfusione"))

    # Calcola DICE
    dice = dice_coefficient(mask1, mask2)
"""

__version__ = "1.0.0"
__author__ = "Bioimmagini Positano"
__date__ = "2025-03-23"

from .utils import (
    crop_to_roi,
    dice_coefficient,
    extract_pixel_time_curves,
    identify_tissue_clusters,
    keep_largest_component,
    load_gold_standard,
    load_perfusion_series,
    plot_time_curves,
    remove_small_regions,
    visualize_segmentation,
)


__all__ = [
    "crop_to_roi",
    "dice_coefficient",
    "extract_pixel_time_curves",
    "identify_tissue_clusters",
    "keep_largest_component",
    "load_gold_standard",
    "load_perfusion_series",
    "plot_time_curves",
    "remove_small_regions",
    "visualize_segmentation"
]
