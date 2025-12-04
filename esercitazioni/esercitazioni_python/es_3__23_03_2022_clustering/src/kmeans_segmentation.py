#!/usr/bin/env python3
"""
Segmentazione automatica di strutture cardiache usando K-means clustering.

Questo script implementa la segmentazione automatica del miocardio,
ventricolo sinistro e ventricolo destro su immagini MRI di perfusione cardiaca
first-pass usando l'algoritmo K-means clustering.

L'algoritmo classifica ogni pixel basandosi sulla sua curva intensità/tempo,
identificando automaticamente le quattro classi principali:
1. Background (nessun contrasto)
2. Ventricolo destro (RV) - picco precoce
3. Ventricolo sinistro (LV) - picco intermedio
4. Miocardio - picco tardivo

La qualità della segmentazione viene valutata usando il DICE coefficient
confrontandola con maschere gold standard.

Usage
-----
    # Segmentazione con parametri default
    python kmeans_segmentation.py

    # Con parametri personalizzati
    python kmeans_segmentation.py --n-frames 40 --distance correlation

    # Con ROI crop per velocizzare
    python kmeans_segmentation.py --crop-roi 50 200 50 200

Author: Bioimmagini Positano
Date: 2025-03-23
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans


# Aggiungi src al path per import
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .exceptions import (
        ClusteringError,
        DataLoadError,
        ValidationError,
    )
except ImportError:
    from exceptions import (
        ClusteringError,
        DataLoadError,
        ValidationError,
    )
try:
    from .custom_types import BinaryMask, Centroids, ClusterLabels, ImageStack
except ImportError:
    from custom_types import BinaryMask, Centroids, ClusterLabels, ImageStack
try:
    from .utils import (
        crop_to_roi,
        dice_coefficient,
        identify_tissue_clusters,
        keep_largest_component,
        load_gold_standard,
        load_perfusion_series,
        remove_small_regions,
        visualize_segmentation,
    )
except ImportError:
    from utils import (
        crop_to_roi,
        dice_coefficient,
        identify_tissue_clusters,
        keep_largest_component,
        load_gold_standard,
        load_perfusion_series,
        remove_small_regions,
        visualize_segmentation,
    )


logger = logging.getLogger(__name__)


def perform_kmeans_clustering(
    image_stack: ImageStack,
    n_clusters: int = 4,
    n_frames: int | None = None,
    metric: str = "euclidean",
    n_init: int = 10,
    random_state: int = 42
) -> tuple[ClusterLabels, Centroids]:
    """
    Esegue K-means clustering su serie temporale di immagini.

    Parameters
    ----------
    image_stack : ImageStack
        Stack di immagini, shape (height, width, n_temporal_frames)
    n_clusters : int
        Numero di cluster (default: 4)
    n_frames : int | None
        Numero di frame temporali da usare per clustering.
        Se None, usa tutti i frame.
    metric : str
        Metrica di distanza: 'euclidean', 'correlation'
    n_init : int
        Numero di inizializzazioni K-means
    random_state : int
        Seed per riproducibilità

    Returns
    -------
    labels : ClusterLabels
        Etichette cluster per ogni pixel, shape (height, width)
    centroids : Centroids
        Centroidi dei cluster, shape (n_clusters, n_frames_used)

    Raises
    ------
    ValidationError
        Se i parametri non sono validi
    ClusteringError
        Se il clustering fallisce
    """
    logger.info(f"Starting K-means clustering with {n_clusters} clusters, metric={metric}")

    # Validation
    if n_clusters < 1:
        error_msg = f"Invalid number of clusters: {n_clusters} (must be >= 1)"
        logger.error(error_msg)
        raise ValidationError(error_msg)

    height, width, n_temporal_frames = image_stack.shape

    # Usa solo primi n_frames se specificato
    if n_frames is not None and n_frames < n_temporal_frames:
        data = image_stack[:, :, :n_frames]
        logger.debug(f"Using first {n_frames} of {n_temporal_frames} frames")
    else:
        data = image_stack
        n_frames = n_temporal_frames

    # Reshape: (height * width, n_frames)
    X = data.reshape(-1, n_frames)
    logger.debug(f"Reshaped data to {X.shape} for clustering")

    # Normalizzazione per correlation metric
    if metric == "correlation":
        logger.debug("Normalizing data for correlation distance")
        # Normalizza ogni curva (pixel)
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)
        stds[stds == 0] = 1  # Evita divisione per zero
        X_normalized = (X - means) / stds
    else:
        X_normalized = X

    # K-means clustering
    try:
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
            algorithm="lloyd"
        )

        labels_flat = kmeans.fit_predict(X_normalized)
        centroids = kmeans.cluster_centers_
    except Exception as e:
        error_msg = "K-means clustering failed"
        logger.error(f"{error_msg}: {e}")
        raise ClusteringError(f"{error_msg}: {e}")

    # Reshape labels
    labels = labels_flat.reshape(height, width)

    # Se usata normalizzazione, de-normalizza i centroidi per interpretazione
    if metric == "correlation":
        # I centroidi sono normalizzati, riportiamo alle intensità originali
        # usando media/std globali approssimative
        global_mean = X.mean()
        global_std = X.std()
        centroids = centroids * global_std + global_mean

    logger.info("K-means clustering completed successfully")
    return labels, centroids


def postprocess_masks(
    masks: dict[str, BinaryMask],
    min_size: int = 50,
    keep_largest: bool = True
) -> dict[str, BinaryMask]:
    """
    Post-processing delle maschere per rimuovere regioni spurie.

    Parameters
    ----------
    masks : dict[str, BinaryMask]
        Dizionario con maschere binarie
    min_size : int
        Dimensione minima regioni da mantenere
    keep_largest : bool
        Se True, mantiene solo la componente connessa più grande

    Returns
    -------
    cleaned_masks : dict[str, BinaryMask]
        Maschere pulite
    """
    cleaned_masks = {}

    for tissue_name, mask in masks.items():
        cleaned = mask.copy()

        # Rimuovi piccole regioni
        if min_size > 0:
            cleaned = remove_small_regions(cleaned, min_size=min_size)

        # Mantieni solo componente più grande
        if keep_largest:
            cleaned = keep_largest_component(cleaned)

        cleaned_masks[tissue_name] = cleaned

    return cleaned_masks


def main() -> int:
    """Funzione principale dello script."""
    parser = argparse.ArgumentParser(
        description="Segmentazione K-means di strutture cardiache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Segmentazione base
    python kmeans_segmentation.py

    # Usa solo primi 40 frame e distanza correlation
    python kmeans_segmentation.py --n-frames 40 --distance correlation

    # Con crop ROI per velocizzare
    python kmeans_segmentation.py --crop-roi 50 200 50 200

    # Senza post-processing
    python kmeans_segmentation.py --no-postprocess

    # Più inizializzazioni K-means
    python kmeans_segmentation.py --n-init 20
        """
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "perfusione",
        help="Directory contenente i file DICOM"
    )

    parser.add_argument(
        "--gold-standard",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "GoldStandard.mat",
        help="File MATLAB con maschere gold standard"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Directory per salvare i risultati"
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Numero di cluster (default: 4)"
    )

    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Numero di frame temporali da usare (default: tutti)"
    )

    parser.add_argument(
        "--distance",
        choices=["euclidean", "correlation"],
        default="euclidean",
        help="Metrica di distanza (default: euclidean)"
    )

    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Numero di inizializzazioni K-means (default: 10)"
    )

    parser.add_argument(
        "--crop-roi",
        nargs=4,
        type=int,
        metavar=("ROW_START", "ROW_END", "COL_START", "COL_END"),
        help="ROI per crop (es: 50 200 50 200)"
    )

    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Disabilita post-processing maschere"
    )

    parser.add_argument(
        "--min-region-size",
        type=int,
        default=50,
        help="Dimensione minima regioni (pixel) per post-processing"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed per riproducibilità"
    )

    args = parser.parse_args()

    # Crea directory output
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SEGMENTAZIONE K-MEANS - PERFUSIONE CARDIACA")
    print("=" * 70)

    # ===== Caricamento dati =====
    print("\n[1/5] Caricamento dati")

    try:
        image_stack, trigger_times = load_perfusion_series(args.data_dir)
        print(f"  ✓ Caricate {image_stack.shape[2]} immagini di dimensione "
              f"{image_stack.shape[0]}x{image_stack.shape[1]}")
    except DataLoadError as e:
        logger.error(f"Failed to load DICOM series: {e}")
        print(f"ERRORE: {e}")
        return 1

    # Crop ROI se specificato
    if args.crop_roi:
        print(f"  → Crop ROI: {args.crop_roi}")
        roi_tuple = tuple(args.crop_roi)
        try:
            image_stack = crop_to_roi(image_stack, roi_tuple)
            print(f"  ✓ Nuove dimensioni: {image_stack.shape[0]}x{image_stack.shape[1]}")
        except ValidationError as e:
            logger.error(f"Failed to crop ROI: {e}")
            print(f"ERRORE: {e}")
            return 1

    # Carica gold standard
    try:
        gold_masks = load_gold_standard(args.gold_standard)
        print("  ✓ Gold standard caricato")

        # Crop anche le maschere se necessario
        if args.crop_roi:
            for tissue in gold_masks:
                gold_masks[tissue] = crop_to_roi(gold_masks[tissue], roi_tuple)
    except DataLoadError as e:
        logger.warning(f"Gold standard not available: {e}")
        print(f"  ⚠ Gold standard non trovato: {args.gold_standard}")
        gold_masks = None

    # ===== K-means clustering =====
    print("\n[2/5] K-means clustering")
    print("  Parametri:")
    print(f"    - Numero cluster:  {args.n_clusters}")
    print(f"    - Frame temporali: {args.n_frames if args.n_frames else 'tutti'}")
    print(f"    - Distanza:        {args.distance}")
    print(f"    - Inizializzazioni: {args.n_init}")
    print(f"    - Random seed:     {args.seed}")

    try:
        labels, centroids = perform_kmeans_clustering(
            image_stack,
            n_clusters=args.n_clusters,
            n_frames=args.n_frames,
            metric=args.distance,
            n_init=args.n_init,
            random_state=args.seed
        )
        print("  ✓ Clustering completato")
    except (ValidationError, ClusteringError) as e:
        logger.error(f"Clustering failed: {e}")
        print(f"ERRORE: {e}")
        return 1

    # ===== Identificazione tessuti =====
    print("\n[3/5] Identificazione tessuti")

    try:
        tissue_map = identify_tissue_clusters(labels, centroids, args.n_clusters)

        print("  Mappatura cluster → tessuto:")
        for tissue, cluster_id in tissue_map.items():
            print(f"    {tissue.upper():12s} → Cluster {cluster_id}")
    except ValidationError as e:
        logger.error(f"Tissue identification failed: {e}")
        print(f"ERRORE: {e}")
        return 1

    # Crea maschere binarie per ogni tessuto
    segmented_masks = {}
    for tissue, cluster_id in tissue_map.items():
        if tissue != "background":
            segmented_masks[tissue] = (labels == cluster_id)

    # ===== Post-processing =====
    print("\n[4/5] Post-processing")

    if not args.no_postprocess:
        print(f"  → Rimozione regioni < {args.min_region_size} pixel")
        print("  → Mantenimento componente più grande")

        segmented_masks = postprocess_masks(
            segmented_masks,
            min_size=args.min_region_size,
            keep_largest=True
        )

        print("  ✓ Post-processing completato")
    else:
        print("  ⊗ Post-processing disabilitato")

    # ===== Valutazione DICE =====
    print("\n[5/5] Valutazione qualità (DICE coefficient)")

    if gold_masks is not None:
        dice_scores = {}

        for tissue in ["rv", "lv", "myo"]:
            if tissue in segmented_masks and tissue in gold_masks:
                dice = dice_coefficient(segmented_masks[tissue], gold_masks[tissue])
                dice_scores[tissue] = dice
                print(f"  {tissue.upper():3s}: DICE = {dice:.4f} "
                      f"({'Excellent' if dice > 0.9 else 'Good' if dice > 0.7 else 'Fair' if dice > 0.5 else 'Poor'})")

        # DICE medio
        if dice_scores:
            mean_dice = np.mean(list(dice_scores.values()))
            print(f"\n  Media: DICE = {mean_dice:.4f}")
    else:
        print("  ⊗ Impossibile calcolare DICE (gold standard non disponibile)")

    # ===== Visualizzazione =====
    print("\nGenerazione visualizzazioni...")

    # Usa immagine al picco del contrasto per visualizzazione
    peak_frame = 12 if image_stack.shape[2] > 12 else image_stack.shape[2] // 2
    display_image = image_stack[:, :, peak_frame]

    # Segmentazione
    save_path = args.output_dir / "kmeans_segmentation.png"
    visualize_segmentation(
        display_image,
        segmented_masks,
        title=f"K-means Segmentation (n_frames={args.n_frames or 'all'}, "
              f"distance={args.distance})",
        save_path=save_path
    )

    # Gold standard (se disponibile)
    if gold_masks is not None:
        save_path_gold = args.output_dir / "gold_standard.png"
        visualize_segmentation(
            display_image,
            {k: v for k, v in gold_masks.items() if k in ["rv", "lv", "myo"]},
            title="Gold Standard Segmentation",
            save_path=save_path_gold
        )

    # Salva maschere come numpy arrays
    np.savez(
        args.output_dir / "segmentation_masks.npz",
        rv=segmented_masks.get("rv", np.array([])),
        lv=segmented_masks.get("lv", np.array([])),
        myo=segmented_masks.get("myo", np.array([])),
        labels=labels,
        centroids=centroids
    )

    print("\n" + "=" * 70)
    print("COMPLETATO!")
    print("=" * 70)
    print(f"\nRisultati salvati in: {args.output_dir}")
    print("  - kmeans_segmentation.png")
    print("  - segmentation_masks.npz")

    if gold_masks is not None:
        print("  - gold_standard.png")
        print("\nQualità segmentazione (DICE):")
        for tissue, dice in dice_scores.items():
            print(f"  {tissue.upper()}: {dice:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
