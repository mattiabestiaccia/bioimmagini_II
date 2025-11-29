#!/usr/bin/env python3
"""
Script per ottimizzazione parametri K-means clustering.

Questo script testa sistematicamente diverse combinazioni di parametri per
l'algoritmo K-means al fine di trovare la configurazione ottimale:

Parametri testati:
- n_frames: numero di frame temporali da usare (10, 20, 30, 40, 50, ALL)
- distance: metrica di distanza ('euclidean', 'correlation')
- postprocessing: con/senza rimozione regioni spurie

L'ottimizzazione è basata sul DICE coefficient medio rispetto alle maschere
gold standard.

Usage
-----
    # Ottimizzazione completa (può richiedere tempo)
    python optimize_kmeans.py

    # Grid search veloce (meno combinazioni)
    python optimize_kmeans.py --quick

    # Test specifici parametri
    python optimize_kmeans.py --test-frames 20 30 40

Author: Bioimmagini Positano
Date: 2025-03-23
"""

import argparse
from itertools import product
import logging
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# Aggiungi src al path per import
sys.path.insert(0, str(Path(__file__).parent))

from .exceptions import (
    ClusteringError,
    DataLoadError,
    ValidationError,
)
from .kmeans_segmentation import perform_kmeans_clustering, postprocess_masks
from .utils import (
    dice_coefficient,
    identify_tissue_clusters,
    load_gold_standard,
    load_perfusion_series,
)


logger = logging.getLogger(__name__)


def test_configuration(
    image_stack: np.ndarray,
    gold_masks: dict[str, Any],
    n_frames: int,
    distance: str,
    postprocess: bool,
    n_init: int = 10,
    random_state: int = 42
) -> dict[str, Any]:
    """
    Testa una singola configurazione di parametri K-means.

    Parameters
    ----------
    image_stack : np.ndarray
        Stack di immagini (height, width, n_temporal_frames)
    gold_masks : dict
        Maschere gold standard
    n_frames : int
        Numero frame temporali da usare (None = tutti)
    distance : str
        Metrica distanza ('euclidean' o 'correlation')
    postprocess : bool
        Se True, applica post-processing
    n_init : int
        Numero inizializzazioni K-means
    random_state : int
        Random seed

    Returns
    -------
    results : dict
        Dizionario con risultati:
        - 'dice_rv', 'dice_lv', 'dice_myo': DICE per ogni tessuto
        - 'dice_mean': DICE medio
        - 'n_frames', 'distance', 'postprocess': parametri usati
    """
    # K-means clustering
    labels, centroids = perform_kmeans_clustering(
        image_stack,
        n_clusters=4,
        n_frames=n_frames,
        metric=distance,
        n_init=n_init,
        random_state=random_state
    )

    # Identifica tessuti
    tissue_map = identify_tissue_clusters(labels, centroids, n_clusters=4)

    # Crea maschere
    segmented_masks = {}
    for tissue, cluster_id in tissue_map.items():
        if tissue != "background":
            segmented_masks[tissue] = (labels == cluster_id)

    # Post-processing se richiesto
    if postprocess:
        segmented_masks = postprocess_masks(
            segmented_masks,
            min_size=50,
            keep_largest=True
        )

    # Calcola DICE scores
    dice_scores: dict[str, Any] = {}
    for tissue in ["rv", "lv", "myo"]:
        if tissue in segmented_masks and tissue in gold_masks:
            dice = dice_coefficient(segmented_masks[tissue], gold_masks[tissue])
            dice_scores[f"dice_{tissue}"] = dice

    # DICE medio
    if dice_scores:
        dice_scores["dice_mean"] = float(np.mean(list(dice_scores.values())))
    else:
        dice_scores["dice_mean"] = 0.0

    # Aggiungi parametri
    dice_scores["n_frames"] = n_frames if n_frames is not None else "all"
    dice_scores["distance"] = distance
    dice_scores["postprocess"] = postprocess

    return dice_scores


def plot_optimization_results(
    results_df: pd.DataFrame,
    save_dir: Path
) -> None:
    """
    Visualizza risultati ottimizzazione.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame con risultati ottimizzazione
    save_dir : Path
        Directory per salvare figure
    """
    # Figure 1: DICE vs n_frames per diverse configurazioni
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    distances = results_df["distance"].unique()
    postprocess_opts = results_df["postprocess"].unique()

    colors = {"euclidean": "blue", "correlation": "red"}
    markers = {True: "o", False: "s"}

    # Plot per ogni tessuto
    tissues = ["rv", "lv", "myo"]
    for idx, tissue in enumerate(tissues):
        ax = axes[idx // 2, idx % 2]

        for distance in distances:
            for postprocess in postprocess_opts:
                mask = (results_df["distance"] == distance) & \
                       (results_df["postprocess"] == postprocess)
                subset = results_df[mask].copy()

                # Converti 'all' in numero per plotting
                subset["n_frames_num"] = subset["n_frames"].apply(
                    lambda x: 79 if x == "all" else x
                )
                subset = subset.sort_values("n_frames_num")

                label = f"{distance}, {'post' if postprocess else 'no-post'}"
                ax.plot(
                    subset["n_frames_num"],
                    subset[f"dice_{tissue}"],
                    marker=markers[postprocess],
                    color=colors[distance],
                    label=label,
                    linewidth=2,
                    markersize=8
                )

        ax.set_xlabel("Number of Frames", fontsize=11)
        ax.set_ylabel(f"DICE - {tissue.upper()}", fontsize=11)
        ax.set_title(f"{tissue.upper()} Segmentation Quality", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

    # Plot DICE medio
    ax = axes[1, 1]
    for distance in distances:
        for postprocess in postprocess_opts:
            mask = (results_df["distance"] == distance) & \
                   (results_df["postprocess"] == postprocess)
            subset = results_df[mask].copy()

            subset["n_frames_num"] = subset["n_frames"].apply(
                lambda x: 79 if x == "all" else x
            )
            subset = subset.sort_values("n_frames_num")

            label = f"{distance}, {'post' if postprocess else 'no-post'}"
            ax.plot(
                subset["n_frames_num"],
                subset["dice_mean"],
                marker=markers[postprocess],
                color=colors[distance],
                label=label,
                linewidth=2,
                markersize=8
            )

    ax.set_xlabel("Number of Frames", fontsize=11)
    ax.set_ylabel("Mean DICE", fontsize=11)
    ax.set_title("Overall Segmentation Quality", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    plt.suptitle("K-means Parameter Optimization Results", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = save_dir / "optimization_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Figure saved: {save_path}")

    plt.show()


def main() -> int:
    """Funzione principale dello script."""
    parser = argparse.ArgumentParser(
        description="Ottimizzazione parametri K-means clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Ottimizzazione completa (lenta)
    python optimize_kmeans.py

    # Ottimizzazione veloce con meno combinazioni
    python optimize_kmeans.py --quick

    # Test frame specifici
    python optimize_kmeans.py --test-frames 20 30 40 50

    # Solo distanza euclidean
    python optimize_kmeans.py --test-distances euclidean
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
        "--quick",
        action="store_true",
        help="Modalità veloce: meno combinazioni da testare"
    )

    parser.add_argument(
        "--test-frames",
        nargs="+",
        type=int,
        default=None,
        help="Frame specifici da testare (es: 20 30 40)"
    )

    parser.add_argument(
        "--test-distances",
        nargs="+",
        choices=["euclidean", "correlation"],
        default=["euclidean", "correlation"],
        help="Metriche distanza da testare"
    )

    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Numero inizializzazioni K-means (default: 10)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Crea directory output
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OTTIMIZZAZIONE PARAMETRI K-MEANS - PERFUSIONE CARDIACA")
    print("=" * 70)

    # ===== Caricamento dati =====
    print("\n[1/3] Caricamento dati")

    try:
        image_stack, _ = load_perfusion_series(args.data_dir)
        n_total_frames = image_stack.shape[2]

        print(f"  ✓ Caricate {n_total_frames} immagini di dimensione "
              f"{image_stack.shape[0]}x{image_stack.shape[1]}")
    except DataLoadError as e:
        logger.error(f"Failed to load DICOM series: {e}")
        print(f"ERRORE: {e}")
        return 1

    # Carica gold standard
    try:
        gold_masks = load_gold_standard(args.gold_standard)
        print("  ✓ Gold standard caricato")
    except DataLoadError as e:
        logger.error(f"Failed to load gold standard: {e}")
        print(f"ERRORE: {e}")
        return 1

    # ===== Definizione grid search =====
    print("\n[2/3] Grid search configurazioni")

    # Configurazioni da testare
    if args.test_frames is not None:
        frames_to_test = args.test_frames
    elif args.quick:
        frames_to_test = [20, 40, None]  # None = tutti
    else:
        frames_to_test = [10, 20, 30, 40, 50, 60, None]

    distances_to_test = args.test_distances
    postprocess_opts = [False, True] if not args.quick else [True]

    # Grid
    configurations = list(product(
        frames_to_test,
        distances_to_test,
        postprocess_opts
    ))

    print("  Parametri:")
    print(f"    - N. frames:       {frames_to_test}")
    print(f"    - Distances:       {distances_to_test}")
    print(f"    - Postprocessing:  {postprocess_opts}")
    print(f"\n  Totale configurazioni da testare: {len(configurations)}")

    # ===== Esecuzione ottimizzazione =====
    print("\n  Esecuzione grid search...")

    results = []

    for n_frames, distance, postprocess in tqdm(configurations, desc="  Progress"):
        try:
            result = test_configuration(
                image_stack,
                gold_masks,
                n_frames,
                distance,
                postprocess,
                n_init=args.n_init,
                random_state=args.seed
            )
            results.append(result)
        except (ValidationError, ClusteringError) as e:
            logger.warning(f"Configuration failed (n_frames={n_frames}, "
                          f"distance={distance}, postprocess={postprocess}): {e}")
            print(f"\n  ⚠ Configurazione saltata: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in configuration (n_frames={n_frames}, "
                        f"distance={distance}, postprocess={postprocess}): {e}")
            print(f"\n  ⚠ Errore imprevisto: {e}")

    # Crea DataFrame
    results_df = pd.DataFrame(results)

    print(f"\n  ✓ Grid search completato: {len(results)} configurazioni testate")

    # ===== Risultati =====
    print("\n[3/3] Analisi risultati")

    # Trova configurazione migliore
    best_idx = results_df["dice_mean"].idxmax()
    best_config = results_df.loc[best_idx]

    print("\n  🏆 CONFIGURAZIONE OTTIMALE:")
    print(f"    N. frames:      {best_config['n_frames']}")
    print(f"    Distance:       {best_config['distance']}")
    print(f"    Postprocessing: {best_config['postprocess']}")
    print("\n  DICE Scores:")
    print(f"    RV:  {best_config['dice_rv']:.4f}")
    print(f"    LV:  {best_config['dice_lv']:.4f}")
    print(f"    Myo: {best_config['dice_myo']:.4f}")
    print(f"    Mean: {best_config['dice_mean']:.4f}")

    # Top 5 configurazioni
    print("\n  Top 5 Configurazioni:")
    top5 = results_df.nlargest(5, "dice_mean")[
        ["n_frames", "distance", "postprocess", "dice_mean", "dice_rv", "dice_lv", "dice_myo"]
    ]
    print(top5.to_string(index=False))

    # Salva risultati completi
    results_csv = args.output_dir / "optimization_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n  ✓ Risultati salvati: {results_csv}")

    # Visualizzazione
    print("\n  Generazione grafici...")
    plot_optimization_results(results_df, args.output_dir)

    print("\n" + "=" * 70)
    print("COMPLETATO!")
    print("=" * 70)
    print(f"\nRisultati salvati in: {args.output_dir}")
    print("  - optimization_results.csv")
    print("  - optimization_results.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
