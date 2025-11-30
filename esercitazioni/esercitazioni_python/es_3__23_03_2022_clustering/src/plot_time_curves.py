#!/usr/bin/env python3
"""
Script per visualizzare le curve intensità/tempo di diversi tessuti cardiaci.

Questo script carica le immagini di perfusione DICOM e visualizza le curve
di intensità temporale per pixel rappresentativi di:
- Ventricolo destro (RV)
- Ventricolo sinistro (LV)
- Miocardio
- Background (tessuto non interessato dal contrasto)

Le curve mostrano come il mezzo di contrasto (Gadolinio) diffonde attraverso
le diverse strutture cardiache durante l'acquisizione first-pass perfusion.

Usage
-----
    python plot_time_curves.py

    # Con coordinate pixel personalizzate
    python plot_time_curves.py --rv-pixel 100 120 --lv-pixel 130 140

Author: Bioimmagini Positano
Date: 2025-03-23
"""

import argparse
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# Aggiungi src al path per import
sys.path.insert(0, str(Path(__file__).parent))

from .exceptions import DataLoadError
from .utils import extract_pixel_time_curves, load_perfusion_series, plot_time_curves


logger = logging.getLogger(__name__)


def select_representative_pixels_interactive(
    image_stack: np.ndarray
) -> dict:
    """
    Permette all'utente di selezionare interattivamente pixel rappresentativi.

    Parameters
    ----------
    image_stack : np.ndarray
        Stack di immagini (height, width, n_frames)

    Returns
    -------
    pixels : dict
        Dizionario con coordinate pixel per ogni tessuto
    """
    # Mostra immagine al picco del contrasto (circa frame 10-15)
    peak_frame = 12
    img = image_stack[:, :, peak_frame]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap="gray")
    ax.set_title("Click to select pixels\n(RV, LV, Myocardium, Background)")
    ax.axis("off")

    print("\nSelect 4 pixels by clicking on the image:")
    print("1. Right Ventricle (RV)")
    print("2. Left Ventricle (LV)")
    print("3. Myocardium")
    print("4. Background")

    coords = plt.ginput(4, timeout=0)
    plt.close(fig)

    if len(coords) != 4:
        print("Error: Expected 4 clicks, using default coordinates")
        return get_default_pixel_coordinates()

    pixels = {
        "rv": (int(coords[0][1]), int(coords[0][0])),
        "lv": (int(coords[1][1]), int(coords[1][0])),
        "myo": (int(coords[2][1]), int(coords[2][0])),
        "background": (int(coords[3][1]), int(coords[3][0]))
    }

    return pixels


def get_default_pixel_coordinates() -> dict:
    """
    Restituisce coordinate pixel di default basate su anatomia tipica.

    Le coordinate sono ottimizzate per immagini 256x256 con il cuore centrato.

    Returns
    -------
    pixels : dict
        Dizionario con coordinate (row, col) per ogni tessuto
    """
    pixels = {
        "rv": (100, 140),     # Ventricolo destro (più a destra)
        "lv": (130, 110),     # Ventricolo sinistro (centro-sinistra)
        "myo": (115, 90),     # Miocardio (parete)
        "background": (50, 50)  # Background (angolo sup-sinistra)
    }

    return pixels


def main() -> int:
    """Funzione principale dello script."""
    parser = argparse.ArgumentParser(
        description="Visualizza curve intensità/tempo per perfusione cardiaca",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Usa coordinate default
    python plot_time_curves.py

    # Selezione interattiva
    python plot_time_curves.py --interactive

    # Specifica coordinate manualmente
    python plot_time_curves.py --rv-pixel 100 140 --lv-pixel 130 110

    # Usa solo primi 50 frame
    python plot_time_curves.py --n-frames 50
        """
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "perfusione",
        help="Directory contenente i file DICOM"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Directory per salvare i risultati"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Seleziona pixel interattivamente"
    )

    parser.add_argument(
        "--rv-pixel",
        nargs=2,
        type=int,
        metavar=("ROW", "COL"),
        help="Coordinate pixel ventricolo destro"
    )

    parser.add_argument(
        "--lv-pixel",
        nargs=2,
        type=int,
        metavar=("ROW", "COL"),
        help="Coordinate pixel ventricolo sinistro"
    )

    parser.add_argument(
        "--myo-pixel",
        nargs=2,
        type=int,
        metavar=("ROW", "COL"),
        help="Coordinate pixel miocardio"
    )

    parser.add_argument(
        "--bg-pixel",
        nargs=2,
        type=int,
        metavar=("ROW", "COL"),
        help="Coordinate pixel background"
    )

    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Numero di frame da caricare (default: tutti)"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Non salvare la figura"
    )

    args = parser.parse_args()

    # Crea directory output se non esiste
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VISUALIZZAZIONE CURVE INTENSITÀ/TEMPO - PERFUSIONE CARDIACA")
    print("=" * 70)

    # ===== Caricamento dati =====
    print(f"\n[1/3] Caricamento serie DICOM da: {args.data_dir}")

    try:
        image_stack, trigger_times = load_perfusion_series(
            args.data_dir,
            n_frames=args.n_frames
        )

        print(f"  ✓ Caricate {image_stack.shape[2]} immagini di dimensione "
              f"{image_stack.shape[0]}x{image_stack.shape[1]}")
        print(f"  ✓ Intervallo temporale: {trigger_times[0]:.2f} - "
              f"{trigger_times[-1]:.2f} secondi")
    except DataLoadError as e:
        logger.error(f"Failed to load DICOM series: {e}")
        print(f"ERRORE: {e}")
        return 1

    # ===== Selezione pixel =====
    print("\n[2/3] Selezione pixel rappresentativi")

    if args.interactive:
        print("  → Modalità interattiva attivata")
        pixels = select_representative_pixels_interactive(image_stack)
    else:
        # Usa coordinate fornite o default
        pixels = get_default_pixel_coordinates()

        if args.rv_pixel:
            pixels["rv"] = tuple(args.rv_pixel)
        if args.lv_pixel:
            pixels["lv"] = tuple(args.lv_pixel)
        if args.myo_pixel:
            pixels["myo"] = tuple(args.myo_pixel)
        if args.bg_pixel:
            pixels["background"] = tuple(args.bg_pixel)

    print("  Coordinate pixel selezionate:")
    for tissue, (row, col) in pixels.items():
        print(f"    - {tissue.upper():12s}: ({row:3d}, {col:3d})")

    # ===== Estrazione curve =====
    print("\n[3/3] Estrazione e visualizzazione curve")

    pixel_coords = [pixels["rv"], pixels["lv"], pixels["myo"], pixels["background"]]
    labels = ["Right Ventricle (RV)", "Left Ventricle (LV)", "Myocardium", "Background"]

    curves = extract_pixel_time_curves(image_stack, pixel_coords)

    print(f"  ✓ Estratte {len(curves)} curve con {curves.shape[1]} punti temporali")

    # Statistiche sulle curve
    print("\n  Statistiche curve:")
    for i, label in enumerate(labels):
        baseline = np.mean(curves[i, :5])
        peak = np.max(curves[i, :])
        peak_time = trigger_times[np.argmax(curves[i, :])]
        enhancement = peak - baseline

        print(f"    {label}:")
        print(f"      Baseline:    {baseline:6.1f}")
        print(f"      Peak:        {peak:6.1f} (t={peak_time:.1f}s)")
        print(f"      Enhancement: {enhancement:6.1f} (+{enhancement/baseline*100:.1f}%)")

    # Visualizzazione
    save_path = None if args.no_save else args.output_dir / "time_curves.png"

    plot_time_curves(
        curves,
        trigger_times,
        labels,
        title="Cardiac Perfusion: Intensity/Time Curves",
        save_path=save_path
    )

    print("\n" + "=" * 70)
    print("COMPLETATO!")
    print("=" * 70)

    if not args.no_save:
        print(f"\nRisultati salvati in: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
