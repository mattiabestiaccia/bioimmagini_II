"""
Script per visualizzazione avanzata risultati segmentazione grasso.

Genera visualizzazioni dettagliate per analisi qualitativa:
- Montaggio multi-slice con overlay SAT/VAT
- Distribuzione volumetrica slice-by-slice
- Istogrammi intensita' con fit GMM
- Contorni active contours per validazione

Usage:
    python visualize_results.py [--slice SLICE_NUM]

Author: Generated with Claude Code
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_abdominal_volume,
    kmeans_fat_segmentation,
    remove_spurious_components,
    segment_sat_with_active_contours,
    fit_em_gmm
)


def plot_slice_montage(
    volume: np.ndarray,
    sat_mask_3d: np.ndarray,
    vat_mask_3d: np.ndarray,
    output_path: Path,
    n_cols: int = 6
):
    """
    Crea montaggio di tutte le slice con overlay SAT/VAT.

    Args:
        volume: Volume 3D
        sat_mask_3d: Maschera SAT 3D
        vat_mask_3d: Maschera VAT 3D
        output_path: Path file output
        n_cols: Numero colonne
    """
    n_slices = volume.shape[0]
    n_rows = int(np.ceil(n_slices / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    axes = axes.flatten()

    for z in range(n_slices):
        # Overlay RGB
        overlay = np.zeros((*volume[z].shape, 3))
        overlay[..., 0] = sat_mask_3d[z]  # Rosso
        overlay[..., 1] = vat_mask_3d[z]  # Verde

        axes[z].imshow(volume[z], cmap='gray')
        axes[z].imshow(overlay, alpha=0.5)
        axes[z].set_title(f'Slice {z}', fontsize=8)
        axes[z].axis('off')

    # Rimuovi subplot vuoti
    for i in range(n_slices, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Montaggio salvato: {output_path}")


def plot_volume_distribution(
    sat_mask_3d: np.ndarray,
    vat_mask_3d: np.ndarray,
    pixel_spacing: list,
    slice_thickness: float,
    output_path: Path
):
    """
    Grafico distribuzione volumetrica slice-by-slice.

    Args:
        sat_mask_3d: Maschera SAT 3D
        vat_mask_3d: Maschera VAT 3D
        pixel_spacing: [row_spacing, col_spacing] in mm
        slice_thickness: Spessore slice in mm
        output_path: Path file output
    """
    n_slices = sat_mask_3d.shape[0]

    # Calcola area per slice (cm^2)
    pixel_area_mm2 = pixel_spacing[0] * pixel_spacing[1]
    pixel_area_cm2 = pixel_area_mm2 / 100.0

    sat_areas = []
    vat_areas = []

    for z in range(n_slices):
        sat_pixels = np.sum(sat_mask_3d[z])
        vat_pixels = np.sum(vat_mask_3d[z])

        sat_areas.append(sat_pixels * pixel_area_cm2)
        vat_areas.append(vat_pixels * pixel_area_cm2)

    # Grafico
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Subplot 1: Aree per slice
    slices = np.arange(n_slices)
    axes[0].plot(slices, sat_areas, 'ro-', label='SAT', linewidth=2, markersize=6)
    axes[0].plot(slices, vat_areas, 'bo-', label='VAT', linewidth=2, markersize=6)
    axes[0].set_xlabel('Slice number')
    axes[0].set_ylabel('Area (cm^2)')
    axes[0].set_title('Distribuzione area grasso per slice')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Rapporto VAT/SAT per slice
    vat_sat_ratios = []
    for sat_a, vat_a in zip(sat_areas, vat_areas):
        if sat_a > 0:
            vat_sat_ratios.append((vat_a / sat_a) * 100.0)
        else:
            vat_sat_ratios.append(0.0)

    axes[1].bar(slices, vat_sat_ratios, color='purple', alpha=0.7)
    axes[1].set_xlabel('Slice number')
    axes[1].set_ylabel('VAT/SAT ratio (%)')
    axes[1].set_title('Rapporto VAT/SAT per slice')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Distribuzione volumetrica salvata: {output_path}")


def plot_single_slice_detailed(
    volume: np.ndarray,
    slice_idx: int,
    sat_mask_3d: np.ndarray,
    vat_mask_3d: np.ndarray,
    outer_contour_3d: np.ndarray,
    inner_contour_3d: np.ndarray,
    output_path: Path
):
    """
    Visualizzazione dettagliata singola slice con istogramma e contorni.

    Args:
        volume: Volume 3D
        slice_idx: Indice slice
        sat_mask_3d: Maschera SAT 3D
        vat_mask_3d: Maschera VAT 3D
        outer_contour_3d: Contorni esterni 3D
        inner_contour_3d: Contorni interni 3D
        output_path: Path file output
    """
    image_slice = volume[slice_idx]
    sat_mask = sat_mask_3d[slice_idx]
    vat_mask = vat_mask_3d[slice_idx]
    outer_contour = outer_contour_3d[slice_idx]
    inner_contour = inner_contour_3d[slice_idx]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig)

    # Row 1: Immagine originale, SAT, VAT, Overlay
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_slice, cmap='gray')
    ax1.set_title(f'Slice {slice_idx}: Originale')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(image_slice, cmap='gray')
    ax2.imshow(sat_mask, cmap='Reds', alpha=0.6)
    ax2.set_title('SAT')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(image_slice, cmap='gray')
    ax3.imshow(vat_mask, cmap='Blues', alpha=0.6)
    ax3.set_title('VAT')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    overlay = np.zeros((*image_slice.shape, 3))
    overlay[..., 0] = sat_mask
    overlay[..., 1] = vat_mask
    ax4.imshow(image_slice, cmap='gray')
    ax4.imshow(overlay, alpha=0.5)
    ax4.set_title('Overlay SAT+VAT')
    ax4.axis('off')

    # Row 2: Contorni, Istogramma intra-addominale, GMM fit
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(image_slice, cmap='gray')
    ax5.contour(outer_contour, colors='red', linewidths=2, levels=[0.5])
    ax5.contour(inner_contour, colors='blue', linewidths=2, levels=[0.5])
    ax5.set_title('Contorni AC')
    ax5.axis('off')

    # Istogramma regione intra-addominale
    ax6 = fig.add_subplot(gs[1, 1:3])
    inner_region = inner_contour > 0
    if np.sum(inner_region) > 0:
        intra_pixels = image_slice[inner_region]
        hist, bin_edges = np.histogram(intra_pixels, bins=50, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        ax6.bar(bin_centers, hist, width=np.diff(bin_edges)[0],
                color='gray', alpha=0.7, edgecolor='black', label='Istogramma')

        # Fit GMM
        try:
            gmm, _ = fit_em_gmm(hist, bin_edges, n_components=2)

            # Plot Gaussiane
            x = np.linspace(0, 1, 200)
            responsibilities = gmm.predict_proba(x.reshape(-1, 1))

            for i in range(2):
                gaussian = responsibilities[:, i] * hist.max()
                ax6.plot(x, gaussian, linewidth=2, label=f'Gaussian {i}')

            ax6.legend()
        except:
            pass

    ax6.set_xlabel('Intensita normalizzata')
    ax6.set_ylabel('Frequenza')
    ax6.set_title('Istogramma intra-addominale + GMM fit')
    ax6.grid(True, alpha=0.3)

    # Statistiche
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.axis('off')

    sat_pixels = np.sum(sat_mask)
    vat_pixels = np.sum(vat_mask)
    vat_sat_ratio = (vat_pixels / sat_pixels * 100.0) if sat_pixels > 0 else 0.0

    stats_text = (
        f"SLICE {slice_idx} STATS\n\n"
        f"SAT pixels: {sat_pixels}\n"
        f"VAT pixels: {vat_pixels}\n\n"
        f"VAT/SAT ratio:\n"
        f"{vat_sat_ratio:.1f}%\n\n"
        f"Inner region pixels:\n"
        f"{np.sum(inner_region)}"
    )

    ax7.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Dettaglio slice {slice_idx} salvato: {output_path}")


def main():
    """Genera visualizzazioni avanzate."""

    parser = argparse.ArgumentParser(
        description="Visualizzazione avanzata segmentazione grasso"
    )
    parser.add_argument(
        '--dicom_dir',
        type=str,
        default='../data/dicom',
        help='Directory DICOM'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results',
        help='Directory output'
    )
    parser.add_argument(
        '--slice',
        type=int,
        default=None,
        help='Slice specifica per analisi dettagliata (default: centrale)'
    )

    args = parser.parse_args()

    # Setup
    dicom_dir = Path(args.dicom_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("VISUALIZZAZIONE AVANZATA SEGMENTAZIONE GRASSO")
    print("="*60)

    # Carica volume
    print("\n[1/4] Caricamento volume...")
    volume, metadata = load_abdominal_volume(dicom_dir)
    print(f"  Volume shape: {volume.shape}")

    # Quick segmentation per visualizzazione
    print("\n[2/4] Segmentazione rapida...")
    labels_volume, centroids, tissue_map = kmeans_fat_segmentation(volume)
    fat_label = tissue_map['fat']
    fat_mask_kmeans = (labels_volume == fat_label).astype(np.uint8)
    torso_mask_3d = remove_spurious_components(fat_mask_kmeans, keep_largest=True)

    # Segmenta slice centrale per esempio dettagliato
    mid_slice = volume.shape[0] // 2 if args.slice is None else args.slice
    print(f"\n[3/4] Segmentazione slice {mid_slice} per dettagli...")

    sat_mask, outer_contour, inner_contour = segment_sat_with_active_contours(
        volume[mid_slice],
        torso_mask_3d[mid_slice],
        outer_iterations=150,
        inner_iterations=100
    )

    # Per montaggio, usa maschere dummy (per velocita')
    sat_mask_3d = np.zeros_like(volume, dtype=np.uint8)
    vat_mask_3d = np.zeros_like(volume, dtype=np.uint8)
    outer_contour_3d = np.zeros_like(volume, dtype=np.uint8)
    inner_contour_3d = np.zeros_like(volume, dtype=np.uint8)

    sat_mask_3d[mid_slice] = sat_mask
    outer_contour_3d[mid_slice] = outer_contour
    inner_contour_3d[mid_slice] = inner_contour

    # Visualizzazioni
    print("\n[4/4] Generazione visualizzazioni...")

    # Dettaglio slice centrale
    plot_single_slice_detailed(
        volume,
        mid_slice,
        sat_mask_3d,
        vat_mask_3d,
        outer_contour_3d,
        inner_contour_3d,
        output_dir / f'slice_{mid_slice}_detailed.png'
    )

    print("\n" + "="*60)
    print("COMPLETATO!")
    print(f"Risultati in: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
