"""
Script principale per segmentazione grasso addominale (SAT/VAT).

Implementa la pipeline completa per quantificare:
- SAT (Subcutaneous Adipose Tissue): grasso sottocutaneo
- VAT (Visceral Adipose Tissue): grasso viscerale intra-addominale
- VAT/SAT ratio: indice di rischio cardiovascolare

Pipeline:
    1. Caricamento volume 3D da DICOM
    2. K-means clustering (K=3): aria, acqua/muscolo, grasso
    3. Rimozione braccia tramite labeling componenti connesse
    4. Active contours doppi per SAT (bordo esterno cute + interno fascia)
    5. EM-GMM su istogramma intra-addominale per VAT
    6. Calcolo volumi e salvataggio risultati

Usage:
    python fat_segmentation.py

Expected output (valori di riferimento):
    SAT: ~2091 cm^3
    VAT: ~970 cm^3
    VAT/SAT: ~46%

Author: Generated with Claude Code
Date: 2025-11-20
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_abdominal_volume,
    process_fat_segmentation_pipeline
)


def main():
    """Esegue pipeline completa segmentazione grasso addominale."""

    parser = argparse.ArgumentParser(
        description="Segmentazione grasso addominale SAT/VAT da MRI"
    )
    parser.add_argument(
        '--dicom_dir',
        type=str,
        default='../data/dicom',
        help='Directory contenente file DICOM (default: ../data/dicom)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results',
        help='Directory output risultati (default: ../results)'
    )
    parser.add_argument(
        '--kmeans_clusters',
        type=int,
        default=3,
        help='Numero cluster K-means (default: 3)'
    )
    parser.add_argument(
        '--outer_iterations',
        type=int,
        default=150,
        help='Iterazioni active contour esterno (default: 150)'
    )
    parser.add_argument(
        '--inner_iterations',
        type=int,
        default=100,
        help='Iterazioni active contour interno (default: 100)'
    )
    parser.add_argument(
        '--gmm_components',
        type=int,
        default=2,
        help='Numero componenti Gaussiane GMM (default: 2)'
    )
    parser.add_argument(
        '--save_masks',
        action='store_true',
        help='Salva maschere 3D come file .npy'
    )

    args = parser.parse_args()

    # Percorsi
    dicom_dir = Path(args.dicom_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("SEGMENTAZIONE GRASSO ADDOMINALE SAT/VAT")
    print("="*60)
    print(f"DICOM directory: {dicom_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parametri:")
    print(f"  - K-means clusters: {args.kmeans_clusters}")
    print(f"  - Outer AC iterations: {args.outer_iterations}")
    print(f"  - Inner AC iterations: {args.inner_iterations}")
    print(f"  - GMM components: {args.gmm_components}")
    print("="*60)

    # STEP 1: Caricamento volume 3D
    print("\n[STEP 1/2] Caricamento volume DICOM...")
    volume, metadata = load_abdominal_volume(dicom_dir, expected_slices=18)

    print(f"  Volume shape: {volume.shape}")
    print(f"  Pixel spacing: {metadata['pixel_spacing']} mm")
    print(f"  Slice thickness: {metadata['slice_thickness']} mm")

    # STEP 2: Pipeline segmentazione
    print("\n[STEP 2/2] Esecuzione pipeline segmentazione...\n")

    results = process_fat_segmentation_pipeline(
        volume,
        metadata,
        kmeans_clusters=args.kmeans_clusters,
        outer_iterations=args.outer_iterations,
        inner_iterations=args.inner_iterations,
        gmm_components=args.gmm_components,
        verbose=True
    )

    # Estrai risultati
    sat_mask_3d = results['sat_mask_3d']
    vat_mask_3d = results['vat_mask_3d']
    volumes = results['volumes']

    # STEP 3: Salvataggio risultati
    print("\n[SAVING] Salvataggio risultati...")

    # Salva volumi in file di testo
    results_file = output_dir / 'fat_volumes.txt'
    with open(results_file, 'w') as f:
        f.write("SEGMENTAZIONE GRASSO ADDOMINALE - RISULTATI\n")
        f.write("="*50 + "\n\n")
        f.write(f"SAT volume: {volumes['sat_volume_cm3']:.2f} cm^3\n")
        f.write(f"VAT volume: {volumes['vat_volume_cm3']:.2f} cm^3\n")
        f.write(f"Total fat: {volumes['total_fat_cm3']:.2f} cm^3\n")
        f.write(f"VAT/SAT ratio: {volumes['vat_sat_ratio_percent']:.2f} %\n\n")
        f.write("VALORI DI RIFERIMENTO:\n")
        f.write("-"*50 + "\n")
        f.write("SAT: 2091 cm^3\n")
        f.write("VAT: 970 cm^3\n")
        f.write("VAT/SAT: 46%\n\n")
        f.write("PARAMETRI PIPELINE:\n")
        f.write("-"*50 + "\n")
        f.write(f"K-means clusters: {args.kmeans_clusters}\n")
        f.write(f"Outer AC iterations: {args.outer_iterations}\n")
        f.write(f"Inner AC iterations: {args.inner_iterations}\n")
        f.write(f"GMM components: {args.gmm_components}\n")

    print(f"  Volumi salvati in: {results_file}")

    # Salva maschere se richiesto
    if args.save_masks:
        np.save(output_dir / 'sat_mask_3d.npy', sat_mask_3d)
        np.save(output_dir / 'vat_mask_3d.npy', vat_mask_3d)
        np.save(output_dir / 'volume.npy', volume)
        print(f"  Maschere 3D salvate in: {output_dir}")

    # STEP 4: Visualizzazione slice centrale
    print("\n[VISUALIZATION] Generazione visualizzazione...")

    mid_slice = volume.shape[0] // 2
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Volume originale, SAT, VAT
    axes[0, 0].imshow(volume[mid_slice], cmap='gray')
    axes[0, 0].set_title(f'Slice {mid_slice}: Immagine Originale')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(volume[mid_slice], cmap='gray')
    axes[0, 1].imshow(sat_mask_3d[mid_slice], cmap='Reds', alpha=0.5)
    axes[0, 1].set_title('SAT (Grasso Sottocutaneo)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(volume[mid_slice], cmap='gray')
    axes[0, 2].imshow(vat_mask_3d[mid_slice], cmap='Blues', alpha=0.5)
    axes[0, 2].set_title('VAT (Grasso Viscerale)')
    axes[0, 2].axis('off')

    # Row 2: Overlay, Contorni, Risultati
    overlay = np.zeros((*volume[mid_slice].shape, 3))
    overlay[..., 0] = sat_mask_3d[mid_slice]  # SAT rosso
    overlay[..., 1] = vat_mask_3d[mid_slice]  # VAT verde

    axes[1, 0].imshow(volume[mid_slice], cmap='gray')
    axes[1, 0].imshow(overlay, alpha=0.5)
    axes[1, 0].set_title('Overlay SAT (rosso) + VAT (verde)')
    axes[1, 0].axis('off')

    # Contorni active contours
    axes[1, 1].imshow(volume[mid_slice], cmap='gray')
    axes[1, 1].contour(results['outer_contour_3d'][mid_slice], colors='red', linewidths=2)
    axes[1, 1].contour(results['inner_contour_3d'][mid_slice], colors='blue', linewidths=2)
    axes[1, 1].set_title('Contorni: Esterno (rosso), Interno (blu)')
    axes[1, 1].axis('off')

    # Grafico risultati
    axes[1, 2].axis('off')
    result_text = (
        f"RISULTATI\n\n"
        f"SAT: {volumes['sat_volume_cm3']:.1f} cm³\n"
        f"VAT: {volumes['vat_volume_cm3']:.1f} cm³\n"
        f"Total: {volumes['total_fat_cm3']:.1f} cm³\n\n"
        f"VAT/SAT: {volumes['vat_sat_ratio_percent']:.1f}%\n\n"
        f"RIFERIMENTI:\n"
        f"SAT: 2091 cm³\n"
        f"VAT: 970 cm³\n"
        f"VAT/SAT: 46%"
    )
    axes[1, 2].text(0.1, 0.5, result_text, fontsize=12, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()
    fig_path = output_dir / 'fat_segmentation_results.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Figure salvata in: {fig_path}")

    # Mostra figura
    # plt.show()

    print("\n" + "="*60)
    print("COMPLETATO!")
    print("="*60)


if __name__ == '__main__':
    main()
