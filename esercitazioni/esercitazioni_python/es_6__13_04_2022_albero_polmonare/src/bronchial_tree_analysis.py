"""
Script principale per analisi albero bronchiale da CT toracica.

Implementa la pipeline completa per:
- Segmentazione albero bronchiale (trachea + bronchi primari)
- Estrazione centerline tramite skeletonization 3D
- Misurazione diametro lume lungo albero con sphere method

Pipeline:
    1. Caricamento volume CT 3D (148 slice) con Hounsfield Units
    2. Interpolazione isotropa (voxel cubici)
    3. Region growing 3D da seed in trachea
    4. Filtraggio maschera (riempimento buchi)
    5. Skeletonization 3D per centerline
    6. Identificazione endpoints
    7. Estrazione percorso da endpoint a trachea
    8. Sphere method per diametro lungo centerline
    9. Grafici e visualizzazioni

Valori attesi:
    - Diametro trachea: 15-18 mm
    - Diametro bronchi primari: 10-12 mm

Usage:
    python bronchial_tree_analysis.py --seed_z 10 --seed_y 250 --seed_x 250

Author: Generated with Claude Code
Date: 2025-11-20
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_ct_volume_hounsfield,
    check_isotropic,
    interpolate_to_isotropic,
    region_growing_3d,
    fill_holes_3d,
    skeletonize_3d_clean,
    find_skeleton_endpoints,
    extract_centerline_path,
    measure_diameter_along_path,
    smooth_diameters
)


def main():
    """Esegue pipeline completa analisi albero bronchiale."""

    parser = argparse.ArgumentParser(
        description="Analisi albero bronchiale da CT toracica"
    )
    parser.add_argument(
        '--dicom_dir',
        type=str,
        default='../data/dicom/3000522.000000-04919',
        help='Directory contenente file DICOM (default: ../data/dicom/3000522.000000-04919)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results',
        help='Directory output risultati (default: ../results)'
    )
    parser.add_argument(
        '--seed_z',
        type=int,
        default=10,
        help='Coordinata Z seed nella trachea (default: 10)'
    )
    parser.add_argument(
        '--seed_y',
        type=int,
        default=250,
        help='Coordinata Y seed nella trachea (default: 250)'
    )
    parser.add_argument(
        '--seed_x',
        type=int,
        default=250,
        help='Coordinata X seed nella trachea (default: 250)'
    )
    parser.add_argument(
        '--rg_tolerance',
        type=float,
        default=100.0,
        help='Tolleranza region growing in HU (default: 100.0)'
    )
    parser.add_argument(
        '--skip_interpolation',
        action='store_true',
        help='Salta interpolazione isotropa (usa volume originale)'
    )
    parser.add_argument(
        '--load_mask',
        type=str,
        default=None,
        help='Carica maschera pre-segmentata (path .npy) invece di rieseguire region growing'
    )
    parser.add_argument(
        '--save_mask',
        action='store_true',
        help='Salva maschera segmentata come .npy'
    )

    args = parser.parse_args()

    # Percorsi
    dicom_dir = Path(args.dicom_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("ANALISI ALBERO BRONCHIALE DA CT TORACICA")
    print("="*70)
    print(f"DICOM directory: {dicom_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Region growing seed: ({args.seed_z}, {args.seed_y}, {args.seed_x})")
    print(f"RG tolerance: {args.rg_tolerance} HU")
    print("="*70)

    # STEP 1: Caricamento volume CT
    print("\n[STEP 1/8] Caricamento volume CT con Hounsfield Units...")
    volume, metadata = load_ct_volume_hounsfield(dicom_dir, expected_slices=148)

    print(f"  Volume shape: {volume.shape}")
    print(f"  Pixel spacing: {metadata['pixel_spacing']} mm")
    print(f"  Slice thickness: {metadata['slice_thickness']} mm")
    print(f"  HU range: [{volume.min():.1f}, {volume.max():.1f}]")
    print(f"  RescaleSlope: {metadata['rescale_slope']}, RescaleIntercept: {metadata['rescale_intercept']}")

    # STEP 2: Verifica isotropia e interpolazione
    print("\n[STEP 2/8] Verifica isotropia...")
    is_iso = check_isotropic(metadata)
    print(f"  Volume isotropo: {is_iso}")

    if not is_iso and not args.skip_interpolation:
        print("\n  Interpolazione a voxel cubici...")
        volume, metadata = interpolate_to_isotropic(volume, metadata)
        print(f"  Nuovo shape: {volume.shape}")
    elif args.skip_interpolation:
        print("  Interpolazione saltata (--skip_interpolation)")

    spacing = metadata['pixel_spacing'][0]

    # Aggiusta seed se interpolato
    if not is_iso and not args.skip_interpolation:
        # Ricalcola seed per volume interpolato (approssimazione)
        seed_z = args.seed_z
        seed_y = int(args.seed_y * metadata['rows'] / volume.shape[1])
        seed_x = int(args.seed_x * metadata['cols'] / volume.shape[2])
    else:
        seed_z = args.seed_z
        seed_y = args.seed_y
        seed_x = args.seed_x

    seed = (seed_z, seed_y, seed_x)
    print(f"\n  Seed adjusted: {seed}")
    print(f"  HU at seed: {volume[seed]:.1f}")

    # STEP 3: Region growing o caricamento maschera
    if args.load_mask:
        print(f"\n[STEP 3/8] Caricamento maschera pre-segmentata da {args.load_mask}...")
        mask = np.load(args.load_mask)
        print(f"  Maschera caricata: {mask.shape}, {np.sum(mask)} voxel")
    else:
        print("\n[STEP 3/8] Region growing 3D...")
        print(f"  Seed: {seed}, Tolerance: {args.rg_tolerance} HU")
        mask = region_growing_3d(
            volume,
            seed=seed,
            tolerance=args.rg_tolerance,
            connectivity=26,
            verbose=True
        )
        print(f"  Voxel segmentati: {np.sum(mask)}")

        if args.save_mask:
            mask_path = output_dir / 'bronchial_mask.npy'
            np.save(mask_path, mask)
            print(f"  Maschera salvata in: {mask_path}")

    # STEP 4: Filtraggio maschera
    print("\n[STEP 4/8] Filtraggio maschera (riempimento buchi)...")
    mask_filled = fill_holes_3d(mask, method='label')
    n_holes_filled = np.sum(mask_filled) - np.sum(mask)
    print(f"  Buchi riempiti: {n_holes_filled} voxel")

    # STEP 5: Skeletonization
    print("\n[STEP 5/8] Skeletonization 3D...")
    skeleton = skeletonize_3d_clean(
        mask_filled,
        min_branch_length=10,
        verbose=True
    )

    # STEP 6: Identificazione endpoints
    print("\n[STEP 6/8] Identificazione endpoints...")
    endpoints = find_skeleton_endpoints(skeleton)
    print(f"  Endpoints trovati: {len(endpoints)}")
    for i, ep in enumerate(endpoints[:5]):
        print(f"    Endpoint {i}: {ep}")

    # STEP 7: Estrazione centerline path
    print("\n[STEP 7/8] Estrazione centerline path...")

    # Seleziona endpoint con z massima (ramo terminale piu' lontano da trachea)
    if len(endpoints) == 0:
        print("  ERRORE: Nessun endpoint trovato!")
        return

    endpoint_max_z = endpoints[np.argmax(endpoints[:, 0])]
    print(f"  Endpoint selezionato (z massima): {endpoint_max_z}")

    centerline_path = extract_centerline_path(
        skeleton,
        tuple(endpoint_max_z),
        direction='descending_z'
    )
    print(f"  Centerline path: {len(centerline_path)} punti")

    # STEP 8: Misurazione diametro
    print("\n[STEP 8/8] Misurazione diametro con sphere method...")
    distances_mm, diameters_mm = measure_diameter_along_path(
        mask_filled,
        centerline_path,
        spacing=spacing,
        verbose=True
    )

    # Smooth diameters
    diameters_smooth = smooth_diameters(diameters_mm, window_size=5)

    # Statistiche
    print("\n=== RISULTATI ===")
    print(f"Lunghezza centerline: {distances_mm[-1]:.1f} mm")
    print(f"Diametro massimo: {diameters_mm.max():.2f} mm")
    print(f"Diametro minimo: {diameters_mm.min():.2f} mm")
    print(f"Diametro medio: {diameters_mm.mean():.2f} mm")

    # Stima diametro trachea (primi punti)
    trachea_diam = np.median(diameters_mm[:min(20, len(diameters_mm) // 4)])
    print(f"Diametro trachea stimato: {trachea_diam:.2f} mm")

    # STEP 9: Salvataggio risultati
    print("\n[SAVING] Salvataggio risultati...")

    # Salva dati numerici
    results_file = output_dir / 'diameter_measurements.txt'
    with open(results_file, 'w') as f:
        f.write("ANALISI ALBERO BRONCHIALE - RISULTATI\n")
        f.write("="*50 + "\n\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Region growing tolerance: {args.rg_tolerance} HU\n")
        f.write(f"Voxel spacing: {spacing:.3f} mm\n\n")
        f.write(f"Voxel segmentati: {np.sum(mask)}\n")
        f.write(f"Endpoints: {len(endpoints)}\n")
        f.write(f"Centerline punti: {len(centerline_path)}\n\n")
        f.write(f"Lunghezza centerline: {distances_mm[-1]:.1f} mm\n")
        f.write(f"Diametro trachea stimato: {trachea_diam:.2f} mm\n")
        f.write(f"Diametro massimo: {diameters_mm.max():.2f} mm\n")
        f.write(f"Diametro minimo: {diameters_mm.min():.2f} mm\n\n")
        f.write("VALORI ATTESI:\n")
        f.write("-"*50 + "\n")
        f.write("Diametro trachea: 15-18 mm\n")
        f.write("Diametro bronchi primari: 10-12 mm\n\n")
        f.write("DATI:\n")
        f.write("-"*50 + "\n")
        f.write("distance_mm\tdiameter_raw\tdiameter_smooth\n")
        for dist, diam_raw, diam_smooth in zip(distances_mm, diameters_mm, diameters_smooth):
            f.write(f"{dist:.2f}\t{diam_raw:.2f}\t{diam_smooth:.2f}\n")

    print(f"  Risultati salvati in: {results_file}")

    # Salva maschere e skeleton
    if args.save_mask:
        np.save(output_dir / 'mask_filled.npy', mask_filled)
        np.save(output_dir / 'skeleton.npy', skeleton)
        print(f"  Maschere salvate in: {output_dir}")

    # STEP 10: Visualizzazione
    print("\n[VISUALIZATION] Generazione grafici...")

    # Grafico diametro vs distanza
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(distances_mm, diameters_mm, 'o-', color='lightblue', markersize=3,
            linewidth=1, alpha=0.7, label='RAW')
    ax.plot(distances_mm, diameters_smooth, '-', color='red', linewidth=2,
            label='SMOOTHED')

    ax.set_xlabel('Distance (mm)', fontsize=12)
    ax.set_ylabel('Diameter (mm)', fontsize=12)
    ax.set_title('Bronchial Tree Diameter Along Centerline', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Linee riferimento
    ax.axhline(y=15, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Trachea min (15mm)')
    ax.axhline(y=18, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Trachea max (18mm)')
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Bronchi min (10mm)')
    ax.axhline(y=12, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Bronchi max (12mm)')

    plt.tight_layout()
    fig_path = output_dir / 'diameter_plot.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Grafico salvato in: {fig_path}")

    plt.show()

    print("\n" + "="*70)
    print("COMPLETATO!")
    print("="*70)


if __name__ == '__main__':
    main()
