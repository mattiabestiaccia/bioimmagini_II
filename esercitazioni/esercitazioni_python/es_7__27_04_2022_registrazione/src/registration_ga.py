"""
Script registrazione immagini MRI con Differential Evolution (GA-like).

Pipeline:
    1. Carica T1/PD da BrainWeb MINC
    2. Estrai slice + padding
    3. Disallineamento random
    4. Registrazione con DE optimizer + MI
    5. Validazione parametri

Usage:
    python registration_ga.py

Author: Generated with Claude Code
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_minc_slice,
    pad_to_square,
    random_rigid_transform_2d,
    apply_rigid_transform_2d,
    compute_mutual_information,
    register_with_differential_evolution
)


def main():
    # Percorsi
    data_dir = Path('../data/minc')
    output_dir = Path('../results')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("REGISTRAZIONE IMMAGINI MRI CON DIFFERENTIAL EVOLUTION")
    print("="*70)

    # Carica immagini
    print("\n[1/6] Caricamento immagini BrainWeb...")
    t1_path = data_dir / 't1_icbm_normal_1mm_pn3_rf0.mnc'
    pd_path = data_dir / 'pd_icbm_normal_1mm_pn3_rf0.mnc'

    t1_slice = load_minc_slice(t1_path, slice_idx=62)
    pd_slice = load_minc_slice(pd_path, slice_idx=62)

    print(f"  T1 shape: {t1_slice.shape}, range: [{t1_slice.min():.1f}, {t1_slice.max():.1f}]")
    print(f"  PD shape: {pd_slice.shape}, range: [{pd_slice.min():.1f}, {pd_slice.max():.1f}]")

    # Padding
    print("\n[2/6] Zero padding a immagini quadrate...")
    t1_padded = pad_to_square(t1_slice)
    pd_padded = pad_to_square(pd_slice)
    print(f"  Shape dopo padding: {t1_padded.shape}")

    # MI iniziale
    mi_start = compute_mutual_information(t1_padded, pd_padded)
    print(f"  MI iniziale (aligned): {mi_start:.4f}")

    # Disallineamento random
    print("\n[3/6] Disallineamento random...")
    tx_sim, ty_sim, angle_sim = random_rigid_transform_2d(
        t1_padded.shape,
        max_translation_fraction=0.1,
        max_rotation_deg=60.0
    )
    print(f"  Parametri simulati: tx={tx_sim:.2f}, ty={ty_sim:.2f}, angle={angle_sim:.2f}°")

    pd_misaligned = apply_rigid_transform_2d(pd_padded, tx_sim, ty_sim, angle_sim, order=0)
    mi_misaligned = compute_mutual_information(t1_padded, pd_misaligned)
    print(f"  MI dopo disallineamento: {mi_misaligned:.4f}")

    # Registrazione
    print("\n[4/6] Registrazione con Differential Evolution...")
    results = register_with_differential_evolution(
        t1_padded,
        pd_misaligned,
        maxiter=100,
        popsize=15,
        verbose=False
    )

    tx_reg, ty_reg, angle_reg = results['tx'], results['ty'], results['angle']
    print(f"  Parametri registrati: tx={tx_reg:.2f}, ty={ty_reg:.2f}, angle={angle_reg:.2f}°")
    print(f"  MI finale: {results['mi_final']:.4f}")
    print(f"  Iterazioni: {results['nit']}, Eval: {results['nfev']}")

    # Errori
    print("\n[5/6] Analisi errori...")
    error_tx = tx_reg - (-tx_sim)
    error_ty = ty_reg - (-ty_sim)
    error_angle = angle_reg - (-angle_sim)

    print(f"  Errore tx: {error_tx:.2f} pixel")
    print(f"  Errore ty: {error_ty:.2f} pixel")
    print(f"  Errore angle: {error_angle:.2f}°")

    # Salvataggio
    print("\n[6/6] Salvataggio risultati...")
    results_file = output_dir / 'registration_results.txt'
    with open(results_file, 'w') as f:
        f.write("REGISTRAZIONE IMMAGINI - RISULTATI\n")
        f.write("="*50 + "\n\n")
        f.write(f"MI start: {mi_start:.4f}\n")
        f.write(f"MI misaligned: {mi_misaligned:.4f}\n")
        f.write(f"MI final: {results['mi_final']:.4f}\n\n")
        f.write("PARAMETRI SIMULATI:\n")
        f.write(f"  tx: {tx_sim:.2f}\n")
        f.write(f"  ty: {ty_sim:.2f}\n")
        f.write(f"  angle: {angle_sim:.2f}\n\n")
        f.write("PARAMETRI REGISTRATI:\n")
        f.write(f"  tx: {tx_reg:.2f}\n")
        f.write(f"  ty: {ty_reg:.2f}\n")
        f.write(f"  angle: {angle_reg:.2f}\n\n")
        f.write("ERRORI:\n")
        f.write(f"  tx: {error_tx:.2f}\n")
        f.write(f"  ty: {error_ty:.2f}\n")
        f.write(f"  angle: {error_angle:.2f}\n")

    # Visualizzazione
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(t1_padded, cmap='gray')
    axes[0].set_title(f'T1 (Fixed)\nMI={mi_start:.4f}')
    axes[0].axis('off')

    axes[1].imshow(pd_misaligned, cmap='gray')
    axes[1].set_title(f'PD Misaligned\nMI={mi_misaligned:.4f}\ntx={tx_sim:.1f}, ty={ty_sim:.1f}, θ={angle_sim:.1f}°')
    axes[1].axis('off')

    axes[2].imshow(results['moving_registered'], cmap='gray')
    axes[2].set_title(f'PD Registered\nMI={results["mi_final"]:.4f}\ntx={tx_reg:.1f}, ty={ty_reg:.1f}, θ={angle_reg:.1f}°')
    axes[2].axis('off')

    plt.tight_layout()
    fig_path = output_dir / 'registration_result.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Figure salvata: {fig_path}")

    plt.show()

    print("\n" + "="*70)
    print("COMPLETATO!")
    print("="*70)


if __name__ == '__main__':
    main()
