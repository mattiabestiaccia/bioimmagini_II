"""
Validazione registrazione con N runs e Bland-Altman plots.

Esegue N registrazioni con disallineamenti random diversi
e analizza errori con Bland-Altman.

Usage:
    python validate_registration.py --n_runs 20

Author: Generated with Claude Code
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_minc_slice,
    pad_to_square,
    random_rigid_transform_2d,
    apply_rigid_transform_2d,
    compute_mutual_information,
    register_with_differential_evolution,
    bland_altman_stats
)


def bland_altman_plot(true_vals, est_vals, param_name, ax):
    """Crea Bland-Altman plot."""
    diff = est_vals - true_vals
    mean_vals = (true_vals + est_vals) / 2

    stats = bland_altman_stats(true_vals, est_vals)

    ax.scatter(mean_vals, diff, alpha=0.6, s=50)
    ax.axhline(stats['mean_diff'], color='r', linestyle='--', label=f'Mean: {stats["mean_diff"]:.2f}')
    ax.axhline(stats['loa_lower'], color='g', linestyle=':', label=f'LoA: ±{1.96*stats["std_diff"]:.2f}')
    ax.axhline(stats['loa_upper'], color='g', linestyle=':')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)

    ax.set_xlabel(f'Mean {param_name}')
    ax.set_ylabel(f'Diff (Estimated - True)')
    ax.set_title(f'Bland-Altman: {param_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs', type=int, default=20, help='Numero runs validazione')
    parser.add_argument('--maxiter', type=int, default=50, help='Max iterazioni DE')
    args = parser.parse_args()

    output_dir = Path('../results')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"VALIDAZIONE REGISTRAZIONE ({args.n_runs} runs)")
    print("="*70)

    # Carica immagini
    data_dir = Path('../data/minc')
    t1 = pad_to_square(load_minc_slice(data_dir / 't1_icbm_normal_1mm_pn3_rf0.mnc', 62))
    pd = pad_to_square(load_minc_slice(data_dir / 'pd_icbm_normal_1mm_pn3_rf0.mnc', 62))

    mi_start = compute_mutual_information(t1, pd)

    # Arrays risultati
    tx_true, ty_true, angle_true = [], [], []
    tx_est, ty_est, angle_est = [], [], []
    mi_mis, mi_end = [], []

    # Runs
    for i in range(args.n_runs):
        print(f"\n[Run {i+1}/{args.n_runs}]")

        # Disallineamento random
        tx_s, ty_s, ang_s = random_rigid_transform_2d(t1.shape)
        pd_mis = apply_rigid_transform_2d(pd, tx_s, ty_s, ang_s, order=0)
        mi_m = compute_mutual_information(t1, pd_mis)

        # Registrazione
        res = register_with_differential_evolution(t1, pd_mis, maxiter=args.maxiter, popsize=10, verbose=False)

        # Salva
        tx_true.append(-tx_s)
        ty_true.append(-ty_s)
        angle_true.append(-ang_s)

        tx_est.append(res['tx'])
        ty_est.append(res['ty'])
        angle_est.append(res['angle'])

        mi_mis.append(mi_m)
        mi_end.append(res['mi_final'])

        print(f"  True: tx={-tx_s:.1f}, ty={-ty_s:.1f}, θ={-ang_s:.1f}")
        print(f"  Est:  tx={res['tx']:.1f}, ty={res['ty']:.1f}, θ={res['angle']:.1f}")
        print(f"  MI: {mi_m:.3f} → {res['mi_final']:.3f}")

    # Conversione array
    tx_true, ty_true, angle_true = np.array(tx_true), np.array(ty_true), np.array(angle_true)
    tx_est, ty_est, angle_est = np.array(tx_est), np.array(ty_est), np.array(angle_est)
    mi_mis, mi_end = np.array(mi_mis), np.array(mi_end)

    # Bland-Altman plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    bland_altman_plot(tx_true, tx_est, 'TX (pixel)', axes[0, 0])
    bland_altman_plot(ty_true, ty_est, 'TY (pixel)', axes[0, 1])
    bland_altman_plot(angle_true, angle_est, 'Angle (deg)', axes[1, 0])

    # MI recovery
    mi_diff = mi_end - mi_start
    axes[1, 1].hist(mi_diff, bins=15, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(0, color='r', linestyle='--', label='Perfect recovery')
    axes[1, 1].set_xlabel('MI_end - MI_start')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'MI Recovery (mean={np.mean(mi_diff):.4f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / f'bland_altman_{args.n_runs}runs.png'
    plt.savefig(fig_path, dpi=150)
    print(f"\nBland-Altman salvato: {fig_path}")

    # Statistiche
    stats_file = output_dir / f'validation_stats_{args.n_runs}runs.txt'
    with open(stats_file, 'w') as f:
        f.write(f"VALIDAZIONE REGISTRAZIONE ({args.n_runs} runs)\n")
        f.write("="*50 + "\n\n")

        for param_name, true_vals, est_vals in [('TX', tx_true, tx_est),
                                                  ('TY', ty_true, ty_est),
                                                  ('Angle', angle_true, angle_est)]:
            stats = bland_altman_stats(true_vals, est_vals)
            f.write(f"{param_name}:\n")
            f.write(f"  Bias: {stats['bias']:.3f}\n")
            f.write(f"  Precision (SD): {stats['precision']:.3f}\n")
            f.write(f"  LoA: [{stats['loa_lower']:.3f}, {stats['loa_upper']:.3f}]\n\n")

        f.write(f"MI Recovery:\n")
        f.write(f"  Mean (MI_end - MI_start): {np.mean(mi_diff):.4f}\n")
        f.write(f"  SD: {np.std(mi_diff):.4f}\n")

    print(f"Statistiche salvate: {stats_file}")
    print("\nCOMPLETATO!")


if __name__ == '__main__':
    main()
