#!/usr/bin/env python3
"""
t2star_mapping.py - T2* Parametric Mapping Pipeline

Complete pipeline for T2* relaxometry and iron overload quantification.

Esercitazione 9: Mappe Parametriche T2*
Dataset: 2 patients (PAZIENTE1: iron overload, PAZIENTE2: normal)

Author: Biomedical Imaging Course
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
import argparse
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_multiecho_series,
    create_t2star_map,
    compute_roi_statistics,
    normalize_rmse_map,
    estimate_iron_concentration
)


def plot_multiecho_images(volume: np.ndarray, echo_times: np.ndarray, output_dir: Path):
    """Plot multi-echo images."""
    n_echoes = volume.shape[0]
    ncols = 5
    nrows = int(np.ceil(n_echoes / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3*nrows))
    axes = axes.flatten()

    for i in range(n_echoes):
        axes[i].imshow(volume[i], cmap='gray')
        axes[i].set_title(f'TE = {echo_times[i]:.1f} ms')
        axes[i].axis('off')

    for i in range(n_echoes, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "multiecho_images.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-echo images to {output_dir / 'multiecho_images.png'}")


def plot_t2star_maps(t2star_map: np.ndarray, rmse_map: np.ndarray,
                     model_name: str, output_dir: Path, vmax_t2: float = 50):
    """Plot T2* and RMSE maps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # T2* map
    im1 = axes[0].imshow(t2star_map, cmap='jet', vmin=0, vmax=vmax_t2)
    axes[0].set_title(f'T2* Map ({model_name})', fontsize=14, weight='bold')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('T2* (ms)', fontsize=12)

    # RMSE map
    im2 = axes[0].imshow(rmse_map, cmap='hot', vmin=0)
    axes[1].set_title(f'Fitting Error ({model_name})', fontsize=14, weight='bold')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('RMSE', fontsize=12)

    plt.tight_layout()
    filename = f"t2star_map_{model_name.lower().replace('-', '_')}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved T2* maps to {output_dir / filename}")


def plot_decay_curve(volume: np.ndarray, echo_times: np.ndarray,
                     y: int, x: int, params_s: np.ndarray, params_c: np.ndarray,
                     output_dir: Path):
    """Plot signal decay curve with fitted models."""
    from utils import model_s_exp, model_c_exp

    signal = volume[:, y, x]
    te_fine = np.linspace(echo_times[0], echo_times[-1], 100)

    plt.figure(figsize=(10, 6))
    plt.plot(echo_times, signal, 'ko', markersize=8, label='Measured signal')

    # S-EXP fit
    fitted_s = model_s_exp(te_fine, *params_s)
    plt.plot(te_fine, fitted_s, 'b-', linewidth=2, label=f'S-EXP (T2*={1/params_s[1]:.1f} ms)')

    # C-EXP fit
    fitted_c = model_c_exp(te_fine, *params_c)
    plt.plot(te_fine, fitted_c, 'r-', linewidth=2, label=f'C-EXP (T2*={1/params_c[1]:.1f} ms)')

    plt.xlabel('Echo Time (ms)', fontsize=12)
    plt.ylabel('Signal Intensity', fontsize=12)
    plt.title(f'T2* Decay Curve - Pixel ({y},{x})', fontsize=14, weight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "example_decay_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved decay curve to {output_dir / 'example_decay_curve.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='T2* Parametric Mapping for Iron Overload',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing multi-echo DICOM files')
    parser.add_argument('--output_dir', type=str, default='../results',
                       help='Output directory for results')
    parser.add_argument('--model', type=str, default='both',
                       choices=['s-exp', 'c-exp', 'both'],
                       help='Fitting model to use')
    parser.add_argument('--threshold', type=float, default=10.0,
                       help='Signal threshold for masking')
    parser.add_argument('--vmax', type=float, default=50.0,
                       help='Maximum T2* for colormap (ms)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("T2* PARAMETRIC MAPPING")
    print("=" * 70)

    # Load data
    print(f"\nLoading multi-echo series from {args.data_dir}...")
    volume, echo_times, datasets = load_multiecho_series(args.data_dir)

    # Plot multi-echo images
    plot_multiecho_images(volume, echo_times, output_dir)

    # Create T2* maps
    models_to_run = ['s-exp', 'c-exp'] if args.model == 'both' else [args.model]

    results = {}
    for model in models_to_run:
        print(f"\n{'='*70}")
        print(f"FITTING WITH {model.upper()} MODEL")
        print(f"{'='*70}")

        start_time = time.time()
        t2star_map, rmse_map, r2star_map = create_t2star_map(
            volume, echo_times, model=model,
            threshold=args.threshold, verbose=True
        )
        elapsed = time.time() - start_time
        print(f"\nProcessing time: {elapsed:.1f} seconds")

        results[model] = {
            't2star': t2star_map,
            'rmse': rmse_map,
            'r2star': r2star_map
        }

        # Plot maps
        plot_t2star_maps(t2star_map, rmse_map, model.upper(), output_dir, args.vmax)

        # Save maps as numpy arrays
        np.save(output_dir / f"t2star_map_{model}.npy", t2star_map)
        np.save(output_dir / f"rmse_map_{model}.npy", rmse_map)

    # Example decay curve (center pixel)
    if len(models_to_run) == 2:
        from utils import fit_t2star_pixel
        h, w = volume.shape[1], volume.shape[2]
        y_center, x_center = h // 2, w // 2

        signal = volume[:, y_center, x_center]
        params_s, _ = fit_t2star_pixel(signal, echo_times, model='s-exp')
        params_c, _ = fit_t2star_pixel(signal, echo_times, model='c-exp')

        plot_decay_curve(volume, echo_times, y_center, x_center,
                        params_s, params_c, output_dir)

    # Compare models
    if 's-exp' in results and 'c-exp' in results:
        diff_map = results['c-exp']['t2star'] - results['s-exp']['t2star']

        plt.figure(figsize=(8, 6))
        plt.imshow(diff_map, cmap='RdBu_r', vmin=-10, vmax=10)
        plt.colorbar(label='T2* Difference (ms)', fraction=0.046, pad=0.04)
        plt.title('T2* Difference: C-EXP - S-EXP', fontsize=14, weight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / "t2star_difference.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved difference map to {output_dir / 't2star_difference.png'}")

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED TO {output_dir}")
    print("=" * 70)
    print("\nFiles created:")
    print("  - multiecho_images.png")
    for model in models_to_run:
        print(f"  - t2star_map_{model}.png")
        print(f"  - t2star_map_{model}.npy")
        print(f"  - rmse_map_{model}.npy")
    if len(models_to_run) == 2:
        print("  - example_decay_curve.png")
        print("  - t2star_difference.png")

    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
