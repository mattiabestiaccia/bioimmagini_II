"""
Standard Deviation Calculation on Synthetic Image with Gaussian Noise.

This script demonstrates noise analysis on a synthetic image by:
1. Creating an ideal image with 6 different intensity patterns
2. Adding Gaussian noise with known sigma
3. Computing a standard deviation map
4. Comparing different methods to estimate sigma

Python equivalent of Calcolo_SD.m
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    create_synthetic_image,
    compute_sd_map,
    estimate_sigma_from_histogram,
    compute_statistics
)


def plot_noisy_image_with_histogram(image: np.ndarray, bins: int = 256, title: str = "Gaussian Noise"):
    """
    Plot noisy image alongside its histogram.

    Parameters
    ----------
    image : np.ndarray
        Image to display
    bins : int
        Number of histogram bins
    title : str
        Figure title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Display image
    im = axes[0].imshow(image, cmap='gray', aspect='equal')
    axes[0].set_title('Noisy Image')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0])

    # Display histogram
    axes[1].hist(image.ravel(), bins=bins, color='blue', alpha=0.7, edgecolor='black')
    axes[1].set_title('Intensity Histogram')
    axes[1].set_xlabel('Intensity')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sd_map_with_histogram(sd_map: np.ndarray, bins: int = 100, title: str = "SD Map"):
    """
    Plot standard deviation map alongside its histogram.

    Parameters
    ----------
    sd_map : np.ndarray
        Standard deviation map
    bins : int
        Number of histogram bins
    title : str
        Figure title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Display SD map
    im = axes[0].imshow(sd_map, cmap='gray', aspect='equal')
    axes[0].set_title('SD Map')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0])

    # Display histogram
    hist, bin_edges = np.histogram(sd_map.ravel(), bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axes[1].plot(bin_centers, hist, color='blue', linewidth=2)
    axes[1].set_title('SD Map Histogram')
    axes[1].set_xlabel('Standard Deviation')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sigma_comparison(sigma_true: float, sigma_mean: float, sigma_median: float, sigma_max: float):
    """
    Compare different sigma estimation methods.

    Parameters
    ----------
    sigma_true : float
        True sigma value
    sigma_mean : float
        Mean of SD map
    sigma_median : float
        Median of SD map
    sigma_max : float
        Maximum of histogram
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['True', 'Mean', 'Median', 'Max Hist']
    values = [sigma_true, sigma_mean, sigma_median, sigma_max]
    colors = ['green', 'blue', 'orange', 'red']

    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Sigma Value', fontsize=12, fontweight='bold')
    ax.set_title('Sigma Calculation Methods Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    return fig


def main(output_dir: Path = None, show_plots: bool = True):
    """
    Main execution function.

    Parameters
    ----------
    output_dir : Path, optional
        Directory to save output figures
    show_plots : bool
        Whether to display plots interactively
    """
    print("=" * 70)
    print("Standard Deviation Calculation on Synthetic Image")
    print("=" * 70)

    # Parameters
    dim = 512
    sigma_true = 5.0
    kernel_size = 5
    histogram_bins = 256

    print(f"\nImage dimensions: {dim}x{dim}")
    print(f"True sigma (Gaussian noise): {sigma_true}")
    print(f"SD map kernel size: {kernel_size}x{kernel_size}")

    # Create synthetic image with noise
    print("\n[1/4] Creating synthetic image with Gaussian noise...")
    image_clean, image_noisy = create_synthetic_image(dim=dim, sigma_noise=sigma_true)
    print(f"  ✓ Image created with {len(np.unique(image_clean))} intensity levels")

    # Plot noisy image
    print("\n[2/4] Plotting noisy image and histogram...")
    fig1 = plot_noisy_image_with_histogram(image_noisy, bins=histogram_bins)
    if output_dir:
        fig1.savefig(output_dir / "01_noisy_image.png", dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: 01_noisy_image.png")

    # Compute SD map
    print("\n[3/4] Computing standard deviation map...")
    sd_map = compute_sd_map(image_noisy, kernel_size=kernel_size)
    print(f"  ✓ SD map computed (shape: {sd_map.shape})")

    # Plot SD map
    fig2 = plot_sd_map_with_histogram(sd_map, bins=100)
    if output_dir:
        fig2.savefig(output_dir / "02_sd_map.png", dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: 02_sd_map.png")

    # Calculate sigma using different methods
    print("\n[4/4] Estimating sigma using different methods...")
    sigma_mean = np.mean(sd_map)
    sigma_median = np.median(sd_map)
    sigma_max, hist, bin_centers = estimate_sigma_from_histogram(sd_map.ravel(), bins=100)

    print(f"\n{'Method':<20} {'Value':<10} {'Error (%)':<10}")
    print("-" * 40)
    print(f"{'True Sigma':<20} {sigma_true:<10.4f} {'-':<10}")
    print(f"{'Mean of SD map':<20} {sigma_mean:<10.4f} {abs(sigma_mean - sigma_true) / sigma_true * 100:<10.2f}")
    print(f"{'Median of SD map':<20} {sigma_median:<10.4f} {abs(sigma_median - sigma_true) / sigma_true * 100:<10.2f}")
    print(f"{'Max Histogram':<20} {sigma_max:<10.4f} {abs(sigma_max - sigma_true) / sigma_true * 100:<10.2f}")

    # Plot comparison
    fig3 = plot_sigma_comparison(sigma_true, sigma_mean, sigma_median, sigma_max)
    if output_dir:
        fig3.savefig(output_dir / "03_sigma_comparison.png", dpi=150, bbox_inches='tight')
        print(f"\n  ✓ Saved: 03_sigma_comparison.png")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    if show_plots:
        plt.show()

    return {
        'sigma_true': sigma_true,
        'sigma_mean': sigma_mean,
        'sigma_median': sigma_median,
        'sigma_max': sigma_max,
        'sd_map': sd_map,
        'image_noisy': image_noisy
    }


if __name__ == "__main__":
    # Setup output directory
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / "calcolo_sd"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    results = main(output_dir=results_dir, show_plots=True)
