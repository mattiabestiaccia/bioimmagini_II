"""
Monte Carlo Test for Mean and Standard Deviation Convergence.

This script demonstrates how mean and standard deviation estimates
converge with increasing ROI size through Monte Carlo simulation.

Python equivalent of Test_M_SD.m
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def monte_carlo_roi_test(true_mean: float = 30.0,
                         true_sd: float = 22.0,
                         n_simulations: int = 100,
                         roi_powers: range = range(1, 8)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Monte Carlo simulations to test convergence of mean and SD estimates.

    Parameters
    ----------
    true_mean : float
        True mean value to impose (default: 30.0)
    true_sd : float
        True standard deviation to impose (default: 22.0)
    n_simulations : int
        Number of Monte Carlo simulations (default: 100)
    roi_powers : range
        Powers of 2 for ROI dimensions (default: range(1, 8) -> [2, 4, 8, 16, 32, 64, 128])

    Returns
    -------
    dimensions : np.ndarray
        ROI dimensions tested
    all_means : np.ndarray
        Mean values for all simulations and dimensions (n_simulations x n_dimensions)
    all_sds : np.ndarray
        SD values for all simulations and dimensions (n_simulations x n_dimensions)

    Examples
    --------
    >>> dims, means, sds = monte_carlo_roi_test(true_mean=30, true_sd=22, n_simulations=50)
    >>> print(f"Dimensions tested: {dims}")
    >>> print(f"Mean convergence at largest ROI: {np.mean(means[:, -1]):.2f}")
    """
    dimensions = 2 ** np.array(list(roi_powers))
    n_dims = len(dimensions)

    all_means = np.zeros((n_simulations, n_dims))
    all_sds = np.zeros((n_simulations, n_dims))

    print(f"Running {n_simulations} Monte Carlo simulations...")
    print(f"Testing ROI dimensions: {dimensions.tolist()}")
    print(f"True mean: {true_mean}, True SD: {true_sd}")

    for sim in range(n_simulations):
        if (sim + 1) % 20 == 0:
            print(f"  Simulation {sim + 1}/{n_simulations}...")

        for i, dim in enumerate(dimensions):
            # Generate random image with true mean and SD
            image = np.random.normal(true_mean, true_sd, (dim, dim))

            # Calculate mean and SD
            all_means[sim, i] = np.mean(image)
            all_sds[sim, i] = np.std(image, ddof=1)  # Sample SD (like MATLAB)

    print("  ✓ Simulations complete")

    return dimensions, all_means, all_sds


def plot_mean_convergence(dimensions: np.ndarray, all_means: np.ndarray, true_mean: float):
    """
    Plot mean convergence across ROI dimensions.

    Parameters
    ----------
    dimensions : np.ndarray
        ROI dimensions
    all_means : np.ndarray
        Mean values (n_simulations x n_dimensions)
    true_mean : float
        True mean value
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot all simulation trajectories
    for i in range(all_means.shape[0]):
        ax.plot(dimensions, all_means[i, :], 'b-', alpha=0.15, linewidth=0.8)

    # Plot mean trajectory
    mean_trajectory = np.mean(all_means, axis=0)
    ax.plot(dimensions, mean_trajectory, 'r-', linewidth=3,
           label=f'Mean of simulations', marker='o', markersize=8)

    # Plot true value
    ax.axhline(y=true_mean, color='green', linestyle='--', linewidth=2.5,
              label=f'True Mean = {true_mean}')

    ax.set_xlabel('ROI Dimension (pixels)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Estimated Mean', fontsize=13, fontweight='bold')
    ax.set_title(f'Mean Convergence with ROI Size ({all_means.shape[0]} simulations)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # Add annotations
    final_mean = mean_trajectory[-1]
    final_error = abs(final_mean - true_mean)
    ax.text(0.02, 0.98, f'Final mean: {final_mean:.4f}\nError: {final_error:.4f} ({final_error/true_mean*100:.2f}%)',
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_sd_convergence(dimensions: np.ndarray, all_sds: np.ndarray, true_sd: float):
    """
    Plot standard deviation convergence across ROI dimensions.

    Parameters
    ----------
    dimensions : np.ndarray
        ROI dimensions
    all_sds : np.ndarray
        SD values (n_simulations x n_dimensions)
    true_sd : float
        True standard deviation
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot all simulation trajectories
    for i in range(all_sds.shape[0]):
        ax.plot(dimensions, all_sds[i, :], 'b-', alpha=0.15, linewidth=0.8)

    # Plot mean trajectory
    mean_trajectory = np.mean(all_sds, axis=0)
    ax.plot(dimensions, mean_trajectory, 'r-', linewidth=3,
           label=f'Mean of simulations', marker='o', markersize=8)

    # Plot true value
    ax.axhline(y=true_sd, color='green', linestyle='--', linewidth=2.5,
              label=f'True SD = {true_sd}')

    ax.set_xlabel('ROI Dimension (pixels)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Estimated Standard Deviation', fontsize=13, fontweight='bold')
    ax.set_title(f'Standard Deviation Convergence with ROI Size ({all_sds.shape[0]} simulations)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # Add annotations
    final_sd = mean_trajectory[-1]
    final_error = abs(final_sd - true_sd)
    ax.text(0.02, 0.98, f'Final SD: {final_sd:.4f}\nError: {final_error:.4f} ({final_error/true_sd*100:.2f}%)',
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_combined_statistics(dimensions: np.ndarray, all_means: np.ndarray, all_sds: np.ndarray,
                             true_mean: float, true_sd: float):
    """
    Plot combined statistics: mean, std, and coefficient of variation.

    Parameters
    ----------
    dimensions : np.ndarray
        ROI dimensions
    all_means : np.ndarray
        Mean values (n_simulations x n_dimensions)
    all_sds : np.ndarray
        SD values (n_simulations x n_dimensions)
    true_mean : float
        True mean value
    true_sd : float
        True standard deviation
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Statistical Convergence Analysis', fontsize=16, fontweight='bold')

    # Mean convergence
    mean_traj = np.mean(all_means, axis=0)
    std_traj = np.mean(all_sds, axis=0)

    # 1. Mean of means
    ax = axes[0, 0]
    ax.plot(dimensions, mean_traj, 'b-o', linewidth=2, markersize=6)
    ax.axhline(y=true_mean, color='red', linestyle='--', linewidth=2, label=f'True: {true_mean}')
    ax.set_xlabel('ROI Dimension')
    ax.set_ylabel('Mean of Estimated Means')
    ax.set_title('Mean Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # 2. Mean of SDs
    ax = axes[0, 1]
    ax.plot(dimensions, std_traj, 'g-o', linewidth=2, markersize=6)
    ax.axhline(y=true_sd, color='red', linestyle='--', linewidth=2, label=f'True: {true_sd}')
    ax.set_xlabel('ROI Dimension')
    ax.set_ylabel('Mean of Estimated SDs')
    ax.set_title('SD Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # 3. Variance of means (precision)
    ax = axes[1, 0]
    var_means = np.var(all_means, axis=0, ddof=1)
    ax.plot(dimensions, var_means, 'purple', linewidth=2, marker='s', markersize=6)
    ax.set_xlabel('ROI Dimension')
    ax.set_ylabel('Variance of Estimated Means')
    ax.set_title('Precision of Mean Estimation')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    # 4. Variance of SDs (precision)
    ax = axes[1, 1]
    var_sds = np.var(all_sds, axis=0, ddof=1)
    ax.plot(dimensions, var_sds, 'orange', linewidth=2, marker='^', markersize=6)
    ax.set_xlabel('ROI Dimension')
    ax.set_ylabel('Variance of Estimated SDs')
    ax.set_title('Precision of SD Estimation')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    plt.tight_layout()
    return fig


def print_statistics_table(dimensions: np.ndarray, all_means: np.ndarray, all_sds: np.ndarray,
                           true_mean: float, true_sd: float):
    """Print detailed statistics table."""
    print("\n" + "=" * 90)
    print("CONVERGENCE STATISTICS TABLE")
    print("=" * 90)

    mean_traj = np.mean(all_means, axis=0)
    sd_traj = np.mean(all_sds, axis=0)
    std_of_means = np.std(all_means, axis=0, ddof=1)
    std_of_sds = np.std(all_sds, axis=0, ddof=1)

    print(f"\n{'Dim':<8} {'Est.Mean':<12} {'Mean Err%':<12} {'Mean StD':<12} {'Est.SD':<12} {'SD Err%':<12} {'SD StD':<12}")
    print("-" * 90)

    for i, dim in enumerate(dimensions):
        mean_err = abs(mean_traj[i] - true_mean) / true_mean * 100
        sd_err = abs(sd_traj[i] - true_sd) / true_sd * 100

        print(f"{dim:<8} {mean_traj[i]:<12.4f} {mean_err:<12.2f} {std_of_means[i]:<12.4f} "
              f"{sd_traj[i]:<12.4f} {sd_err:<12.2f} {std_of_sds[i]:<12.4f}")

    print("-" * 90)
    print(f"True values: Mean = {true_mean}, SD = {true_sd}")
    print("=" * 90)


def main(output_dir: Path = None, show_plots: bool = True):
    """
    Main execution function.

    Parameters
    ----------
    output_dir : Path, optional
        Directory to save output figures
    show_plots : bool
        Whether to display plots
    """
    print("=" * 70)
    print("Monte Carlo Test: ROI Size vs Statistical Accuracy")
    print("=" * 70)

    # Parameters
    true_mean = 30.0
    true_sd = 22.0
    n_simulations = 100
    roi_powers = range(1, 8)  # 2^1 to 2^7 -> [2, 4, 8, 16, 32, 64, 128]

    print(f"\nSimulation parameters:")
    print(f"  True Mean (imposed): {true_mean}")
    print(f"  True SD (imposed): {true_sd}")
    print(f"  Number of simulations: {n_simulations}")
    print(f"  ROI dimensions: 2^{min(roi_powers)} to 2^{max(roi_powers)}")

    # Run Monte Carlo simulations
    print("\n" + "-" * 70)
    dimensions, all_means, all_sds = monte_carlo_roi_test(
        true_mean=true_mean,
        true_sd=true_sd,
        n_simulations=n_simulations,
        roi_powers=roi_powers
    )

    # Print statistics table
    print_statistics_table(dimensions, all_means, all_sds, true_mean, true_sd)

    # Generate plots
    print("\n" + "-" * 70)
    print("Generating plots...")

    fig1 = plot_mean_convergence(dimensions, all_means, true_mean)
    if output_dir:
        fig1.savefig(output_dir / "01_mean_convergence.png", dpi=150, bbox_inches='tight')
        print("  ✓ Saved: 01_mean_convergence.png")

    fig2 = plot_sd_convergence(dimensions, all_sds, true_sd)
    if output_dir:
        fig2.savefig(output_dir / "02_sd_convergence.png", dpi=150, bbox_inches='tight')
        print("  ✓ Saved: 02_sd_convergence.png")

    fig3 = plot_combined_statistics(dimensions, all_means, all_sds, true_mean, true_sd)
    if output_dir:
        fig3.savefig(output_dir / "03_combined_statistics.png", dpi=150, bbox_inches='tight')
        print("  ✓ Saved: 03_combined_statistics.png")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    # Key findings
    print("\nKEY FINDINGS:")
    print("-" * 70)
    print("1. Mean estimates converge quickly with small ROIs")
    print("2. SD estimates require larger ROIs for accurate convergence")
    print("3. Variance of estimates decreases with ROI size (better precision)")
    print("4. At 128x128 pixels, estimates are very close to true values")

    if show_plots:
        plt.show()

    return {
        'dimensions': dimensions,
        'all_means': all_means,
        'all_sds': all_sds
    }


if __name__ == "__main__":
    # Setup output directory
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / "test_m_sd"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    results = main(output_dir=results_dir, show_plots=True)
