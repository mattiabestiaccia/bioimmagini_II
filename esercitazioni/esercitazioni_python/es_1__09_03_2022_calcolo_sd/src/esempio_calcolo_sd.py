"""
Noise Calculation on MRI Phantom Image.

This script demonstrates noise analysis on a real MRI phantom using:
1. Manual ROI-based measurement (with interactive ROI drawing)
2. Automatic SD map analysis (with different kernel sizes and thresholds)
3. Rayleigh correction for background noise

Python equivalent of EsempioCalcoloSD.m
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import pydicom
from pathlib import Path
import sys
from typing import Tuple, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    compute_sd_map,
    estimate_sigma_from_histogram,
    apply_rayleigh_correction,
    exclude_zero_padding,
    rayleigh_correction_factor
)

# Try to import centralized DICOM module
try:
    project_root = Path(__file__).parent.parent.parent.parent.parent
    src_path = project_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        from dicom_import import read_dicom_file, extract_metadata
        DICOM_IMPORT_AVAILABLE = True
    else:
        DICOM_IMPORT_AVAILABLE = False
except ImportError:
    DICOM_IMPORT_AVAILABLE = False


def load_dicom_image(dicom_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Load DICOM image using centralized module if available.

    Parameters
    ----------
    dicom_path : Path
        Path to DICOM file

    Returns
    -------
    image : np.ndarray
        Pixel array as float
    metadata : dict
        Dictionary with DICOM metadata
    """
    if DICOM_IMPORT_AVAILABLE:
        try:
            pixel_data, ds = read_dicom_file(dicom_path)
            metadata = extract_metadata(ds)
            return pixel_data.astype(float), metadata
        except Exception as e:
            print(f"Centralized DICOM loading failed: {e}, using fallback")

    # Fallback to direct pydicom
    dcm = pydicom.dcmread(dicom_path)
    metadata = {
        'patient': {
            'patient_name': str(dcm.get('PatientName', 'N/A')),
            'patient_id': str(dcm.get('PatientID', 'N/A')),
        },
        'study': {
            'study_date': str(dcm.get('StudyDate', 'N/A')),
            'study_description': str(dcm.get('StudyDescription', 'N/A')),
        },
        'image': {
            'rows': int(dcm.get('Rows', 0)),
            'columns': int(dcm.get('Columns', 0)),
        }
    }
    return dcm.pixel_array.astype(float), metadata


class ROISelector:
    """Interactive ROI selection tool for circular regions."""

    def __init__(self, image: np.ndarray, roi_names: List[str], roi_colors: List[str]):
        """
        Initialize ROI selector.

        Parameters
        ----------
        image : np.ndarray
            Image to display
        roi_names : List[str]
            Names for each ROI
        roi_colors : List[str]
            Colors for each ROI
        """
        self.image = image
        self.roi_names = roi_names
        self.roi_colors = roi_colors
        self.rois = []
        self.current_roi = 0
        self.center = None
        self.circle = None

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.imshow(image, cmap='gray')
        self.ax.set_title(f'Click center, then edge for: {roi_names[0]}', fontsize=12, fontweight='bold')

        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return

        if self.center is None:
            # First click: set center
            self.center = (event.xdata, event.ydata)
            self.ax.plot(self.center[0], self.center[1], 'r+', markersize=15, markeredgewidth=2)
            self.fig.canvas.draw()
        else:
            # Second click: set radius and create ROI
            radius = np.sqrt((event.xdata - self.center[0])**2 + (event.ydata - self.center[1])**2)

            # Create circular mask
            y, x = np.ogrid[:self.image.shape[0], :self.image.shape[1]]
            mask = (x - self.center[0])**2 + (y - self.center[1])**2 <= radius**2

            # Draw circle
            color = self.roi_colors[self.current_roi]
            circle = Circle(self.center, radius, fill=False, edgecolor=color, linewidth=2,
                          label=self.roi_names[self.current_roi])
            self.ax.add_patch(circle)

            # Add label
            self.ax.text(self.center[0], self.center[1] - radius - 10,
                        self.roi_names[self.current_roi],
                        color=color, fontsize=12, fontweight='bold',
                        ha='center', va='bottom')

            self.fig.canvas.draw()

            # Store ROI
            self.rois.append({
                'name': self.roi_names[self.current_roi],
                'center': self.center,
                'radius': radius,
                'mask': mask,
                'color': color
            })

            # Move to next ROI
            self.current_roi += 1
            self.center = None

            if self.current_roi < len(self.roi_names):
                self.ax.set_title(f'Click center, then edge for: {self.roi_names[self.current_roi]}',
                                fontsize=12, fontweight='bold')
            else:
                self.ax.set_title('ROI selection complete - Close window to continue',
                                fontsize=12, fontweight='bold', color='green')
                self.fig.canvas.mpl_disconnect(self.cid_press)

    def get_rois(self) -> List[dict]:
        """Return collected ROIs."""
        plt.show()
        return self.rois


def manual_roi_analysis(image: np.ndarray, interactive: bool = True) -> dict:
    """
    Perform manual ROI-based noise analysis.

    Parameters
    ----------
    image : np.ndarray
        MRI phantom image
    interactive : bool
        If True, use interactive ROI selection; otherwise use default ROIs

    Returns
    -------
    dict
        Dictionary with ROI statistics
    """
    print("\n[Manual ROI Analysis]")

    if interactive:
        print("  → Drawing ROIs interactively...")
        print("    Instructions: Click center, then click edge to define radius")
        roi_selector = ROISelector(
            image,
            roi_names=['Oil', 'Water', 'Background'],
            roi_colors=['green', 'red', 'white']
        )
        rois = roi_selector.get_rois()
    else:
        # Default ROIs (approximate positions for phantom.dcm)
        print("  → Using default ROI positions...")
        h, w = image.shape
        rois = [
            {'name': 'Oil', 'center': (w*0.3, h*0.3), 'radius': 20},
            {'name': 'Water', 'center': (w*0.7, h*0.3), 'radius': 20},
            {'name': 'Background', 'center': (w*0.5, h*0.8), 'radius': 15}
        ]
        # Create masks
        for roi in rois:
            y, x = np.ogrid[:h, :w]
            mask = (x - roi['center'][0])**2 + (y - roi['center'][1])**2 <= roi['radius']**2
            roi['mask'] = mask

    # Calculate statistics for each ROI
    results = {}
    print(f"\n  {'ROI':<15} {'Mean':<12} {'Std Dev':<12} {'N pixels':<10}")
    print("  " + "-" * 50)

    for roi in rois:
        roi_values = image[roi['mask']]
        mean_val = np.mean(roi_values)
        std_val = np.std(roi_values, ddof=1)  # Use ddof=1 for sample std (like MATLAB)

        results[roi['name']] = {
            'mean': mean_val,
            'std': std_val,
            'n_pixels': len(roi_values)
        }

        print(f"  {roi['name']:<15} {mean_val:<12.2f} {std_val:<12.4f} {len(roi_values):<10}")

    # Apply Rayleigh correction to background
    std_back_corr = apply_rayleigh_correction(results['Background']['std'])
    results['Background']['std_corrected'] = std_back_corr

    print(f"\n  Background Rayleigh correction factor: {rayleigh_correction_factor():.4f}")
    print(f"  Background SD corrected: {std_back_corr:.4f}")

    return results, rois


def automatic_sd_map_analysis(image: np.ndarray, kernel_size: int = 3,
                              threshold: float = 0.0, bins: int = 256) -> dict:
    """
    Perform automatic SD map analysis.

    Parameters
    ----------
    image : np.ndarray
        MRI phantom image
    kernel_size : int
        Kernel size for SD map calculation
    threshold : float
        Intensity threshold to exclude background/padding
    bins : int
        Number of histogram bins

    Returns
    -------
    dict
        Dictionary with SD map statistics
    """
    print(f"\n[Automatic SD Map Analysis - Kernel {kernel_size}x{kernel_size}, Threshold: {threshold}]")

    # Compute SD map
    sd_map = compute_sd_map(image, kernel_size=kernel_size)

    # Exclude zero padding and low intensities
    if threshold > 0:
        mask = image > threshold
        valid_values = sd_map[mask]
        print(f"  → Using {len(valid_values)} pixels (intensity > {threshold})")
    else:
        mask = image != 0
        valid_values = sd_map[mask]
        print(f"  → Excluding zero-padding: {len(valid_values)} valid pixels")

    # Calculate statistics
    mean_sd = np.mean(valid_values)
    median_sd = np.median(valid_values)

    # Estimate from histogram
    sigma_hist, hist, bin_centers = estimate_sigma_from_histogram(valid_values, bins=bins)

    print(f"\n  {'Method':<20} {'Value':<12}")
    print("  " + "-" * 32)
    print(f"  {'Mean':<20} {mean_sd:<12.4f}")
    print(f"  {'Median':<20} {median_sd:<12.4f}")
    print(f"  {'Histogram Max':<20} {sigma_hist:<12.4f}")

    return {
        'sd_map': sd_map,
        'mean': mean_sd,
        'median': median_sd,
        'hist_max': sigma_hist,
        'hist': hist,
        'bin_centers': bin_centers,
        'mask': mask
    }


def plot_sd_map_analysis(image: np.ndarray, sd_map: np.ndarray, mask: np.ndarray,
                        hist: np.ndarray, bin_centers: np.ndarray, title: str):
    """Plot SD map with histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # SD map
    im = axes[0].imshow(sd_map, cmap='gray')
    axes[0].set_title('SD Map')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0])

    # Histogram
    axes[1].bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0],
               color='blue', alpha=0.7, edgecolor='black')
    axes[1].set_title('SD Histogram (excluding masked pixels)')
    axes[1].set_xlabel('Standard Deviation')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main(dicom_path: Path, output_dir: Path = None, interactive: bool = False, show_plots: bool = True):
    """
    Main execution function.

    Parameters
    ----------
    dicom_path : Path
        Path to phantom DICOM file
    output_dir : Path, optional
        Directory to save output figures
    interactive : bool
        Use interactive ROI selection
    show_plots : bool
        Whether to display plots
    """
    print("=" * 70)
    print("MRI Phantom Noise Analysis")
    print("=" * 70)

    # Load DICOM image
    print(f"\nLoading DICOM: {dicom_path.name}")
    if DICOM_IMPORT_AVAILABLE:
        print("  Using centralized dicom_import module")
    image, metadata = load_dicom_image(dicom_path)

    print(f"  Image shape: {image.shape}")
    print(f"  Intensity range: [{image.min():.1f}, {image.max():.1f}]")
    print(f"  Patient: {metadata['patient'].get('patient_name', 'N/A')}")
    print(f"  Study Date: {metadata['study'].get('study_date', 'N/A')}")

    # Section 1: Manual ROI analysis
    print("\n" + "=" * 70)
    print("SECTION 1: Manual ROI-based Analysis")
    print("=" * 70)
    roi_results, rois = manual_roi_analysis(image, interactive=interactive)

    # Section 2: Automatic SD map (kernel 3x3, no threshold)
    print("\n" + "=" * 70)
    print("SECTION 2: Automatic SD Map Analysis (3x3 kernel)")
    print("=" * 70)
    auto_results_3x3 = automatic_sd_map_analysis(image, kernel_size=3, threshold=0)

    fig1 = plot_sd_map_analysis(
        image, auto_results_3x3['sd_map'], auto_results_3x3['mask'],
        auto_results_3x3['hist'], auto_results_3x3['bin_centers'],
        "SD Map Analysis - 3x3 Kernel"
    )
    if output_dir:
        fig1.savefig(output_dir / "01_sd_map_3x3.png", dpi=150, bbox_inches='tight')

    # Section 3: Automatic SD map (kernel 9x9, threshold 100)
    print("\n" + "=" * 70)
    print("SECTION 3: Automatic SD Map Analysis (9x9 kernel, threshold>100)")
    print("=" * 70)
    auto_results_9x9 = automatic_sd_map_analysis(image, kernel_size=9, threshold=100, bins=128)

    fig2 = plot_sd_map_analysis(
        image, auto_results_9x9['sd_map'], auto_results_9x9['mask'],
        auto_results_9x9['hist'], auto_results_9x9['bin_centers'],
        "SD Map Analysis - 9x9 Kernel, Threshold > 100"
    )
    if output_dir:
        fig2.savefig(output_dir / "02_sd_map_9x9_threshold.png", dpi=150, bbox_inches='tight')

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n{'Method':<30} {'SD Value':<12}")
    print("-" * 42)
    print(f"{'Manual - Background':<30} {roi_results['Background']['std']:<12.4f}")
    print(f"{'Manual - Background (Corr.)':<30} {roi_results['Background']['std_corrected']:<12.4f}")
    print(f"{'Manual - Water':<30} {roi_results['Water']['std']:<12.4f}")
    print(f"{'Manual - Oil':<30} {roi_results['Oil']['std']:<12.4f}")
    print(f"{'Auto 3x3 - Mean':<30} {auto_results_3x3['mean']:<12.4f}")
    print(f"{'Auto 3x3 - Hist Max':<30} {auto_results_3x3['hist_max']:<12.4f}")
    print(f"{'Auto 9x9 (th>100) - Mean':<30} {auto_results_9x9['mean']:<12.4f}")
    print(f"{'Auto 9x9 (th>100) - Hist Max':<30} {auto_results_9x9['hist_max']:<12.4f}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    if show_plots:
        plt.show()

    return {
        'roi_results': roi_results,
        'auto_3x3': auto_results_3x3,
        'auto_9x9': auto_results_9x9
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MRI Phantom Noise Analysis')
    parser.add_argument('--dicom', type=str,
                       default='../data/phantom.dcm',
                       help='Path to phantom DICOM file')
    parser.add_argument('--interactive', action='store_true',
                       help='Use interactive ROI selection')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display plots')

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    dicom_path = script_dir.parent / args.dicom
    results_dir = script_dir.parent / "results" / "esempio_calcolo_sd"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not dicom_path.exists():
        print(f"ERROR: DICOM file not found: {dicom_path}")
        print("Please copy phantom.dcm to the data directory")
        sys.exit(1)

    # Run analysis
    results = main(dicom_path, output_dir=results_dir,
                  interactive=args.interactive,
                  show_plots=not args.no_display)
