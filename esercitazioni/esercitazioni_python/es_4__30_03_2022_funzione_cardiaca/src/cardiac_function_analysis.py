#!/usr/bin/env python3
"""
cardiac_function_analysis.py - Cardiac Function Analysis Pipeline

Complete pipeline for left ventricle segmentation and cardiac parameter calculation
using Active Contours (Chan-Vese) on cardiac MRI cine images.

Esercitazione 4: Segmentazione Funzione Cardiaca
Dataset: 15 slices x 30 temporal frames (450 DICOM images)

Author: Biomedical Imaging Course
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_cardiac_4d,
    find_cardiac_phases,
    create_circular_seed,
    segment_lv_active_contour,
    refine_segmentation,
    compute_volume_from_masks,
    calculate_bsa,
    calculate_cardiac_parameters,
    generate_cardiac_report
)


def plot_4d_overview(volume_4d: np.ndarray, output_dir: Path):
    """
    Plot overview of 4D cardiac dataset (montage style).

    Parameters
    ----------
    volume_4d : np.ndarray
        4D cardiac volume (n_frames, n_slices, height, width)
    output_dir : Path
        Directory to save plot
    """
    n_frames, n_slices = volume_4d.shape[:2]

    # Select subset of frames for visualization (every 3rd frame)
    frame_indices = range(0, n_frames, 3)
    n_frames_display = len(frame_indices)

    fig, axes = plt.subplots(n_slices, n_frames_display, figsize=(20, 12))

    if n_slices == 1:
        axes = axes.reshape(1, -1)

    for slice_idx in range(n_slices):
        for col_idx, frame_idx in enumerate(frame_indices):
            ax = axes[slice_idx, col_idx]
            ax.imshow(volume_4d[frame_idx, slice_idx], cmap='gray')
            ax.axis('off')

            if slice_idx == 0:
                ax.set_title(f'F{frame_idx}', fontsize=8)

            if col_idx == 0:
                ax.text(-0.2, 0.5, f'S{slice_idx}', transform=ax.transAxes,
                       fontsize=10, va='center', rotation=90)

    plt.suptitle('4D Cardiac MRI Overview (Slices x Frames)', fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_path = output_dir / "cardiac_4d_overview.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved 4D overview to {output_path}")


def plot_phase_comparison(diastolic_volume: np.ndarray,
                         systolic_volume: np.ndarray,
                         diastolic_frame: int,
                         systolic_frame: int,
                         output_dir: Path):
    """
    Plot comparison of diastolic and systolic phases.

    Parameters
    ----------
    diastolic_volume : np.ndarray
        Diastolic volume (n_slices, height, width)
    systolic_volume : np.ndarray
        Systolic volume (n_slices, height, width)
    diastolic_frame : int
        Diastolic frame index
    systolic_frame : int
        Systolic frame index
    output_dir : Path
        Directory to save plot
    """
    n_slices = diastolic_volume.shape[0]

    # Select middle slices for visualization
    slice_indices = range(3, min(14, n_slices))
    n_display = len(slice_indices)

    fig, axes = plt.subplots(2, n_display, figsize=(15, 6))

    if n_display == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, slice_idx in enumerate(slice_indices):
        # Diastolic
        axes[0, col_idx].imshow(diastolic_volume[slice_idx], cmap='gray')
        axes[0, col_idx].set_title(f'Slice {slice_idx}', fontsize=10)
        axes[0, col_idx].axis('off')

        # Systolic
        axes[1, col_idx].imshow(systolic_volume[slice_idx], cmap='gray')
        axes[1, col_idx].axis('off')

    axes[0, 0].text(-0.3, 0.5, f'Diastole (F{diastolic_frame})', transform=axes[0, 0].transAxes,
                    fontsize=12, va='center', rotation=90, weight='bold')
    axes[1, 0].text(-0.3, 0.5, f'Systole (F{systolic_frame})', transform=axes[1, 0].transAxes,
                    fontsize=12, va='center', rotation=90, weight='bold')

    plt.suptitle('Cardiac Phases: Diastole vs Systole', fontsize=14, y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = output_dir / "cardiac_phases_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved phase comparison to {output_path}")


def plot_segmentation_results(volume: np.ndarray,
                             masks: np.ndarray,
                             phase_name: str,
                             frame_idx: int,
                             output_dir: Path):
    """
    Plot segmentation results for all slices.

    Parameters
    ----------
    volume : np.ndarray
        Volume (n_slices, height, width)
    masks : np.ndarray
        Segmentation masks (n_slices, height, width)
    phase_name : str
        Phase name ('Diastolic' or 'Systolic')
    frame_idx : int
        Frame index
    output_dir : Path
        Directory to save plot
    """
    n_slices = volume.shape[0]

    # Select slices with LV cavity
    if phase_name == 'Diastolic':
        slice_indices = range(3, min(14, n_slices))
    else:  # Systolic
        slice_indices = range(4, min(13, n_slices))

    n_display = len(slice_indices)

    fig, axes = plt.subplots(2, n_display, figsize=(15, 6))

    if n_display == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, slice_idx in enumerate(slice_indices):
        # Original image
        axes[0, col_idx].imshow(volume[slice_idx], cmap='gray')
        axes[0, col_idx].set_title(f'Slice {slice_idx}', fontsize=10)
        axes[0, col_idx].axis('off')

        # Segmentation overlay
        axes[1, col_idx].imshow(volume[slice_idx], cmap='gray')

        # Overlay mask contour in green
        from skimage import segmentation as skimage_seg
        contours = skimage_seg.find_boundaries(masks[slice_idx], mode='thick')
        axes[1, col_idx].contour(contours, colors='lime', linewidths=2)

        axes[1, col_idx].axis('off')

        # Display area
        area_pixels = np.sum(masks[slice_idx])
        axes[1, col_idx].text(0.5, -0.1, f'{area_pixels:.0f} px',
                             transform=axes[1, col_idx].transAxes,
                             fontsize=8, ha='center')

    axes[0, 0].text(-0.2, 0.5, 'Original', transform=axes[0, 0].transAxes,
                    fontsize=11, va='center', rotation=90, weight='bold')
    axes[1, 0].text(-0.2, 0.5, 'Segmentation', transform=axes[1, 0].transAxes,
                    fontsize=11, va='center', rotation=90, weight='bold')

    plt.suptitle(f'{phase_name} Phase Segmentation (Frame {frame_idx})', fontsize=14, y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    filename = f"segmentation_{phase_name.lower()}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved {phase_name.lower()} segmentation to {output_path}")


def plot_volume_curves(edlv: float, eslv: float, output_dir: Path):
    """
    Plot volume bar chart.

    Parameters
    ----------
    edlv : float
        End-diastolic volume (mL)
    eslv : float
        End-systolic volume (mL)
    output_dir : Path
        Directory to save plot
    """
    stroke_volume = edlv - eslv
    ejection_fraction = (stroke_volume / edlv) * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Volume bar chart
    volumes = [edlv, eslv, stroke_volume]
    labels = ['EDLV', 'ESLV', 'Stroke Volume']
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    bars = ax1.bar(labels, volumes, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Volume (mL)', fontsize=12)
    ax1.set_title('Ventricular Volumes', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, vol in zip(bars, volumes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{vol:.1f} mL', ha='center', va='bottom', fontsize=11, weight='bold')

    # Ejection fraction pie chart
    colors_pie = ['#F18F01', '#E0E0E0']
    ax2.pie([ejection_fraction, 100 - ejection_fraction],
           labels=['Ejected', 'Remaining'],
           colors=colors_pie,
           autopct='%1.1f%%',
           startangle=90,
           textprops={'fontsize': 12, 'weight': 'bold'})
    ax2.set_title(f'Ejection Fraction: {ejection_fraction:.0f}%', fontsize=14, weight='bold')

    plt.tight_layout()

    output_path = output_dir / "cardiac_volumes.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved volume plots to {output_path}")


def segment_phase(volume: np.ndarray,
                 phase_name: str,
                 slice_range: range,
                 seed_centers: dict,
                 seed_radius: int = 30,
                 n_iterations: int = 100,
                 smoothing: float = 2.0,
                 verbose: bool = True) -> np.ndarray:
    """
    Segment all slices in a cardiac phase.

    Parameters
    ----------
    volume : np.ndarray
        Volume (n_slices, height, width)
    phase_name : str
        Phase name for logging
    slice_range : range
        Range of slices to segment
    seed_centers : dict
        Dictionary mapping slice index to seed center (y, x)
    seed_radius : int, optional
        Seed radius (default: 30)
    n_iterations : int, optional
        Active contour iterations (default: 100)
    smoothing : float, optional
        Smoothing factor (default: 2.0)
    verbose : bool, optional
        Print progress (default: True)

    Returns
    -------
    masks : np.ndarray
        Segmentation masks (n_slices, height, width)
    """
    n_slices, height, width = volume.shape
    masks = np.zeros((n_slices, height, width), dtype=np.uint8)

    if verbose:
        print(f"\nSegmenting {phase_name} phase (slices {slice_range.start}-{slice_range.stop-1})...")

    previous_mask = None

    for slice_idx in slice_range:
        if verbose:
            print(f"  Slice {slice_idx}...", end=' ')

        image = volume[slice_idx]

        # Get seed for this slice
        if slice_idx in seed_centers:
            # User-specified seed center
            center = seed_centers[slice_idx]
            seed_mask = create_circular_seed(image.shape, center=center, radius=seed_radius)
        elif previous_mask is not None:
            # Use previous slice mask as seed
            seed_mask = previous_mask
        else:
            # Default: center seed
            seed_mask = create_circular_seed(image.shape, center=None, radius=seed_radius)

        # Segment with active contours
        try:
            mask = segment_lv_active_contour(
                image,
                seed_mask,
                n_iterations=n_iterations,
                smoothing=smoothing
            )

            # Refine segmentation
            mask = refine_segmentation(mask, min_area=100, fill_holes=True)

            masks[slice_idx] = mask
            previous_mask = mask

            area = np.sum(mask)
            if verbose:
                print(f"Area: {area:.0f} pixels")

        except Exception as e:
            print(f"Error: {e}")
            masks[slice_idx] = seed_mask  # Fallback to seed

    return masks


def main():
    """
    Main function for cardiac function analysis.
    """
    parser = argparse.ArgumentParser(
        description='Cardiac Function Analysis with Active Contours',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data/FUNZIONE',
        help='Directory containing DICOM files'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results',
        help='Directory to save results'
    )

    parser.add_argument(
        '--diastolic_frame',
        type=int,
        default=None,
        help='Diastolic frame index (if None, auto-detect)'
    )

    parser.add_argument(
        '--systolic_frame',
        type=int,
        default=None,
        help='Systolic frame index (if None, auto-detect)'
    )

    parser.add_argument(
        '--seed_radius',
        type=int,
        default=30,
        help='Seed radius for initial contour (pixels)'
    )

    parser.add_argument(
        '--n_iterations',
        type=int,
        default=100,
        help='Number of active contour iterations'
    )

    parser.add_argument(
        '--smoothing',
        type=float,
        default=2.0,
        help='Smoothing factor for active contours'
    )

    parser.add_argument(
        '--weight',
        type=float,
        default=47.0,
        help='Patient weight (kg)'
    )

    parser.add_argument(
        '--height',
        type=float,
        default=180.0,
        help='Patient height (cm)'
    )

    parser.add_argument(
        '--heart_rate',
        type=float,
        default=68.0,
        help='Heart rate (bpm)'
    )

    parser.add_argument(
        '--skip_overview',
        action='store_true',
        help='Skip 4D overview plot (faster)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CARDIAC FUNCTION ANALYSIS WITH ACTIVE CONTOURS")
    print("=" * 70)

    # Load 4D cardiac dataset
    print(f"\nLoading 4D cardiac dataset from {args.data_dir}...")
    volume_4d, datasets, metadata = load_cardiac_4d(args.data_dir)

    # Plot 4D overview (optional, can be slow)
    if not args.skip_overview:
        print("\nGenerating 4D overview plot...")
        plot_4d_overview(volume_4d, output_dir)

    # Find cardiac phases
    print("\nIdentifying cardiac phases...")
    if args.diastolic_frame is not None and args.systolic_frame is not None:
        diastolic_frame = args.diastolic_frame
        systolic_frame = args.systolic_frame
        print(f"  Using user-specified frames:")
        print(f"    Diastolic: {diastolic_frame}")
        print(f"    Systolic: {systolic_frame}")
    else:
        diastolic_frame, systolic_frame = find_cardiac_phases(
            volume_4d,
            trigger_times=metadata.get('trigger_times', None)
        )

    # Extract diastolic and systolic volumes
    diastolic_volume = volume_4d[diastolic_frame]  # (n_slices, height, width)
    systolic_volume = volume_4d[systolic_frame]

    # Plot phase comparison
    print("\nGenerating phase comparison plot...")
    plot_phase_comparison(diastolic_volume, systolic_volume,
                         diastolic_frame, systolic_frame, output_dir)

    # Segment diastolic phase
    # Slices 3-14 contain LV cavity in diastole (from PDF)
    diastolic_slices = range(3, min(14, metadata['n_slices']))

    diastolic_masks = segment_phase(
        diastolic_volume,
        'Diastolic',
        diastolic_slices,
        seed_centers={},  # Empty: will use default center
        seed_radius=args.seed_radius,
        n_iterations=args.n_iterations,
        smoothing=args.smoothing,
        verbose=True
    )

    # Segment systolic phase
    # Slices 4-13 contain LV cavity in systole (from PDF)
    systolic_slices = range(4, min(13, metadata['n_slices']))

    systolic_masks = segment_phase(
        systolic_volume,
        'Systolic',
        systolic_slices,
        seed_centers={},
        seed_radius=args.seed_radius,
        n_iterations=args.n_iterations,
        smoothing=args.smoothing,
        verbose=True
    )

    # Plot segmentation results
    print("\nGenerating segmentation plots...")
    plot_segmentation_results(diastolic_volume, diastolic_masks,
                             'Diastolic', diastolic_frame, output_dir)
    plot_segmentation_results(systolic_volume, systolic_masks,
                             'Systolic', systolic_frame, output_dir)

    # Compute volumes
    print("\nComputing ventricular volumes...")
    pixel_spacing = metadata['pixel_spacing']
    slice_thickness = metadata['slice_thickness']

    edlv = compute_volume_from_masks(diastolic_masks, pixel_spacing, slice_thickness)
    eslv = compute_volume_from_masks(systolic_masks, pixel_spacing, slice_thickness)

    print(f"  EDLV: {edlv:.1f} mL")
    print(f"  ESLV: {eslv:.1f} mL")

    # Plot volume charts
    plot_volume_curves(edlv, eslv, output_dir)

    # Generate report
    print("\nGenerating cardiac function report...")

    diastolic_time = metadata['trigger_times'][diastolic_frame] if metadata['trigger_times'] else None
    systolic_time = metadata['trigger_times'][systolic_frame] if metadata['trigger_times'] else None

    report = generate_cardiac_report(
        edlv,
        eslv,
        args.weight,
        args.height,
        args.heart_rate,
        diastolic_frame,
        systolic_frame,
        diastolic_time,
        systolic_time
    )

    print(report)

    # Save report to file
    report_path = output_dir / "cardiac_function_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to {report_path}")

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED TO {output_dir}")
    print("Files created:")
    print("  - cardiac_4d_overview.png: 4D dataset overview")
    print("  - cardiac_phases_comparison.png: Diastole vs systole")
    print("  - segmentation_diastolic.png: Diastolic segmentation")
    print("  - segmentation_systolic.png: Systolic segmentation")
    print("  - cardiac_volumes.png: Volume and EF charts")
    print("  - cardiac_function_report.txt: Complete report")

    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
