#!/usr/bin/env python3
"""
temporal_registration.py - Temporal Series Registration with Demons Algorithm

Pipeline for registering temporal MRI series with respiratory motion using:
1. Hierarchical clustering to group similar images
2. Demons algorithm for non-rigid registration within and between clusters
3. Perfusion curve extraction before and after registration

Esercitazione 8: Registrazione Serie Temporali
Dataset: Renal perfusion MRI (70 frames, 2D+T)

Author: Biomedical Imaging Course
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.cluster.hierarchy import dendrogram
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_dicom_series,
    normalize_image,
    compute_distance_matrix,
    hierarchical_clustering,
    demons_registration,
    multi_scale_demons,
    apply_displacement_to_series,
    extract_perfusion_curve
)


def select_subset(images: np.ndarray, n_subset: int = 20, strategy: str = 'uniform') -> np.ndarray:
    """
    Select a subset of images from temporal series.

    Parameters
    ----------
    images : np.ndarray
        Full temporal series (n_frames, height, width)
    n_subset : int, optional
        Number of images to select (default: 20, as in PDF)
    strategy : str, optional
        Selection strategy: 'uniform' (evenly spaced) or 'random'

    Returns
    -------
    indices : np.ndarray
        Indices of selected images
    """
    n_frames = images.shape[0]

    if n_subset >= n_frames:
        return np.arange(n_frames)

    if strategy == 'uniform':
        # Evenly spaced indices
        indices = np.linspace(0, n_frames - 1, n_subset, dtype=int)
    elif strategy == 'random':
        # Random selection
        indices = np.sort(np.random.choice(n_frames, n_subset, replace=False))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return indices


def plot_dendrogram(linkage_matrix: np.ndarray, output_dir: Path):
    """
    Plot hierarchical clustering dendrogram.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering
    output_dir : Path
        Directory to save plot
    """
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=14)
    plt.xlabel('Image Index', fontsize=12)
    plt.ylabel('Distance (MSE)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "dendrogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved dendrogram to {output_path}")


def plot_cluster_assignment(images: np.ndarray, labels: np.ndarray, output_dir: Path):
    """
    Plot cluster assignments with sample images.

    Parameters
    ----------
    images : np.ndarray
        Images (n_images, height, width)
    labels : np.ndarray
        Cluster labels (n_images,)
    output_dir : Path
        Directory to save plot
    """
    n_clusters = len(np.unique(labels))

    fig, axes = plt.subplots(n_clusters, 5, figsize=(15, n_clusters * 3))

    if n_clusters == 1:
        axes = axes.reshape(1, -1)

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        n_in_cluster = len(cluster_indices)

        # Select up to 5 representative images
        n_show = min(5, n_in_cluster)
        show_indices = cluster_indices[:n_show]

        for col in range(5):
            ax = axes[cluster_id, col]

            if col < n_show:
                img_idx = show_indices[col]
                ax.imshow(images[img_idx], cmap='gray')
                ax.set_title(f'Cluster {cluster_id}: Image {img_idx}', fontsize=10)
            else:
                ax.axis('off')

            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('Cluster Assignments', fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = output_dir / "cluster_assignment.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved cluster assignment to {output_path}")


def plot_registration_comparison(moving: np.ndarray,
                                fixed: np.ndarray,
                                registered: np.ndarray,
                                title: str,
                                output_dir: Path):
    """
    Plot registration results: moving, fixed, registered, difference.

    Parameters
    ----------
    moving : np.ndarray
        Moving image before registration
    fixed : np.ndarray
        Fixed (reference) image
    registered : np.ndarray
        Registered moving image
    title : str
        Plot title
    output_dir : Path
        Directory to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Normalize for display
    moving_norm = normalize_image(moving)
    fixed_norm = normalize_image(fixed)
    registered_norm = normalize_image(registered)

    # Row 1: Images
    axes[0, 0].imshow(moving_norm, cmap='gray')
    axes[0, 0].set_title('Moving Image', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(fixed_norm, cmap='gray')
    axes[0, 1].set_title('Fixed Image', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(registered_norm, cmap='gray')
    axes[0, 2].set_title('Registered Image', fontsize=12)
    axes[0, 2].axis('off')

    # Row 2: Differences
    diff_before = np.abs(moving_norm - fixed_norm)
    diff_after = np.abs(registered_norm - fixed_norm)

    im1 = axes[1, 0].imshow(diff_before, cmap='hot')
    axes[1, 0].set_title(f'Difference Before\nMSE={np.mean(diff_before**2):.4f}', fontsize=11)
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im2 = axes[1, 1].imshow(diff_after, cmap='hot')
    axes[1, 1].set_title(f'Difference After\nMSE={np.mean(diff_after**2):.4f}', fontsize=11)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Checkerboard overlay
    checkerboard = create_checkerboard_overlay(fixed_norm, registered_norm)
    axes[1, 2].imshow(checkerboard, cmap='gray')
    axes[1, 2].set_title('Checkerboard Overlay', fontsize=12)
    axes[1, 2].axis('off')

    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()

    # Save with sanitized filename
    filename = title.replace(' ', '_').replace(':', '').replace('/', '_') + ".png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved registration comparison to {output_path}")


def create_checkerboard_overlay(image1: np.ndarray, image2: np.ndarray, square_size: int = 20) -> np.ndarray:
    """
    Create checkerboard overlay of two images.

    Parameters
    ----------
    image1 : np.ndarray
        First image
    image2 : np.ndarray
        Second image
    square_size : int, optional
        Size of checkerboard squares in pixels (default: 20)

    Returns
    -------
    checkerboard : np.ndarray
        Checkerboard overlay
    """
    height, width = image1.shape
    checkerboard = np.zeros_like(image1)

    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                checkerboard[i:i+square_size, j:j+square_size] = image1[i:i+square_size, j:j+square_size]
            else:
                checkerboard[i:i+square_size, j:j+square_size] = image2[i:i+square_size, j:j+square_size]

    return checkerboard


def plot_perfusion_curves(curves_dict: dict, output_dir: Path):
    """
    Plot perfusion curves before and after registration.

    Parameters
    ----------
    curves_dict : dict
        Dictionary with keys 'before' and 'after', values are curves
    output_dir : Path
        Directory to save plot
    """
    plt.figure(figsize=(12, 6))

    for label, curve in curves_dict.items():
        plt.plot(curve, marker='o', markersize=4, label=label, linewidth=2)

    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Mean Intensity', fontsize=12)
    plt.title('Perfusion Curves: Before vs After Registration', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "perfusion_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved perfusion curves to {output_path}")


def register_temporal_series(images: np.ndarray,
                            n_clusters: int = 2,
                            use_multiscale: bool = True,
                            n_iterations: int = 50,
                            alpha: float = 2.5,
                            sigma_diffusion: float = 1.0,
                            output_dir: Path = None,
                            verbose: bool = True) -> np.ndarray:
    """
    Register temporal series using hierarchical clustering and Demons algorithm.

    Pipeline:
    1. Compute distance matrix between all images (MSE metric)
    2. Hierarchical clustering to group similar images (pre/post contrast)
    3. Select reference image within each cluster (median)
    4. Register images within each cluster to cluster reference
    5. Register cluster references to each other
    6. Apply combined displacement fields to all images

    Parameters
    ----------
    images : np.ndarray
        Temporal series (n_frames, height, width)
    n_clusters : int, optional
        Number of clusters (default: 2 for pre/post contrast)
    use_multiscale : bool, optional
        Use multi-scale pyramid approach (default: True)
    n_iterations : int, optional
        Number of Demons iterations (default: 50)
    alpha : float, optional
        Demons regularization parameter (default: 2.5)
    sigma_diffusion : float, optional
        Gaussian smoothing sigma for displacement field (default: 1.0)
    output_dir : Path, optional
        Directory to save visualizations
    verbose : bool, optional
        Print progress information (default: True)

    Returns
    -------
    registered_series : np.ndarray
        Registered temporal series (n_frames, height, width)
    """
    n_frames = images.shape[0]

    if verbose:
        print(f"\n=== TEMPORAL SERIES REGISTRATION ===")
        print(f"Number of frames: {n_frames}")
        print(f"Image shape: {images.shape[1]} x {images.shape[2]}")

    # Step 1: Compute distance matrix
    if verbose:
        print(f"\nStep 1: Computing distance matrix (MSE metric)...")

    distance_matrix = compute_distance_matrix(images, metric='mse')

    # Step 2: Hierarchical clustering
    if verbose:
        print(f"\nStep 2: Hierarchical clustering (n_clusters={n_clusters})...")

    clustering_results = hierarchical_clustering(distance_matrix, n_clusters=n_clusters, method='average')
    labels = clustering_results['labels']
    linkage_matrix = clustering_results['linkage_matrix']

    if output_dir is not None:
        plot_dendrogram(linkage_matrix, output_dir)
        plot_cluster_assignment(images, labels, output_dir)

    # Print cluster statistics
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        print(f"  Cluster {cluster_id}: {len(cluster_indices)} images (indices: {cluster_indices[:5]}...)")

    # Step 3: Select reference images for each cluster
    if verbose:
        print(f"\nStep 3: Selecting reference images for each cluster...")

    cluster_references = {}
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]

        # Select median image as reference (most representative)
        median_idx = cluster_indices[len(cluster_indices) // 2]
        cluster_references[cluster_id] = median_idx

        print(f"  Cluster {cluster_id} reference: Image {median_idx}")

    # Step 4: Register within clusters
    if verbose:
        print(f"\nStep 4: Registering images within each cluster...")

    within_cluster_displacements = {}

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        reference_idx = cluster_references[cluster_id]
        reference_image = images[reference_idx]

        print(f"\n  Cluster {cluster_id}:")

        for idx in cluster_indices:
            if idx == reference_idx:
                # Reference image: zero displacement
                within_cluster_displacements[idx] = np.zeros((2, images.shape[1], images.shape[2]), dtype=np.float32)
            else:
                # Register to cluster reference
                moving_image = images[idx]

                if use_multiscale:
                    displacement, registered = multi_scale_demons(
                        moving_image,
                        reference_image,
                        scales=[4, 2, 1],
                        n_iterations=n_iterations,
                        alpha=alpha,
                        sigma_diffusion=sigma_diffusion,
                        verbose=False
                    )
                else:
                    displacement, registered = demons_registration(
                        moving_image,
                        reference_image,
                        n_iterations=n_iterations,
                        alpha=alpha,
                        sigma_diffusion=sigma_diffusion,
                        verbose=False
                    )

                within_cluster_displacements[idx] = displacement

                # Save example registration
                if output_dir is not None and idx == cluster_indices[0]:
                    plot_registration_comparison(
                        moving_image,
                        reference_image,
                        registered,
                        f"Within Cluster {cluster_id} Registration",
                        output_dir
                    )

    # Step 5: Register between clusters
    if verbose:
        print(f"\nStep 5: Registering between clusters...")

    if n_clusters > 1:
        # Use cluster 0 as global reference
        global_reference_idx = cluster_references[0]
        global_reference_image = images[global_reference_idx]

        between_cluster_displacements = {}
        between_cluster_displacements[0] = np.zeros((2, images.shape[1], images.shape[2]), dtype=np.float32)

        for cluster_id in range(1, n_clusters):
            reference_idx = cluster_references[cluster_id]
            moving_image = images[reference_idx]

            print(f"\n  Registering Cluster {cluster_id} reference to Cluster 0 reference...")

            if use_multiscale:
                displacement, registered = multi_scale_demons(
                    moving_image,
                    global_reference_image,
                    scales=[4, 2, 1],
                    n_iterations=n_iterations,
                    alpha=alpha,
                    sigma_diffusion=sigma_diffusion,
                    verbose=False
                )
            else:
                displacement, registered = demons_registration(
                    moving_image,
                    global_reference_image,
                    n_iterations=n_iterations,
                    alpha=alpha,
                    sigma_diffusion=sigma_diffusion,
                    verbose=False
                )

            between_cluster_displacements[cluster_id] = displacement

            if output_dir is not None:
                plot_registration_comparison(
                    moving_image,
                    global_reference_image,
                    registered,
                    f"Between Clusters {cluster_id} to 0 Registration",
                    output_dir
                )
    else:
        between_cluster_displacements = {0: np.zeros((2, images.shape[1], images.shape[2]), dtype=np.float32)}

    # Step 6: Combine displacements and apply to all images
    if verbose:
        print(f"\nStep 6: Applying combined displacement fields...")

    registered_series = np.zeros_like(images)

    for idx in range(n_frames):
        cluster_id = labels[idx]

        # Combined displacement: within-cluster + between-cluster
        disp_within = within_cluster_displacements[idx]
        disp_between = between_cluster_displacements[cluster_id]

        # Simple composition: add displacements (approximation)
        # More accurate would be to compose transformations
        combined_displacement = disp_within + disp_between

        # Apply to image
        from utils import warp_image
        registered_series[idx] = warp_image(images[idx], combined_displacement)

    if verbose:
        print(f"\nRegistration complete!")

    return registered_series


def main():
    """
    Main function for temporal series registration.
    """
    parser = argparse.ArgumentParser(
        description='Temporal Series Registration with Demons Algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data/RENAL_PERF',
        help='Directory containing DICOM temporal series'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results',
        help='Directory to save results'
    )

    parser.add_argument(
        '--n_subset',
        type=int,
        default=20,
        help='Number of images to use (subset of full series, 0 = all)'
    )

    parser.add_argument(
        '--n_clusters',
        type=int,
        default=2,
        help='Number of clusters for hierarchical clustering'
    )

    parser.add_argument(
        '--n_iterations',
        type=int,
        default=50,
        help='Number of Demons iterations per scale'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=2.5,
        help='Demons regularization parameter'
    )

    parser.add_argument(
        '--sigma_diffusion',
        type=float,
        default=1.0,
        help='Gaussian smoothing sigma for displacement field'
    )

    parser.add_argument(
        '--no_multiscale',
        action='store_true',
        help='Disable multi-scale pyramid approach'
    )

    parser.add_argument(
        '--roi',
        type=int,
        nargs=4,
        default=None,
        metavar=('Y_MIN', 'Y_MAX', 'X_MIN', 'X_MAX'),
        help='ROI for perfusion curve extraction (y_min y_max x_min x_max)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TEMPORAL SERIES REGISTRATION WITH DEMONS ALGORITHM")
    print("=" * 70)

    # Load DICOM series
    print(f"\nLoading DICOM series from {args.data_dir}...")
    images, datasets = load_dicom_series(args.data_dir)

    # Select subset if requested
    if args.n_subset > 0 and args.n_subset < images.shape[0]:
        print(f"\nSelecting subset of {args.n_subset} images (uniformly spaced)...")
        subset_indices = select_subset(images, n_subset=args.n_subset, strategy='uniform')
        images_subset = images[subset_indices]
        print(f"Selected indices: {subset_indices}")
    else:
        images_subset = images
        subset_indices = np.arange(images.shape[0])

    # Extract perfusion curve before registration
    print(f"\nExtracting perfusion curve before registration...")
    if args.roi is not None:
        roi_coords = tuple(args.roi)
        print(f"  Using ROI: y=[{roi_coords[0]}, {roi_coords[1]}], x=[{roi_coords[2]}, {roi_coords[3]}]")
        curve_before = extract_perfusion_curve(images_subset, roi_coords=roi_coords)
    else:
        print(f"  Using whole image (no ROI specified)")
        curve_before = extract_perfusion_curve(images_subset)

    # Register temporal series
    registered_series = register_temporal_series(
        images_subset,
        n_clusters=args.n_clusters,
        use_multiscale=not args.no_multiscale,
        n_iterations=args.n_iterations,
        alpha=args.alpha,
        sigma_diffusion=args.sigma_diffusion,
        output_dir=output_dir,
        verbose=True
    )

    # Extract perfusion curve after registration
    print(f"\nExtracting perfusion curve after registration...")
    if args.roi is not None:
        curve_after = extract_perfusion_curve(registered_series, roi_coords=roi_coords)
    else:
        curve_after = extract_perfusion_curve(registered_series)

    # Plot perfusion curves
    plot_perfusion_curves(
        {'Before Registration': curve_before, 'After Registration': curve_after},
        output_dir
    )

    # Compute and display statistics
    print(f"\n=== PERFUSION CURVE STATISTICS ===")
    print(f"Before registration:")
    print(f"  Mean: {np.mean(curve_before):.2f}, Std: {np.std(curve_before):.2f}")
    print(f"  Range: [{np.min(curve_before):.2f}, {np.max(curve_before):.2f}]")
    print(f"\nAfter registration:")
    print(f"  Mean: {np.mean(curve_after):.2f}, Std: {np.std(curve_after):.2f}")
    print(f"  Range: [{np.min(curve_after):.2f}, {np.max(curve_after):.2f}]")

    # Compute curve smoothness (variance of first derivative)
    diff_before = np.diff(curve_before)
    diff_after = np.diff(curve_after)
    smoothness_before = np.var(diff_before)
    smoothness_after = np.var(diff_after)

    print(f"\nCurve smoothness (variance of derivative):")
    print(f"  Before: {smoothness_before:.4f}")
    print(f"  After: {smoothness_after:.4f}")
    print(f"  Improvement: {(smoothness_before - smoothness_after) / smoothness_before * 100:.1f}%")

    print(f"\n=== RESULTS SAVED TO {output_dir} ===")
    print("Files created:")
    print("  - dendrogram.png: Hierarchical clustering dendrogram")
    print("  - cluster_assignment.png: Cluster assignments with sample images")
    print("  - Within_Cluster_*_Registration.png: Within-cluster registration examples")
    print("  - Between_Clusters_*_Registration.png: Between-cluster registration examples")
    print("  - perfusion_curves.png: Perfusion curves before/after registration")

    print("\nRegistration pipeline completed successfully!")


if __name__ == "__main__":
    main()
