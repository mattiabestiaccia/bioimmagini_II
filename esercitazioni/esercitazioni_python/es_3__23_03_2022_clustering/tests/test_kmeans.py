
import numpy as np
import pytest
from src.kmeans_segmentation import perform_kmeans_clustering, postprocess_masks
from src.utils import identify_tissue_clusters

def test_perform_kmeans_clustering_shapes():
    """Test output shapes of clustering."""
    # Create synthetic data: 10x10 image with 20 frames
    height, width, n_frames = 10, 10, 20
    image_stack = np.random.rand(height, width, n_frames).astype(np.float32)
    
    n_clusters = 4
    labels, centroids = perform_kmeans_clustering(
        image_stack, 
        n_clusters=n_clusters,
        n_frames=None,
        metric="euclidean",
        n_init=2,
        random_state=42
    )
    
    assert labels.shape == (height, width)
    assert centroids.shape == (n_clusters, n_frames)
    assert len(np.unique(labels)) <= n_clusters

def test_perform_kmeans_clustering_correlation():
    """Test clustering with correlation metric."""
    height, width, n_frames = 10, 10, 20
    image_stack = np.random.rand(height, width, n_frames).astype(np.float32)
    
    # Ensure no constant curves to avoid division by zero in correlation
    image_stack += np.random.rand(height, width, n_frames) * 0.1
    
    labels, centroids = perform_kmeans_clustering(
        image_stack, 
        n_clusters=4,
        metric="correlation",
        n_init=2
    )
    
    assert labels.shape == (height, width)
    # Centroids should be de-normalized, so we just check shape
    assert centroids.shape == (4, n_frames)

def test_identify_tissue_clusters():
    """Test tissue identification logic."""
    n_frames = 50
    n_clusters = 4
    
    # Create synthetic centroids for 4 tissues
    centroids = np.zeros((n_clusters, n_frames))
    
    # 0: Background (low constant)
    centroids[0, :] = 10
    
    # 1: RV (early peak)
    t = np.arange(n_frames)
    centroids[1, :] = 100 * np.exp(-(t - 10)**2 / 20) + 20
    
    # 2: LV (mid peak)
    centroids[2, :] = 90 * np.exp(-(t - 20)**2 / 20) + 20
    
    # 3: Myo (late peak, lower intensity)
    centroids[3, :] = 60 * np.exp(-(t - 30)**2 / 30) + 20
    
    # Random labels
    labels = np.zeros((10, 10), dtype=int)
    
    tissue_map = identify_tissue_clusters(labels, centroids, n_clusters)
    
    # Check mapping
    # Background should be index 0 (lowest contrast)
    assert tissue_map["background"] == 0
    
    # RV should be index 1 (earliest peak)
    assert tissue_map["rv"] == 1
    
    # LV should be index 2 (higher peak than Myo among remaining)
    assert tissue_map["lv"] == 2
    
    # Myo should be index 3
    assert tissue_map["myo"] == 3

def test_postprocess_masks():
    """Test mask post-processing."""
    # Create a mask with a large region and a small region
    mask = np.zeros((20, 20), dtype=bool)
    
    # Large region (10x10 = 100 pixels)
    mask[0:10, 0:10] = True
    
    # Small region (2x2 = 4 pixels)
    mask[15:17, 15:17] = True
    
    masks = {"test": mask}
    
    # Should remove small region (min_size=10)
    cleaned = postprocess_masks(masks, min_size=10, keep_largest=False)
    assert cleaned["test"][0:10, 0:10].all()
    assert not cleaned["test"][15:17, 15:17].any()
    
    # Should keep only largest
    cleaned_largest = postprocess_masks(masks, min_size=0, keep_largest=True)
    assert cleaned_largest["test"][0:10, 0:10].all()
    assert not cleaned_largest["test"][15:17, 15:17].any()
