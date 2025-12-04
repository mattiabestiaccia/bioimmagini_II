
import numpy as np
import pytest
from src.utils import (
    dice_coefficient,
    remove_small_regions,
    keep_largest_component,
    crop_to_roi
)
from src.exceptions import ShapeMismatchError, ValidationError

def test_dice_coefficient():
    """Test DICE coefficient calculation."""
    # Perfect overlap
    mask1 = np.ones((10, 10), dtype=bool)
    mask2 = np.ones((10, 10), dtype=bool)
    assert dice_coefficient(mask1, mask2) == 1.0
    
    # No overlap
    mask1 = np.zeros((10, 10), dtype=bool)
    mask1[0:5, :] = True
    mask2 = np.zeros((10, 10), dtype=bool)
    mask2[5:, :] = True
    assert dice_coefficient(mask1, mask2) == 0.0
    
    # Partial overlap (50%)
    # mask1: 100 pixels
    # mask2: 100 pixels
    # intersection: 50 pixels
    # DICE = 2 * 50 / (100 + 100) = 0.5
    mask1 = np.ones((10, 10), dtype=bool)
    mask2 = np.zeros((10, 10), dtype=bool)
    mask2[0:5, :] = True
    assert dice_coefficient(mask1, mask2) == 2 * 50 / (100 + 50)

def test_dice_coefficient_mismatch():
    """Test error on shape mismatch."""
    mask1 = np.ones((10, 10))
    mask2 = np.ones((11, 10))
    with pytest.raises(ShapeMismatchError):
        dice_coefficient(mask1, mask2)

def test_remove_small_regions():
    """Test removal of small connected components."""
    mask = np.zeros((10, 10), dtype=bool)
    
    # Region 1: 4 pixels
    mask[0:2, 0:2] = True
    
    # Region 2: 9 pixels
    mask[5:8, 5:8] = True
    
    # Remove < 5 pixels
    cleaned = remove_small_regions(mask, min_size=5)
    
    assert not cleaned[0:2, 0:2].any()  # Removed
    assert cleaned[5:8, 5:8].all()      # Kept

def test_keep_largest_component():
    """Test keeping only largest component."""
    mask = np.zeros((10, 10), dtype=bool)
    
    # Region 1: 4 pixels
    mask[0:2, 0:2] = True
    
    # Region 2: 9 pixels
    mask[5:8, 5:8] = True
    
    largest = keep_largest_component(mask)
    
    assert not largest[0:2, 0:2].any()  # Removed
    assert largest[5:8, 5:8].all()      # Kept

def test_crop_to_roi():
    """Test ROI cropping."""
    # 2D image
    img2d = np.zeros((100, 100))
    roi = (10, 20, 30, 45) # rows 10-20, cols 30-45
    cropped = crop_to_roi(img2d, roi)
    assert cropped.shape == (10, 15)
    
    # 3D image
    img3d = np.zeros((100, 100, 50))
    cropped3d = crop_to_roi(img3d, roi)
    assert cropped3d.shape == (10, 15, 50)
    
    # Invalid dim
    img1d = np.zeros((100,))
    with pytest.raises(ValidationError):
        crop_to_roi(img1d, roi)
