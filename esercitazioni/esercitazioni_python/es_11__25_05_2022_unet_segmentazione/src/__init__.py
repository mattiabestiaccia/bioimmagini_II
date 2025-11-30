"""
U-Net Brain MRI Segmentation.

This package implements U-Net deep learning for semantic segmentation
of brain structures from MRI images (BrainWeb dataset).
"""

__version__ = '1.0.0'
__author__ = 'Bioimmagini Positano'

from .utils import (
    load_image_and_mask,
    load_dataset,
    build_unet,
    compile_unet,
    dice_coefficient,
    dice_loss,
    combined_loss,
    calculate_segmentation_metrics,
    visualize_segmentation,
    plot_training_history,
    print_metrics_report
)

__all__ = [
    'load_image_and_mask',
    'load_dataset',
    'build_unet',
    'compile_unet',
    'dice_coefficient',
    'dice_loss',
    'combined_loss',
    'calculate_segmentation_metrics',
    'visualize_segmentation',
    'plot_training_history',
    'print_metrics_report'
]
