"""
Cardiac MRI Slice Classification with CNN.

This package implements deep learning-based automatic classification
of short-axis cardiac MRI slices according to AHA anatomical position.
"""

__version__ = '1.0.0'
__author__ = 'Bioimmagini Positano'

from .utils import (
    load_and_preprocess_dicom,
    load_dataset,
    create_data_splits,
    build_cnn_model,
    compile_model,
    calculate_metrics,
    plot_confusion_matrix,
    plot_training_history,
    visualize_misclassified,
    print_classification_report
)

__all__ = [
    'load_and_preprocess_dicom',
    'load_dataset',
    'create_data_splits',
    'build_cnn_model',
    'compile_model',
    'calculate_metrics',
    'plot_confusion_matrix',
    'plot_training_history',
    'visualize_misclassified',
    'print_classification_report'
]
