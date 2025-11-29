"""
Utility functions for CNN-based cardiac MRI slice classification.

This module implements:
- DICOM image loading and preprocessing
- Data augmentation
- CNN model architecture (VGG-style)
- Training and evaluation utilities
- Performance metrics calculation
"""

import numpy as np
import pydicom
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import cv2


def load_and_preprocess_dicom(
    dicom_path: str,
    target_size: Tuple[int, int] = (128, 128),
    crop_center: bool = True,
    crop_size: Optional[int] = None
) -> np.ndarray:
    """
    Load and preprocess a single DICOM image.

    Pipeline:
    1. Load DICOM with pydicom
    2. Center crop to focus on heart region
    3. Resize to target_size
    4. Normalize to [0, 1]

    Parameters
    ----------
    dicom_path : str
        Path to DICOM file
    target_size : tuple
        Target image size (height, width)
    crop_center : bool
        Whether to crop center region
    crop_size : int, optional
        Size of center crop. If None, uses min(height, width) * 0.7

    Returns
    -------
    image : ndarray
        Preprocessed image of shape (height, width, 1)
    """
    # Load DICOM
    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array.astype(np.float32)

    # Center crop to focus on heart
    if crop_center:
        h, w = image.shape
        if crop_size is None:
            crop_size = int(min(h, w) * 0.7)

        center_y, center_x = h // 2, w // 2
        half_crop = crop_size // 2

        y_start = max(0, center_y - half_crop)
        y_end = min(h, center_y + half_crop)
        x_start = max(0, center_x - half_crop)
        x_end = min(w, center_x + half_crop)

        image = image[y_start:y_end, x_start:x_end]

    # Resize to target size
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1]
    image_min = image.min()
    image_max = image.max()
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    else:
        image = np.zeros_like(image)

    # Add channel dimension
    image = np.expand_dims(image, axis=-1)

    return image


def load_dataset(
    data_dir: str,
    target_size: Tuple[int, int] = (128, 128),
    classes: List[str] = ['Apical', 'Basal', 'Middle'],
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load complete dataset from directory structure.

    Expected structure:
    data_dir/
        Apical/*.dcm
        Basal/*.dcm
        Middle/*.dcm

    Parameters
    ----------
    data_dir : str
        Path to data directory
    target_size : tuple
        Target image size
    classes : list
        List of class names (directory names)
    verbose : bool
        Print progress

    Returns
    -------
    X : ndarray
        Images array of shape (n_samples, height, width, 1)
    y : ndarray
        Labels array of shape (n_samples,)
    class_names : list
        List of class names
    """
    data_path = Path(data_dir)
    images = []
    labels = []

    for class_idx, class_name in enumerate(classes):
        class_dir = data_path / class_name
        if not class_dir.exists():
            raise ValueError(f"Class directory not found: {class_dir}")

        dicom_files = sorted(class_dir.glob('*.dcm'))

        if verbose:
            print(f"Loading {class_name}: {len(dicom_files)} images")

        for dicom_file in dicom_files:
            try:
                image = load_and_preprocess_dicom(str(dicom_file), target_size)
                images.append(image)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {dicom_file}: {e}")
                continue

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    if verbose:
        print(f"\nDataset loaded: {X.shape[0]} images")
        print(f"Image shape: {X.shape[1:]}")
        print(f"Class distribution: {np.bincount(y)}")

    return X, y, classes


def create_data_splits(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train/validation/test sets.

    Parameters
    ----------
    X : ndarray
        Images
    y : ndarray
        Labels
    train_ratio : float
        Ratio for training set
    val_ratio : float
        Ratio for validation set
    test_ratio : float
        Ratio for test set
    random_state : int
        Random seed

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : ndarrays
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # First split: train vs (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_ratio + test_ratio),
        stratify=y,
        random_state=random_state
    )

    # Second split: val vs test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_test_ratio),
        stratify=y_temp,
        random_state=random_state
    )

    print(f"Train set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_cnn_model(
    input_shape: Tuple[int, int, int],
    num_classes: int = 3,
    architecture: str = 'vgg_small'
) -> keras.Model:
    """
    Build CNN model for slice classification.

    Architecture follows VGG-style:
    INPUT -> [[CONV -> RELU]*N -> POOL]*M -> [FC -> RELU]*K -> FC

    Parameters
    ----------
    input_shape : tuple
        Input image shape (height, width, channels)
    num_classes : int
        Number of output classes
    architecture : str
        Model architecture: 'vgg_small', 'vgg_medium', 'simple'

    Returns
    -------
    model : keras.Model
    """
    model = models.Sequential(name=f'CNN_{architecture}')

    if architecture == 'simple':
        # Simple baseline: 2 conv blocks + FC
        model.add(layers.Input(shape=input_shape))

        # Block 1
        model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Block 2
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Classifier
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))

    elif architecture == 'vgg_small':
        # VGG-style small: 3 blocks
        model.add(layers.Input(shape=input_shape))

        # Block 1: 128x128 -> 64x64
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())

        # Block 2: 64x64 -> 32x32
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())

        # Block 3: 32x32 -> 16x16
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())

        # Classifier
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))

    elif architecture == 'vgg_medium':
        # VGG-style medium: 4 blocks, deeper
        model.add(layers.Input(shape=input_shape))

        # Block 1
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Block 2
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Block 3
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Block 4
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Classifier
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    optimizer: str = 'adam'
) -> keras.Model:
    """
    Compile CNN model.

    Parameters
    ----------
    model : keras.Model
        Model to compile
    learning_rate : float
        Learning rate
    optimizer : str
        Optimizer name: 'adam', 'sgd', 'rmsprop'

    Returns
    -------
    model : keras.Model
        Compiled model
    """
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate sensitivity, specificity, and accuracy for each class.

    For each class i:
    - TP = class i correctly classified as i
    - TN = non-class i correctly classified as non-i
    - FP = non-class i incorrectly classified as i
    - FN = class i incorrectly classified as non-i

    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Parameters
    ----------
    y_true : ndarray
        True labels (class indices)
    y_pred : ndarray
        Predicted labels (class indices)
    class_names : list
        Class names

    Returns
    -------
    metrics : dict
        Metrics for each class
    """
    num_classes = len(class_names)
    metrics = {}

    for class_idx, class_name in enumerate(class_names):
        # Binary classification: class_i vs rest
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)

        TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        metrics[class_name] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN)
        }

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot confusion matrix with annotations.

    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    class_names : list
        Class names
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with counts and percentages
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_normalized[i, j] * 100
            text_color = "white" if cm_normalized[i, j] > thresh else "black"
            ax.text(j, i, f'{count}\n({percentage:.1f}%)',
                   ha="center", va="center", color=text_color, fontsize=12)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_training_history(
    history: keras.callbacks.History,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot training history (loss and accuracy).

    Parameters
    ----------
    history : keras.callbacks.History
        Training history
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")

    plt.show()


def visualize_misclassified(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    num_samples: int = 9,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Visualize misclassified images.

    Parameters
    ----------
    X : ndarray
        Images
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    class_names : list
        Class names
    num_samples : int
        Number of misclassified samples to show
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    # Find misclassified indices
    misclassified_idx = np.where(y_true != y_pred)[0]

    if len(misclassified_idx) == 0:
        print("No misclassified images!")
        return

    print(f"Total misclassified: {len(misclassified_idx)}")

    # Sample random misclassified
    num_show = min(num_samples, len(misclassified_idx))
    show_idx = np.random.choice(misclassified_idx, num_show, replace=False)

    # Plot
    n_cols = 3
    n_rows = (num_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if num_show > 1 else [axes]

    for i, idx in enumerate(show_idx):
        ax = axes[i]

        # Show image (squeeze channel dimension if grayscale)
        img = X[idx].squeeze()
        ax.imshow(img, cmap='gray')

        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]

        title = f'True: {true_label}\nPred: {pred_label}'
        ax.set_title(title, fontsize=12, color='red')
        ax.axis('off')

    # Hide unused subplots
    for i in range(num_show, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Misclassified Images', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Misclassified visualization saved to {save_path}")

    plt.show()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    dataset_name: str = 'Test Set'
):
    """
    Print detailed classification report.

    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    class_names : list
        Class names
    dataset_name : str
        Name of dataset (for printing)
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} Performance Report")
    print(f"{'='*60}\n")

    # Overall accuracy
    overall_acc = np.mean(y_true == y_pred)
    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)\n")

    # Sklearn classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Custom metrics (sensitivity, specificity)
    metrics = calculate_metrics(y_true, y_pred, class_names)

    print("\nDetailed Metrics per Class:")
    print(f"{'Class':<12} {'Sens':<8} {'Spec':<8} {'Acc':<8} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}")
    print("-" * 60)

    for class_name in class_names:
        m = metrics[class_name]
        print(f"{class_name:<12} "
              f"{m['sensitivity']:.4f}   "
              f"{m['specificity']:.4f}   "
              f"{m['accuracy']:.4f}   "
              f"{m['TP']:>4} {m['TN']:>4} {m['FP']:>4} {m['FN']:>4}")

    print(f"{'='*60}\n")
