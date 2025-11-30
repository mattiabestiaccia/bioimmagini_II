"""
Utility functions for U-Net based brain MRI segmentation.

This module implements:
- U-Net 2D architecture (encoder-decoder with skip connections)
- Data loading from BrainWeb dataset
- DICE coefficient and segmentation metrics
- Visualization utilities
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def load_image_and_mask(
    image_path: str,
    mask_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MR image and corresponding segmentation mask.

    Parameters
    ----------
    image_path : str
        Path to MR image (PNG format)
    mask_path : str
        Path to mask image (PNG format)
    target_size : tuple, optional
        Target size for resizing (height, width)
    normalize : bool
        Normalize image to [0, 1]

    Returns
    -------
    image : ndarray
        Image array of shape (height, width, 1)
    mask : ndarray
        Binary mask of shape (height, width, 1)
    """
    # Load image (grayscale)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise ValueError(f"Could not load image or mask: {image_path}, {mask_path}")

    # Resize if needed
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # Normalize image
    if normalize:
        image = image.astype(np.float32) / 255.0

    # Binarize mask (0 or 1)
    mask = (mask > 127).astype(np.float32)

    # Add channel dimension
    image = np.expand_dims(image, axis=-1)
    mask = np.expand_dims(mask, axis=-1)

    return image, mask


def load_dataset(
    image_dir: str,
    mask_dir: str,
    target_size: Optional[Tuple[int, int]] = None,
    max_samples: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load complete dataset of MR images and masks.

    Parameters
    ----------
    image_dir : str
        Directory containing MR images
    mask_dir : str
        Directory containing masks
    target_size : tuple, optional
        Target image size (height, width)
    max_samples : int, optional
        Maximum number of samples to load
    verbose : bool
        Print progress

    Returns
    -------
    X : ndarray
        Images array of shape (n_samples, height, width, 1)
    y : ndarray
        Masks array of shape (n_samples, height, width, 1)
    """
    image_path = Path(image_dir)
    mask_path = Path(mask_dir)

    # Get sorted list of files
    image_files = sorted(image_path.glob('*.png'))
    mask_files = sorted(mask_path.glob('*.png'))

    if len(image_files) != len(mask_files):
        raise ValueError(f"Mismatch: {len(image_files)} images vs {len(mask_files)} masks")

    # Limit samples if requested
    if max_samples is not None:
        image_files = image_files[:max_samples]
        mask_files = mask_files[:max_samples]

    if verbose:
        print(f"Loading {len(image_files)} image-mask pairs...")

    images = []
    masks = []

    for img_file, mask_file in zip(image_files, mask_files):
        try:
            image, mask = load_image_and_mask(
                str(img_file),
                str(mask_file),
                target_size=target_size
            )
            images.append(image)
            masks.append(mask)
        except Exception as e:
            print(f"Error loading {img_file.name}: {e}")
            continue

    X = np.array(images, dtype=np.float32)
    y = np.array(masks, dtype=np.float32)

    if verbose:
        print(f"Dataset loaded: {X.shape[0]} samples")
        print(f"Image shape: {X.shape[1:]}")
        print(f"Mask shape: {y.shape[1:]}")
        print(f"Positive pixels: {np.sum(y > 0.5) / y.size * 100:.2f}%")

    return X, y


def build_unet(
    input_shape: Tuple[int, int, int],
    num_classes: int = 1,
    encoder_depth: int = 4,
    num_first_filters: int = 32,
    filter_size: int = 3,
    use_batch_norm: bool = True,
    dropout_rate: float = 0.2
) -> keras.Model:
    """
    Build U-Net model for semantic segmentation.

    U-Net architecture (Ronneberger et al. 2015):
    - Encoder (contracting path): Conv blocks + MaxPool
    - Decoder (expanding path): UpConv + Skip connections + Conv blocks
    - Symmetric architecture with skip connections

    Parameters
    ----------
    input_shape : tuple
        Input image shape (height, width, channels)
    num_classes : int
        Number of output classes (1 for binary segmentation)
    encoder_depth : int
        Number of encoder/decoder blocks (default: 4)
    num_first_filters : int
        Number of filters in first conv layer (doubles each block)
    filter_size : int
        Convolutional filter size (default: 3x3)
    use_batch_norm : bool
        Use batch normalization
    dropout_rate : float
        Dropout rate in bottleneck

    Returns
    -------
    model : keras.Model
        U-Net model
    """
    inputs = layers.Input(shape=input_shape, name='input')

    # Encoder (contracting path)
    encoder_blocks = []
    x = inputs

    for i in range(encoder_depth):
        num_filters = num_first_filters * (2 ** i)

        # Double convolution
        x = layers.Conv2D(num_filters, filter_size, activation='relu', padding='same',
                         kernel_initializer='he_normal')(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)

        x = layers.Conv2D(num_filters, filter_size, activation='relu', padding='same',
                         kernel_initializer='he_normal')(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)

        # Save for skip connection
        encoder_blocks.append(x)

        # MaxPooling (except for last block)
        if i < encoder_depth - 1:
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Bottleneck with dropout
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    # Decoder (expanding path)
    for i in range(encoder_depth - 2, -1, -1):
        num_filters = num_first_filters * (2 ** i)

        # UpConvolution (transpose convolution)
        x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(x)

        # Skip connection (concatenate with encoder block)
        x = layers.concatenate([x, encoder_blocks[i]], axis=-1)

        # Double convolution
        x = layers.Conv2D(num_filters, filter_size, activation='relu', padding='same',
                         kernel_initializer='he_normal')(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)

        x = layers.Conv2D(num_filters, filter_size, activation='relu', padding='same',
                         kernel_initializer='he_normal')(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)

    # Output layer
    if num_classes == 1:
        # Binary segmentation
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', name='output')(x)
    else:
        # Multi-class segmentation
        outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='U-Net')

    return model


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Compute DICE coefficient (F1 score for segmentation).

    DICE = 2 * |A ∩ B| / (|A| + |B|)
         = 2 * TP / (2*TP + FP + FN)

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth masks
    y_pred : tf.Tensor
        Predicted masks
    smooth : float
        Smoothing constant to avoid division by zero

    Returns
    -------
    dice : tf.Tensor
        DICE coefficient (range [0, 1])
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    return dice


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    DICE loss for training (1 - DICE coefficient).

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth masks
    y_pred : tf.Tensor
        Predicted masks

    Returns
    -------
    loss : tf.Tensor
        DICE loss
    """
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.5) -> tf.Tensor:
    """
    Combined Binary Cross-Entropy + DICE loss.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth masks
    y_pred : tf.Tensor
        Predicted masks
    alpha : float
        Weight for BCE (1-alpha for DICE)

    Returns
    -------
    loss : tf.Tensor
        Combined loss
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)

    return alpha * bce + (1 - alpha) * dice


def compile_unet(
    model: keras.Model,
    learning_rate: float = 0.001,
    optimizer: str = 'adam',
    loss: str = 'dice'
) -> keras.Model:
    """
    Compile U-Net model.

    Parameters
    ----------
    model : keras.Model
        U-Net model
    learning_rate : float
        Learning rate
    optimizer : str
        Optimizer: 'adam', 'sgd', 'rmsprop'
    loss : str
        Loss function: 'dice', 'bce', 'combined'

    Returns
    -------
    model : keras.Model
        Compiled model
    """
    # Optimizer
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # Loss function
    if loss == 'dice':
        loss_fn = dice_loss
    elif loss == 'bce':
        loss_fn = 'binary_crossentropy'
    elif loss == 'combined':
        loss_fn = combined_loss
    else:
        raise ValueError(f"Unknown loss: {loss}")

    # Compile
    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[dice_coefficient, 'accuracy']
    )

    return model


def calculate_segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate segmentation metrics.

    Parameters
    ----------
    y_true : ndarray
        Ground truth masks (binary)
    y_pred : ndarray
        Predicted masks (probabilities)
    threshold : float
        Threshold for binarizing predictions

    Returns
    -------
    metrics : dict
        Dictionary with metrics (dice, accuracy, iou, etc.)
    """
    # Binarize predictions
    y_pred_binary = (y_pred > threshold).astype(np.float32)

    # Flatten
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_binary.flatten()

    # True Positives, False Positives, False Negatives, True Negatives
    TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))

    # DICE coefficient
    dice = 2 * TP / (2 * TP + FP + FN + 1e-6)

    # IoU (Intersection over Union)
    iou = TP / (TP + FP + FN + 1e-6)

    # Pixel Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Sensitivity (Recall)
    sensitivity = TP / (TP + FN + 1e-6)

    # Specificity
    specificity = TN / (TN + FP + 1e-6)

    # Precision
    precision = TP / (TP + FP + 1e-6)

    metrics = {
        'dice': float(dice),
        'iou': float(iou),
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'TN': int(TN)
    }

    return metrics


def visualize_segmentation(
    images: np.ndarray,
    masks_true: np.ndarray,
    masks_pred: np.ndarray,
    indices: Optional[List[int]] = None,
    num_samples: int = 6,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 10)
):
    """
    Visualize segmentation results.

    Parameters
    ----------
    images : ndarray
        Input images (N, H, W, 1)
    masks_true : ndarray
        Ground truth masks (N, H, W, 1)
    masks_pred : ndarray
        Predicted masks (N, H, W, 1)
    indices : list, optional
        Indices of samples to show
    num_samples : int
        Number of samples to visualize
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    if indices is None:
        indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)

    n_samples = len(indices)
    fig, axes = plt.subplots(n_samples, 4, figsize=figsize)

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        # Original image
        axes[i, 0].imshow(images[idx].squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Image {idx}')
        axes[i, 0].axis('off')

        # Ground truth
        axes[i, 1].imshow(masks_true[idx].squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Prediction
        axes[i, 2].imshow(masks_pred[idx].squeeze(), cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

        # Overlay
        overlay = images[idx].squeeze()
        mask_pred_binary = (masks_pred[idx].squeeze() > 0.5).astype(float)

        # Create RGB overlay
        overlay_rgb = np.stack([overlay, overlay, overlay], axis=-1)
        overlay_rgb[mask_pred_binary > 0.5, 0] = 1.0  # Red channel for predictions

        axes[i, 3].imshow(overlay_rgb)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def plot_training_history(
    history: keras.callbacks.History,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot training history.

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

    # DICE
    axes[1].plot(history.history['dice_coefficient'], label='Train DICE', linewidth=2)
    axes[1].plot(history.history['val_dice_coefficient'], label='Val DICE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('DICE Coefficient', fontsize=12)
    axes[1].set_title('Training and Validation DICE', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")

    plt.show()


def print_metrics_report(
    metrics: Dict[str, float],
    dataset_name: str = 'Test Set'
):
    """
    Print detailed metrics report.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary
    dataset_name : str
        Name of dataset
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} Performance Report")
    print(f"{'='*60}\n")

    print(f"DICE Coefficient:  {metrics['dice']:.4f}")
    print(f"IoU (Jaccard):     {metrics['iou']:.4f}")
    print(f"Pixel Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Sensitivity:       {metrics['sensitivity']:.4f}")
    print(f"Specificity:       {metrics['specificity']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['TP']:>10d}")
    print(f"  FP: {metrics['FP']:>10d}")
    print(f"  FN: {metrics['FN']:>10d}")
    print(f"  TN: {metrics['TN']:>10d}")

    print(f"{'='*60}\n")
