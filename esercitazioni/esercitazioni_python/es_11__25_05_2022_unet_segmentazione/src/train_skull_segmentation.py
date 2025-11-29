#!/usr/bin/env python3
"""
Skull Segmentation using U-Net (Task 1 - Easy).

This script trains a U-Net model to segment the entire skull
from brain MRI images. This is a relatively easy task that serves
as pretraining for the more difficult brain matter segmentation.

Dataset: BrainWeb simulator (1810 T1 images + GRAY_MASK_B masks)
Expected Performance: DICE > 0.99, Accuracy > 0.99

Usage:
    python train_skull_segmentation.py --data_dir ../data --epochs 30
"""

import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

from utils import (
    load_dataset,
    build_unet,
    compile_unet,
    calculate_segmentation_metrics,
    visualize_segmentation,
    plot_training_history,
    print_metrics_report
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train U-Net for skull segmentation'
    )

    # Data parameters
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results/skull',
        help='Output directory for results'
    )

    # Model parameters
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Target image size (will be resized to image_size x image_size)'
    )
    parser.add_argument(
        '--encoder_depth',
        type=int,
        default=4,
        help='U-Net encoder depth'
    )
    parser.add_argument(
        '--num_first_filters',
        type=int,
        default=32,
        help='Number of filters in first encoder layer'
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='combined',
        choices=['dice', 'bce', 'combined'],
        help='Loss function'
    )

    # Data split parameters
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.70,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )

    # Other parameters
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to use (for quick testing)'
    )

    return parser.parse_args()


def main():
    """Main training pipeline for skull segmentation."""
    args = parse_args()

    # Set random seeds
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Skull Segmentation - U-Net Training")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  Encoder depth: {args.encoder_depth}")
    print(f"  First filters: {args.num_first_filters}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Loss: {args.loss}")
    print()

    # Load dataset
    print("\n" + "="*70)
    print("Step 1: Loading Dataset")
    print("="*70)

    image_dir = Path(args.data_dir) / 'MR'
    mask_dir = Path(args.data_dir) / 'GRAY_MASK_B'  # Skull masks

    target_size = (args.image_size, args.image_size) if args.image_size != 256 else None

    X, y = load_dataset(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        target_size=target_size,
        max_samples=args.max_samples,
        verbose=True
    )

    # Create data splits
    print("\n" + "="*70)
    print("Step 2: Creating Train/Val/Test Splits")
    print("="*70)

    # First split: train vs (val+test)
    test_val_ratio = 1 - args.train_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_val_ratio,
        random_state=args.random_seed
    )

    # Second split: val vs test
    val_ratio_adjusted = args.val_ratio / test_val_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio_adjusted),
        random_state=args.random_seed
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Build U-Net model
    print("\n" + "="*70)
    print("Step 3: Building U-Net Model")
    print("="*70)

    input_shape = X_train.shape[1:]  # (height, width, channels)

    model = build_unet(
        input_shape=input_shape,
        num_classes=1,  # Binary segmentation
        encoder_depth=args.encoder_depth,
        num_first_filters=args.num_first_filters,
        use_batch_norm=True,
        dropout_rate=0.2
    )

    model = compile_unet(
        model,
        learning_rate=args.learning_rate,
        optimizer='adam',
        loss=args.loss
    )

    print(f"\nModel: {model.name}")
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(output_dir / 'best_model.h5'),
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train model
    print("\n" + "="*70)
    print("Step 4: Training Model")
    print("="*70)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Plot training history
    print("\n" + "="*70)
    print("Step 5: Visualizing Training History")
    print("="*70)

    plot_training_history(
        history,
        save_path=str(output_dir / 'training_history.png')
    )

    # Evaluate on training set
    print("\n" + "="*70)
    print("Step 6: Evaluating on Training Set")
    print("="*70)

    y_train_pred = model.predict(X_train, verbose=0)
    metrics_train = calculate_segmentation_metrics(y_train, y_train_pred)
    print_metrics_report(metrics_train, 'Training Set')

    # Evaluate on validation set
    print("\n" + "="*70)
    print("Step 7: Evaluating on Validation Set")
    print("="*70)

    y_val_pred = model.predict(X_val, verbose=0)
    metrics_val = calculate_segmentation_metrics(y_val, y_val_pred)
    print_metrics_report(metrics_val, 'Validation Set')

    # Evaluate on test set
    print("\n" + "="*70)
    print("Step 8: Evaluating on Test Set")
    print("="*70)

    y_test_pred = model.predict(X_test, verbose=0)
    metrics_test = calculate_segmentation_metrics(y_test, y_test_pred)
    print_metrics_report(metrics_test, 'Test Set')

    # Visualize results
    print("\n" + "="*70)
    print("Step 9: Visualizing Segmentation Results")
    print("="*70)

    visualize_segmentation(
        X_test, y_test, y_test_pred,
        num_samples=6,
        save_path=str(output_dir / 'segmentation_results.png')
    )

    # Save model
    print("\n" + "="*70)
    print("Step 10: Saving Model")
    print("="*70)

    model_path = output_dir / 'final_model.h5'
    model.save(str(model_path))
    print(f"Model saved to {model_path}")

    # Save metrics
    np.savez(
        str(output_dir / 'metrics.npz'),
        train=metrics_train,
        val=metrics_val,
        test=metrics_test
    )
    print(f"Metrics saved to {output_dir / 'metrics.npz'}")

    # Final summary
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)

    print(f"\nFinal Test Performance:")
    print(f"  DICE:     {metrics_test['dice']:.4f}")
    print(f"  IoU:      {metrics_test['iou']:.4f}")
    print(f"  Accuracy: {metrics_test['accuracy']:.4f}")

    print(f"\nResults saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
