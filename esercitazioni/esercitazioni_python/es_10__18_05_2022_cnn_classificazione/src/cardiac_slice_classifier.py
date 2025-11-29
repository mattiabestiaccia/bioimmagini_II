#!/usr/bin/env python3
"""
Cardiac MRI Slice Classification using CNN.

This script trains and evaluates a Convolutional Neural Network (CNN) to
automatically classify short-axis cardiac MRI slices into three anatomical
positions according to the AHA (American Heart Association) model:
- Basal: Base of the heart (near valves)
- Middle: Mid-ventricular region
- Apical: Apex of the heart

The dataset consists of DICOM images from multiple MR sequences:
- Perfusion (first-pass contrast)
- Function (fast-cine)
- T2* (multi-echo)
- LGE (late gadolinium enhancement)

Usage:
    python cardiac_slice_classifier.py --data_dir ../data --epochs 50
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from utils import (
    load_dataset,
    create_data_splits,
    build_cnn_model,
    compile_model,
    plot_confusion_matrix,
    plot_training_history,
    visualize_misclassified,
    print_classification_report
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CNN for cardiac slice classification'
    )

    # Data parameters
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data',
        help='Path to data directory containing Apical/Basal/Middle folders'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results',
        help='Output directory for results'
    )

    # Model parameters
    parser.add_argument(
        '--architecture',
        type=str,
        default='vgg_small',
        choices=['simple', 'vgg_small', 'vgg_medium'],
        help='CNN architecture'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=128,
        help='Target image size (square)'
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'sgd', 'rmsprop'],
        help='Optimizer'
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
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Test set ratio'
    )

    # Other parameters
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--use_data_augmentation',
        action='store_true',
        help='Use data augmentation during training'
    )

    return parser.parse_args()


def create_data_augmentation() -> keras.Sequential:
    """
    Create data augmentation pipeline.

    Returns
    -------
    augmentation : keras.Sequential
        Data augmentation layers
    """
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),  # +/- 10%
        keras.layers.RandomZoom(0.1),      # +/- 10%
        keras.layers.RandomTranslation(0.1, 0.1),  # +/- 10%
    ], name='data_augmentation')

    return augmentation


def main():
    """Main training and evaluation pipeline."""
    args = parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Cardiac Slice Classification - CNN Training")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Data augmentation: {args.use_data_augmentation}")
    print()

    # Load dataset
    print("\n" + "="*70)
    print("Step 1: Loading Dataset")
    print("="*70)

    X, y, class_names = load_dataset(
        data_dir=args.data_dir,
        target_size=(args.image_size, args.image_size),
        verbose=True
    )

    # Create data splits
    print("\n" + "="*70)
    print("Step 2: Creating Train/Val/Test Splits")
    print("="*70)

    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(
        X, y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed
    )

    # Convert labels to one-hot
    num_classes = len(class_names)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Build model
    print("\n" + "="*70)
    print("Step 3: Building CNN Model")
    print("="*70)

    input_shape = (args.image_size, args.image_size, 1)
    model = build_cnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        architecture=args.architecture
    )

    model = compile_model(
        model,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer
    )

    print(f"\nModel: {model.name}")
    model.summary()

    # Data augmentation (optional)
    if args.use_data_augmentation:
        print("\nApplying data augmentation...")
        augmentation = create_data_augmentation()

        # Add augmentation to model
        augmented_model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            augmentation,
            model
        ], name='augmented_model')

        augmented_model.compile(
            optimizer=model.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model = augmented_model

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
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train model
    print("\n" + "="*70)
    print("Step 4: Training Model")
    print("="*70)

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
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

    y_train_pred_probs = model.predict(X_train, verbose=0)
    y_train_pred = np.argmax(y_train_pred_probs, axis=1)

    print_classification_report(
        y_train, y_train_pred, class_names, dataset_name='Training Set'
    )

    plot_confusion_matrix(
        y_train, y_train_pred, class_names,
        title='Confusion Matrix - Training Set',
        save_path=str(output_dir / 'confusion_matrix_train.png')
    )

    # Evaluate on validation set
    print("\n" + "="*70)
    print("Step 7: Evaluating on Validation Set")
    print("="*70)

    y_val_pred_probs = model.predict(X_val, verbose=0)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)

    print_classification_report(
        y_val, y_val_pred, class_names, dataset_name='Validation Set'
    )

    plot_confusion_matrix(
        y_val, y_val_pred, class_names,
        title='Confusion Matrix - Validation Set',
        save_path=str(output_dir / 'confusion_matrix_val.png')
    )

    # Evaluate on test set
    print("\n" + "="*70)
    print("Step 8: Evaluating on Test Set")
    print("="*70)

    y_test_pred_probs = model.predict(X_test, verbose=0)
    y_test_pred = np.argmax(y_test_pred_probs, axis=1)

    print_classification_report(
        y_test, y_test_pred, class_names, dataset_name='Test Set'
    )

    plot_confusion_matrix(
        y_test, y_test_pred, class_names,
        title='Confusion Matrix - Test Set',
        save_path=str(output_dir / 'confusion_matrix_test.png')
    )

    # Visualize misclassified samples
    print("\n" + "="*70)
    print("Step 9: Analyzing Misclassified Samples")
    print("="*70)

    visualize_misclassified(
        X_test, y_test, y_test_pred, class_names,
        num_samples=9,
        save_path=str(output_dir / 'misclassified_samples.png')
    )

    # Save model
    print("\n" + "="*70)
    print("Step 10: Saving Model")
    print("="*70)

    model_path = output_dir / 'final_model.h5'
    model.save(str(model_path))
    print(f"Model saved to {model_path}")

    # Save predictions
    np.savez(
        str(output_dir / 'predictions.npz'),
        y_train_true=y_train,
        y_train_pred=y_train_pred,
        y_val_true=y_val,
        y_val_pred=y_val_pred,
        y_test_true=y_test,
        y_test_pred=y_test_pred,
        class_names=class_names
    )
    print(f"Predictions saved to {output_dir / 'predictions.npz'}")

    # Final summary
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)

    train_acc = np.mean(y_train == y_train_pred)
    val_acc = np.mean(y_val == y_val_pred)
    test_acc = np.mean(y_test == y_test_pred)

    print(f"\nFinal Accuracies:")
    print(f"  Training:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Validation: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Test:       {test_acc:.4f} ({test_acc*100:.2f}%)")

    print(f"\nResults saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
