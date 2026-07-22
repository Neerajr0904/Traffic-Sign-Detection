"""
train.py
--------
End-to-end training pipeline for the traffic sign detection CNN:
load GTSRB data -> preprocess -> augment -> train -> evaluate -> save.

Usage:
    python train.py --train_dir data/Train --test_dir data/Test \
        --test_csv data/Test.csv --epochs 30 --batch_size 64
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from preprocessing import (
    load_training_data,
    load_test_data,
    prepare_splits,
    get_train_augmentor,
    CLASS_NAMES,
)
from model import build_cnn


def parse_args():
    parser = argparse.ArgumentParser(description="Train the traffic sign detection CNN")
    parser.add_argument("--train_dir", type=str, default="data/Train")
    parser.add_argument("--test_dir", type=str, default="data/Test")
    parser.add_argument("--test_csv", type=str, default="data/Test.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_dir", type=str, default="models")
    return parser.parse_args()


def plot_history(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="train_accuracy")
    axes[0].plot(history.history["val_accuracy"], label="val_accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title("Confusion Matrix - Traffic Sign Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    print("Loading training data...")
    images, labels = load_training_data(args.train_dir)
    print(f"Loaded {len(images)} training images across {len(set(labels))} classes")

    X_train, X_val, y_train, y_val = prepare_splits(images, labels)

    print("Building model...")
    model = build_cnn()
    model.summary()

    datagen = get_train_augmentor()
    datagen.fit(X_train)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(
            os.path.join(args.model_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    print("Training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plot_history(history, args.output_dir)

    # Final evaluation on the held-out GTSRB test set, if available
    if os.path.exists(args.test_dir) and os.path.exists(args.test_csv):
        print("Evaluating on test set...")
        X_test, y_test = load_test_data(args.test_dir, args.test_csv)
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        print(classification_report(
            y_test, y_pred, target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
        ))
        plot_confusion_matrix(y_test, y_pred, args.output_dir)

    model.save(os.path.join(args.model_dir, "traffic_sign_cnn_final.keras"))
    print(f"Model saved to {args.model_dir}/traffic_sign_cnn_final.keras")


if __name__ == "__main__":
    main()
