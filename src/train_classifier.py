#!/usr/bin/env python3
"""
Stage B: NSFW Classifier Training for pNSFWMedia

Trains an MLP classifier on CLIP embeddings using TensorFlow/Keras.
Includes KerasTuner for hyperparameter optimization, tqdm for progress bars,
and TensorBoard for visualization.

Reference: Twitter's pNSFWMedia implementation (nsfw_media.py)
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import keras_tuner as kt
import numpy as np
import sklearn.metrics
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

# Configure GPU/CUDA
def setup_gpu():
    """Setup GPU/CUDA for training."""
    physical_devices = tf.config.list_physical_devices('GPU')

    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"[GPU] CUDA is available. Found {len(physical_devices)} GPU(s):")
            for i, device in enumerate(physical_devices):
                print(f"  [{i}] {device.name}")
            return True
        except RuntimeError as e:
            print(f"[GPU] Error configuring GPU: {e}")
            return False
    else:
        print("[GPU] CUDA is not available. Using CPU for training.")
        return False

CUDA_AVAILABLE = setup_gpu()


class EmbeddingDataLoader:
    """Load embeddings from .npy files for training with automatic train/val split."""

    def __init__(
        self,
        embeddings_dir: str,
        embedding_dim: int = 256,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Initialize the data loader.

        Automatically splits data into train/val sets.
        Expects sfw/nsfw subdirectories directly under embeddings_dir.

        Args:
            embeddings_dir: Root directory containing sfw/ and nsfw/ subdirs
            embedding_dim: Expected embedding dimension (default: 256)
            val_ratio: Validation ratio (default: 0.15 = 15%)
            seed: Random seed for reproducible splitting
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.embedding_dim = embedding_dim
        self.val_ratio = val_ratio
        self.seed = seed

        # Cache for split data
        self._train_data = None
        self._val_data = None

        # Always use automatic split
        self._prepare_auto_split()

    def _prepare_auto_split(self):
        """
        Load all embeddings and split into train/val automatically.
        Train: 85%, Val: 15%
        """
        print(f"\nAuto-split mode enabled (val_ratio={self.val_ratio:.0%})")

        all_embeddings = []
        all_labels = []
        all_files = []

        for cls, label in [("sfw", 0), ("nsfw", 1)]:
            # Try direct sfw/nsfw subdirs first
            cls_dir = self.embeddings_dir / cls
            if not cls_dir.exists():
                # Fallback: try without subdirs (flat structure)
                print(f"Warning: {cls_dir} does not exist")
                continue

            npy_files = sorted(cls_dir.glob("*.npy"))
            print(f"Found {len(npy_files)} {cls} embeddings")

            for npy_file in tqdm(npy_files, desc=f"Loading {cls}", leave=False):
                try:
                    emb = np.load(npy_file)
                    if emb.shape == (self.embedding_dim,):
                        all_embeddings.append(emb)
                        all_labels.append(label)
                        all_files.append(str(npy_file))
                except Exception as e:
                    print(f"Error loading {npy_file}: {e}")

        if not all_embeddings:
            raise ValueError(f"No embeddings found in {self.embeddings_dir}")

        all_embeddings = np.array(all_embeddings, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int32)

        # Stratified split to maintain class balance
        np.random.seed(self.seed)

        # Get indices for each class
        sfw_indices = np.where(all_labels == 0)[0]
        nsfw_indices = np.where(all_labels == 1)[0]

        # Shuffle indices
        np.random.shuffle(sfw_indices)
        np.random.shuffle(nsfw_indices)

        # Split each class
        n_sfw_val = int(len(sfw_indices) * self.val_ratio)
        n_nsfw_val = int(len(nsfw_indices) * self.val_ratio)

        sfw_val_idx = sfw_indices[:n_sfw_val]
        sfw_train_idx = sfw_indices[n_sfw_val:]
        nsfw_val_idx = nsfw_indices[:n_nsfw_val]
        nsfw_train_idx = nsfw_indices[n_nsfw_val:]

        # Combine indices
        train_indices = np.concatenate([sfw_train_idx, nsfw_train_idx])
        val_indices = np.concatenate([sfw_val_idx, nsfw_val_idx])

        # Shuffle combined indices
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        # Store split data
        self._train_data = (all_embeddings[train_indices], all_labels[train_indices])
        self._val_data = (all_embeddings[val_indices], all_labels[val_indices])

        # Print summary
        train_emb, train_lbl = self._train_data
        val_emb, val_lbl = self._val_data

        print(f"\n{'=' * 50}")
        print("Auto-split Summary")
        print(f"{'=' * 50}")
        print(f"Total: {len(all_embeddings)} embeddings")
        print(f"Train: {len(train_emb)} ({len(train_emb)/len(all_embeddings)*100:.1f}%)")
        print(f"  - SFW:  {np.sum(train_lbl == 0)}")
        print(f"  - NSFW: {np.sum(train_lbl == 1)}")
        print(f"Val:   {len(val_emb)} ({len(val_emb)/len(all_embeddings)*100:.1f}%)")
        print(f"  - SFW:  {np.sum(val_lbl == 0)}")
        print(f"  - NSFW: {np.sum(val_lbl == 1)}")
        print(f"{'=' * 50}\n")

    def load_split(self, split: str) -> tuple:
        """
        Load embeddings and labels for a given split.

        Args:
            split: 'train' or 'val'

        Returns:
            Tuple of (embeddings, labels) as numpy arrays
        """
        if split == "train":
            return self._train_data
        elif split == "val":
            return self._val_data
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train' or 'val'.")

    def create_tf_dataset(
        self,
        split: str,
        batch_size: int = 64,
        shuffle: bool = True,
        repeat: bool = False
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset from embeddings.

        Args:
            split: 'train' or 'val'
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            repeat: Whether to repeat the dataset

        Returns:
            tf.data.Dataset
        """
        embeddings, labels = self.load_split(split)

        dataset = tf.data.Dataset.from_tensor_slices((embeddings, labels))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(embeddings))

        dataset = dataset.batch(batch_size)

        if repeat:
            dataset = dataset.repeat()

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset, len(embeddings)


def compute_class_weights(labels: np.ndarray) -> dict:
    """
    Compute class weights for imbalanced dataset.

    Args:
        labels: Array of labels (0 or 1)

    Returns:
        Dictionary mapping class indices to weights
    """
    n_samples = len(labels)
    n_classes = 2
    n_pos = np.sum(labels)
    n_neg = n_samples - n_pos

    # Balanced weights
    weight_pos = n_samples / (n_classes * n_pos)
    weight_neg = n_samples / (n_classes * n_neg)

    return {0: weight_neg, 1: weight_pos}


def get_metrics():
    """Get the metrics used for model evaluation (matching pNSFWMedia)."""
    return [
        tf.keras.metrics.PrecisionAtRecall(
            recall=0.9, num_thresholds=200, name="precision_at_recall_0.9"
        ),
        tf.keras.metrics.AUC(
            num_thresholds=200,
            curve="PR",
            name="pr_auc"
        ),
        tf.keras.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            name="roc_auc"
        ),
    ]


def build_model(hp=None, config: dict = None):
    """
    Build the NSFW classifier model.

    Matches the pNSFWMedia architecture:
    - BatchNormalization + Dense blocks (1-2 layers)
    - Output: Dense(1, sigmoid)

    Args:
        hp: KerasTuner HyperParameters object (for tuning)
        config: Fixed configuration dict (for final training)

    Returns:
        Compiled Keras Sequential model
    """
    model = tf.keras.Sequential()

    # Hyperparameters
    if hp is not None:
        # Tuning mode
        activation = hp.Choice("activation", ["tanh", "gelu"])
        kernel_initializer = hp.Choice("kernel_initializer", ["he_uniform", "glorot_uniform"])
        num_layers = hp.Int("num_layers", 1, 2)
        units = hp.Int("units", min_value=128, max_value=256, step=128)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
        dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)
    else:
        # Fixed config mode
        config = config or {}
        activation = config.get("activation", "tanh")
        kernel_initializer = config.get("kernel_initializer", "he_uniform")
        num_layers = config.get("num_layers", 1)
        units = config.get("units", 256)
        learning_rate = config.get("learning_rate", 1e-3)
        dropout_rate = config.get("dropout_rate", 0.0)

    # Build layers (matching pNSFWMedia structure)
    for i in range(num_layers):
        model.add(tf.keras.layers.BatchNormalization())

        if i == 0:
            model.add(
                tf.keras.layers.Dense(
                    units=units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    input_shape=(256,)
                )
            )
        else:
            model.add(
                tf.keras.layers.Dense(
                    units=units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                )
            )

        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    # Output layer
    model.add(
        tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer=kernel_initializer
        )
    )

    # Optimizer (matching pNSFWMedia)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        amsgrad=False,
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=get_metrics()
    )

    return model


class TqdmCallback(tf.keras.callbacks.Callback):
    """Custom callback for tqdm progress bar during training."""

    def __init__(self, epochs: int, steps_per_epoch: int = None):
        super().__init__()
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.epoch_bar = None
        self.batch_bar = None

    def on_train_begin(self, logs=None):
        self.epoch_bar = tqdm(total=self.epochs, desc="Training", unit="epoch", position=0)

    def on_epoch_begin(self, epoch, logs=None):
        if self.steps_per_epoch:
            self.batch_bar = tqdm(
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="batch",
                position=1,
                leave=False
            )

    def on_batch_end(self, batch, logs=None):
        if self.batch_bar:
            logs = logs or {}
            loss = logs.get('loss', 0)
            pr_auc = logs.get('pr_auc', 0)
            self.batch_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'pr_auc': f'{pr_auc:.4f}'
            })
            self.batch_bar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        if self.batch_bar:
            self.batch_bar.close()

        logs = logs or {}
        val_loss = logs.get('val_loss', 0)
        val_pr_auc = logs.get('val_pr_auc', 0)
        val_p_at_r = logs.get('val_precision_at_recall_0.9', 0)

        self.epoch_bar.set_postfix({
            'val_loss': f'{val_loss:.4f}',
            'val_pr_auc': f'{val_pr_auc:.4f}',
            'val_p@r0.9': f'{val_p_at_r:.4f}'
        })
        self.epoch_bar.update(1)

    def on_train_end(self, logs=None):
        if self.epoch_bar:
            self.epoch_bar.close()


def run_hyperparameter_search(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    steps_per_epoch: int,
    max_trials: int = 30,
    epochs: int = 100,
    patience: int = 5,
    log_dir: str = "logs"
):
    """
    Run Bayesian hyperparameter optimization using KerasTuner.

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        steps_per_epoch: Steps per epoch
        max_trials: Maximum number of trials
        epochs: Maximum epochs per trial
        patience: Early stopping patience
        log_dir: Directory for TensorBoard logs

    Returns:
        Best hyperparameters
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tuner_dir = f"tuner_dir/pnsfwmedia_{timestamp}"

    tuner = kt.tuners.BayesianOptimization(
        build_model,
        objective=kt.Objective('val_pr_auc', direction="max"),
        max_trials=max_trials,
        directory=tuner_dir,
        project_name='pnsfwmedia'
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=patience,
            verbose=1,
            mode='auto',
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f"{log_dir}/pnsfwmedia/{timestamp}/tuning",
            histogram_freq=0,
            write_graph=True,
            update_freq="batch"
        )
    ]

    print("\n" + "=" * 60)
    print("Starting Hyperparameter Search")
    print("=" * 60)

    tuner.search(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=2,
        validation_data=val_ds,
        callbacks=callbacks
    )

    tuner.results_summary()

    best_hps = tuner.get_best_hyperparameters()[0]
    print("\nBest hyperparameters:")
    for key, value in best_hps.values.items():
        print(f"  {key}: {value}")

    return best_hps.values


def train_model(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    steps_per_epoch: int,
    config: dict,
    epochs: int = 40,
    patience: int = 10,
    class_weight: dict = None,
    log_dir: str = "logs",
    model_save_path: str = "models/pnsfwmedia_classifier.keras"
):
    """
    Train the final model with given configuration.

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        steps_per_epoch: Steps per epoch
        config: Model configuration
        epochs: Number of epochs
        patience: Early stopping patience
        class_weight: Class weights for imbalanced data
        log_dir: Directory for TensorBoard logs
        model_save_path: Path to save the trained model

    Returns:
        Trained model and history
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_dir = f"{log_dir}/pnsfwmedia/{timestamp}/training"

    model = build_model(config=config)
    model.build(input_shape=(None, 256))

    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_pr_auc',
            min_delta=0,
            patience=patience,
            verbose=1,
            mode='max',
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq="batch"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path.replace('.keras', '_best.keras'),
            monitor='val_pr_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        TqdmCallback(epochs=epochs, steps_per_epoch=steps_per_epoch)
    ]

    print("\n" + "=" * 60)
    print("Starting Training")
    print(f"TensorBoard logs: {tensorboard_dir}")
    print("=" * 60 + "\n")

    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0  # Disable default progress since we use tqdm
    )

    # Save final model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")

    return model, history


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    save_dir: str = "results"
):
    """
    Evaluate the model and generate metrics/plots.

    Args:
        model: Trained Keras model
        test_ds: Test dataset
        save_dir: Directory to save evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Evaluating Model")
    print("=" * 60)

    # Collect predictions
    all_labels = []
    all_preds = []

    for batch_features, batch_labels in tqdm(test_ds, desc="Evaluating"):
        preds = model.predict(batch_features, verbose=0)
        all_preds.extend(preds.flatten())
        all_labels.extend(batch_labels.numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Calculate metrics
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
        all_labels, all_preds
    )
    pr_auc = sklearn.metrics.auc(recall, precision)

    fpr, tpr, _ = sklearn.metrics.roc_curve(all_labels, all_preds)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    # Find precision at specific recall thresholds
    def get_precision_at_recall(target_recall, recall, precision):
        idx = np.argmin(np.abs(recall - target_recall))
        return precision[idx], recall[idx]

    p_at_r90, actual_r90 = get_precision_at_recall(0.9, recall, precision)
    p_at_r95, actual_r95 = get_precision_at_recall(0.95, recall, precision)
    p_at_r50, actual_r50 = get_precision_at_recall(0.5, recall, precision)

    # Print results
    print("\n" + "-" * 40)
    print("Evaluation Results")
    print("-" * 40)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Precision @ Recall=0.9:  {p_at_r90:.4f}")
    print(f"Precision @ Recall=0.95: {p_at_r95:.4f}")
    print(f"Precision @ Recall=0.5:  {p_at_r50:.4f}")
    print("-" * 40)

    n_total = len(all_labels)
    n_pos = np.sum(all_labels)
    n_neg = n_total - n_pos
    print(f"Test set: {n_total} samples ({n_pos} NSFW, {n_neg} SFW)")

    # Save metrics to file
    with open(f"{save_dir}/evaluation_metrics.txt", "w") as f:
        f.write("pNSFWMedia Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC:  {pr_auc:.4f}\n")
        f.write(f"Precision @ Recall=0.9:  {p_at_r90:.4f}\n")
        f.write(f"Precision @ Recall=0.95: {p_at_r95:.4f}\n")
        f.write(f"Precision @ Recall=0.5:  {p_at_r50:.4f}\n")
        f.write(f"Test samples: {n_total} ({n_pos} NSFW, {n_neg} SFW)\n")

    # Plot PR curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.4f})', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Mark key points
    plt.axvline(x=0.9, color='r', linestyle='--', alpha=0.5, label=f'P@R=0.9: {p_at_r90:.4f}')
    plt.axhline(y=p_at_r90, color='r', linestyle='--', alpha=0.5)
    plt.scatter([0.9], [p_at_r90], color='r', s=100, zorder=5)

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/pr_curve.png", dpi=150)
    plt.savefig(f"{save_dir}/pr_curve.pdf")
    plt.close()

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve (AUC = {roc_auc:.4f})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curve.png", dpi=150)
    plt.savefig(f"{save_dir}/roc_curve.pdf")
    plt.close()

    # Plot prediction histogram
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(all_preds[all_labels == 0], bins=50, alpha=0.7, label='SFW', color='blue')
    plt.hist(all_preds[all_labels == 1], bins=50, alpha=0.7, label='NSFW', color='red')
    plt.xlabel('NSFW Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Distribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(all_preds, bins=50, alpha=0.7, color='green')
    plt.xlabel('NSFW Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Overall Prediction Distribution', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/prediction_histogram.png", dpi=150)
    plt.close()

    print(f"\nEvaluation results saved to: {save_dir}/")

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision_at_recall_0.9': p_at_r90,
        'precision_at_recall_0.95': p_at_r95,
        'precision_at_recall_0.5': p_at_r50
    }


def plot_training_history(history, save_dir: str = "results"):
    """Plot training history curves."""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(20, 5))

    # PR-AUC
    plt.subplot(1, 3, 1)
    plt.plot(history.history.get('pr_auc', []), label='train')
    plt.plot(history.history.get('val_pr_auc', []), label='val')
    plt.title('PR-AUC')
    plt.ylabel('PR-AUC')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history.get('loss', []), label='train')
    plt.plot(history.history.get('val_loss', []), label='val')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Precision at Recall
    plt.subplot(1, 3, 3)
    plt.plot(history.history.get('precision_at_recall_0.9', []), label='train')
    plt.plot(history.history.get('val_precision_at_recall_0.9', []), label='val')
    plt.title('Precision @ Recall=0.9')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_history.png", dpi=150)
    plt.savefig(f"{save_dir}/training_history.pdf")
    plt.close()

    print(f"Training history plots saved to: {save_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Train pNSFWMedia classifier on CLIP embeddings"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="dataset/embeddings",
        help="Directory containing sfw/ and nsfw/ embedding subdirectories"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Maximum number of epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--units",
        type=int,
        default=256,
        help="Number of units in hidden layers"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of hidden layers"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="tanh",
        choices=["tanh", "gelu", "relu"],
        help="Activation function"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning"
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=30,
        help="Maximum trials for hyperparameter tuning"
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="models/pnsfwmedia_classifier.keras",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for evaluation results"
    )
    parser.add_argument(
        "--use-class-weight",
        action="store_true",
        help="Use class weighting for imbalanced data"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio for train/val split (default: 0.15 = 15%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data splitting"
    )

    args = parser.parse_args()

    # Load data
    print("\n" + "=" * 60)
    print("Loading Embeddings")
    if CUDA_AVAILABLE:
        print("[GPU] Training will use CUDA acceleration")
    else:
        print("[CPU] Training will use CPU")
    print("=" * 60)

    data_loader = EmbeddingDataLoader(
        args.embeddings_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    train_ds, train_size = data_loader.create_tf_dataset(
        "train",
        batch_size=args.batch_size,
        shuffle=True,
        repeat=True
    )

    val_ds, val_size = data_loader.create_tf_dataset(
        "val",
        batch_size=args.batch_size,
        shuffle=False,
        repeat=False
    )

    steps_per_epoch = train_size // args.batch_size
    print(f"\nSteps per epoch: {steps_per_epoch}")

    # Compute class weights if requested
    class_weight = None
    if args.use_class_weight:
        _, train_labels = data_loader.load_split("train")
        class_weight = compute_class_weights(train_labels)
        print(f"Class weights: {class_weight}")

    # Hyperparameter tuning or fixed training
    if args.tune:
        best_config = run_hyperparameter_search(
            train_ds,
            val_ds,
            steps_per_epoch,
            max_trials=args.max_trials,
            epochs=args.epochs,
            patience=args.patience,
            log_dir=args.log_dir
        )
    else:
        best_config = {
            "activation": args.activation,
            "kernel_initializer": "he_uniform",
            "num_layers": args.num_layers,
            "units": args.units,
            "learning_rate": args.learning_rate,
            "dropout_rate": args.dropout
        }

    # Train final model
    model, history = train_model(
        train_ds,
        val_ds,
        steps_per_epoch,
        config=best_config,
        epochs=args.epochs,
        patience=args.patience * 2,  # More patience for final training
        class_weight=class_weight,
        log_dir=args.log_dir,
        model_save_path=args.model_save_path
    )

    # Plot training history
    plot_training_history(history, args.results_dir)

    # Evaluate on validation set
    val_ds_eval, _ = data_loader.create_tf_dataset(
        "val",
        batch_size=args.batch_size,
        shuffle=False,
        repeat=False
    )

    evaluate_model(model, val_ds_eval, args.results_dir)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {args.model_save_path}")
    print(f"TensorBoard logs: {args.log_dir}/pnsfwmedia/")
    print(f"Evaluation results: {args.results_dir}/")
    print("\nTo view TensorBoard logs:")
    print(f"  tensorboard --logdir={args.log_dir}")


if __name__ == "__main__":
    main()
