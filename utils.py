# utils.py
"""
Utility functions for data loading, visualization and logging.

This module collects reusable helpers such as dataloader creation, random seeding,
batch plotting, accuracy/loss computation, and experiment logging.
"""

import os
import csv
import random
from typing import Dict, Any, Callable, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import ALPHA, DATALOADER_CONFIG
from flexible_dataset import FlexibleDataset

def get_device() -> torch.device:
    """Return the best available PyTorch device (CUDA if available)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def seed_worker(worker_id: int) -> None:
    """
    Initialize the random seed for each DataLoader worker.

    Albumentations uses random operations (e.g., RandomCrop) that must be
    independent across workers to avoid correlation in augmented batches.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def create_dataloader(
    images_dir: str,
    masks_dir: str,
    n_channels: int,
    batch_size: int,
    crop_size: int,
    filter_masks: bool = False,
) -> DataLoader:
    """
    Construct a DataLoader with deterministic worker seeding.

    Args:
        images_dir: Directory with input images.
        masks_dir: Directory with ground‑truth masks.
        n_channels: Number of channels to use (RGB plus optional extra).
        batch_size: Number of samples per batch.
        crop_size: Square patch size to crop from each image.
        filter_masks: If True, only keep masks with specific file endings.

    Returns:
        A PyTorch DataLoader over the FlexibleDataset.
    """
    transform = A.Compose([A.RandomCrop(width=crop_size, height=crop_size)])
    dataset = FlexibleDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        number_of_channels=n_channels,
        transform=transform,
        alpha=ALPHA,
        alpha_init=None,
        filter_masks=filter_masks,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        **DATALOADER_CONFIG,
    )

def save_batch_visualization(
    predictions: Tensor,
    gt_masks: Tensor,
    filename: str,
    out_dir: str = "",
) -> None:
    """
    Save a grid of predictions and ground‑truth masks for qualitative inspection.

    The function plots the RGB channels, the predicted binary mask, the ground
    truth mask, and any additional channels.  Each column corresponds to one
    sample in the batch.

    Args:
        predictions: Model outputs of shape (B, C, H, W).
        gt_masks: Ground truth masks of shape (B, 1, H, W).
        filename: Name of the saved image file (without extension).
        out_dir: Optional subdirectory to save the image into.
    """
    output_dir = os.path.join(out_dir, "Images_batch_predictions")
    os.makedirs(output_dir, exist_ok=True)

    B, C, _, _ = predictions.shape
    rows = max(3, C)
    fig, axs = plt.subplots(rows, B, figsize=(4 * B, 3 * rows), squeeze=False)

    for idx in range(B):
        rgb = predictions[idx, :3].cpu().permute(1, 2, 0)
        pred_mask = (predictions[idx, 3].cpu() > 0.5).float()
        gt = gt_masks[idx, 0].cpu()

        axs[0, idx].imshow(rgb)
        axs[0, idx].axis("off")
        axs[1, idx].imshow(pred_mask, cmap="gray")
        axs[1, idx].axis("off")
        axs[2, idx].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axs[2, idx].axis("off")

        for extra_idx in range(4, C):
            row = extra_idx - 1
            axs[row, idx].imshow(predictions[idx, extra_idx].cpu(), cmap="gray")
            axs[row, idx].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close(fig)

def compute_pixel_accuracy(
    predicted: Tensor,
    target: Tensor,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute the percentage of correctly and incorrectly classified pixels.

    Args:
        predicted: Raw model outputs of shape (B, 1, H, W).
        target: Binary masks of shape (B, 1, H, W).
        threshold: Decision threshold to binarize predicted logits.

    Returns:
        A tuple of (accuracy_percent, error_percent).
    """
    binary_pred = (predicted > threshold).float()
    correct = (binary_pred == target.squeeze(1)).float()
    accuracy = correct.mean().item() * 100.0
    return accuracy, 100.0 - accuracy

def compute_binary_cross_entropy(prediction: Tensor, target: Tensor) -> Tensor:
    """Wrapper for BCEWithLogitsLoss to match the composite_loss API."""
    loss_fn = torch.nn.BCEWithLogitsLoss()
    return loss_fn(prediction, target)

def log_metrics_to_files(
    epoch: int,
    train_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    model_name: str,
    run_id: int,
    model_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> None:
    """
    Append training/test metrics to both a human‑readable log file and a CSV file.

    Args:
        epoch: Current epoch number (starting from 0).
        train_metrics: Dictionary of metrics computed on the training set.
        test_metrics: Dictionary of metrics computed on the validation/test set.
        model_name: Name of the model, used as a prefix for log directories.
        run_id: Identifier of the current experiment run.
        model_config: Optional dictionary of hyperparameters for the model.
        verbose: If True, print the metrics to stdout.
    """
    log_dir = model_name[:3]
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{model_name}_metrics_run_{run_id}.log")
    entries: List[str] = [f"Epoch: {epoch}"]
    if model_config:
        entries.extend(f"Config {k}: {v}" for k, v in model_config.items())
    entries.extend(
        f"Train {k}: {v:.4f}" if isinstance(v, float) else f"Train {k}: {v}"
        for k, v in train_metrics.items()
    )
    entries.extend(
        f"Test {k}: {v:.4f}" if isinstance(v, float) else f"Test {k}: {v}"
        for k, v in test_metrics.items()
    )
    with open(log_path, "a") as f:
        f.write(", ".join(entries) + "\n")
    if verbose:
        print(", ".join(entries))

    csv_path = os.path.join(log_dir, f"{model_name}_metrics_run_{run_id}.csv")
    header = ["epoch"]
    if model_config:
        header += [f"conf_{k}" for k in model_config]
    header += [f"train_{k}" for k in train_metrics] + [f"test_{k}" for k in test_metrics]
    values = [epoch]
    if model_config:
        values += list(model_config.values())
    values += list(train_metrics.values()) + list(test_metrics.values())

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(values)

def log_model_statistics(
    run_id: int,
    model_name: str,
    stats: Dict[str, Any],
    directory: Optional[str] = None,
) -> None:
    """
    Log model architecture statistics (e.g., params, MACs) to text and CSV files.

    Args:
        run_id: Experiment run identifier.
        model_name: Prefix used for output directories.
        stats: Dictionary containing statistic names and values.
        directory: Optional override for the log directory.  Defaults to
            f"{model_name[:3]}_hw".
    """
    log_dir = directory or f"{model_name[:3]}_hw"
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, f"{model_name}_hardware_run_{run_id}.log"), "a") as f:
        f.write(", ".join(f"{k}: {v}" for k, v in stats.items()) + "\n")

    csv_path = os.path.join(log_dir, f"{model_name}_hardware_run_{run_id}.csv")
    header = list(stats.keys())
    values = list(stats.values())
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(values)
