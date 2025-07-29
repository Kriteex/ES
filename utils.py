# utils.py
"""
Utility functions for data loading, plotting, logging, a náhodné seedování workerů.
"""

import os
import csv
import random
from typing import List, Tuple
from typing import Optional, Dict, Any

import albumentations as A
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import ALPHA, DATALOADER_CONFIG, TRAINING_CONFIG
from flexible_dataset import FlexibleDataset

def get_device() -> torch.device:
    """
    Return the available device (CUDA or CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    return device
    
def seed_worker(worker_id):
    """
    Inicializuje seed pro každého worker procesu DataLoaderu,
    aby A.RandomCrop byl pokaždé jiný.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(
    images_dir: str,
    masks_dir: str,
    n_channels: int,
    batch_size: int,
    size: int,
    filter_masks: bool
) -> DataLoader:
    """
    Vytvoří DataLoader se seed_worker pro nezávislé náhodné ořezy.
    """
    transform = A.Compose([A.RandomCrop(width=size, height=size)])
    dataset = FlexibleDataset(
        images_dir,
        masks_dir,
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
        **DATALOADER_CONFIG
    )


def plot_batch_x_channels(
    predictions: Tensor,
    gt_masks: Tensor,
    fig_name: str,
    main_folder: str = ""
) -> None:
    output_dir = os.path.join(main_folder, "Images_batch_predictions")
    os.makedirs(output_dir, exist_ok=True)

    batch_size, num_channels, _, _ = predictions.shape
    fig, axs = plt.subplots(num_channels - 1, batch_size, figsize=(20, 10), squeeze=False)

    for i in range(batch_size):
        # původní RGB
        img = predictions[i, :3].cpu().permute(1, 2, 0)
        axs[0, i].imshow(img, cmap="gray")
        axs[0, i].axis("off")

        # predikovaná maska
        pred_mask = (predictions[i, 3].cpu() > 0.5).float()
        axs[1, i].imshow(pred_mask, cmap="gray")
        axs[1, i].axis("off")

        # ground truth
        gt = gt_masks[i, 0].cpu()
        axs[2, i].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axs[2, i].axis("off")

        # případné další kanály
        if num_channels > 4:
            for j in range(4, num_channels):
                axs[j-1, i].imshow(predictions[i, j].cpu(), cmap="gray")
                axs[j-1, i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{fig_name}.png"))
    plt.close()


def calculate_pixel_accuracy(predicted_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float]:
    predicted_binary = (predicted_mask > threshold).float()
    correct = (predicted_binary == true_mask.squeeze(1)).float()
    accuracy = correct.mean().item() * 100
    return accuracy, 100 - accuracy


def composite_loss(prediction: torch.Tensor, target: torch.Tensor, pixel_accuracy_weight: float = 0.1) -> torch.Tensor:
    #mse_loss = ((target - prediction) ** 2).mean()
    #correct_pct, _ = calculate_pixel_accuracy(prediction, target)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    loss   = loss_fn(prediction, target)   # místo composite_loss

    #return mse_loss + pixel_accuracy_weight * (1.0 - correct_pct / 100.0)
    return loss

def log_loss(
    epoch: int,
    train_metrics: dict,
    test_metrics: dict,
    model_name: str,
    run_id: int,
    model_config: dict = None,
    print_results: bool = False,
) -> None:
    log_dir = model_name[:3]
    os.makedirs(log_dir, exist_ok=True)

    # --- 1. LOG FILE (.log) ---
    log_file = os.path.join(log_dir, f"{model_name}_metrics_run_{run_id}.log")
    entry = [f"Epoch: {epoch}"]

    if model_config:
        entry += [f"Config {k}: {v}" for k, v in model_config.items()]

    entry += [
        f"Train {k}: {v:.4f}" if isinstance(v, float) else f"Train {k}: {v}"
        for k, v in train_metrics.items()
    ]
    entry += [
        f"Test {k}: {v:.4f}" if isinstance(v, float) else f"Test {k}: {v}"
        for k, v in test_metrics.items()
    ]

    with open(log_file, "a") as f:
        f.write(", ".join(entry) + "\n")

    if print_results:
        print(", ".join(entry))

    # --- 2. CSV FILE FOR DATAFRAME-READY ANALYSIS ---
    csv_file = os.path.join(log_dir, f"{model_name}_metrics_run_{run_id}.csv")
    fieldnames = (
        ["epoch"]
        + ([f"conf_{k}" for k in model_config.keys()] if model_config else [])
        + [f"train_{k}" for k in train_metrics.keys()]
        + [f"test_{k}" for k in test_metrics.keys()]
    )

    values = (
        [epoch]
        + ([v for v in model_config.values()] if model_config else [])
        + [v for v in train_metrics.values()]
        + [v for v in test_metrics.values()]
    )

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow(values)



def log_hw_metrics(
    run_id: int,
    model_name: str,
    model_stats: Dict[str, Any],
    log_dir: Optional[str] = None,
) -> None:
    """
    Log model architecture and physical stats into a separate CSV and text log file.

    Parameters:
        run_id: Experiment run identifier.
        model_name: Name of the model (e.g., 'nca', 'unet').
        model_stats: Dictionary with model statistics like param_count, macs, adds, etc.
        log_dir: Optional path to store logs; if None, uses model_name prefix.
    """
    if log_dir is None:
        log_dir = f"{model_name[:3]}_hw"
    os.makedirs(log_dir, exist_ok=True)

    print(log_dir)

    # Text log file (.log)
    log_file = os.path.join(log_dir, f"{model_name}_hardware_run_{run_id}.log")
    log_entry = [f"{k}: {v}" for k, v in model_stats.items()]
    with open(log_file, "a") as f:
        f.write(", ".join(log_entry) + "\n")

    # CSV log file (.csv)
    csv_file = os.path.join(log_dir, f"{model_name}_hardware_run_{run_id}.csv")
    fieldnames = list(model_stats.keys())
    values = list(model_stats.values())

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow(values)