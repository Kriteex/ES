"""
Combined training script (short version enhanced) that merges the functionality of
the original short "train.py" with the extended capabilities from the longer file.

Key additions:
- NCA now supports composite_loss with pixel-accuracy weighting (train/eval),
  falling back to BCE if composite_loss is not available.
- Utilities are abstracted via compatibility aliases to support both naming
  schemes (compute_/create_/log_metrics_to_files vs calculate_/get_/log_loss).
- Logging of HW/model statistics uses a unified wrapper (log_model_statistics vs log_hw_metrics).
- best_epoch is tracked and returned from main() and correctly used in __main__.
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.signal import convolve2d
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from segmentation_models_pytorch import PSPNet
from torchvision.models.segmentation import deeplabv3_resnet50

from config import (
    CAMODEL_CONFIG,
    DEEPLABV3_CONFIG,
    DEEPLABV3_CONFIG_CONV1,
    DEVICE,
    PSPNET_CONFIG,
    SEGNET_CONFIG,
    SEED,
    TRAINING_CONFIG,
    UNET_CONFIG,
)
from model import CAModel, SegNet, UNetNormal, UNetTiny

# --- Utilities: bring both naming schemes under common aliases ----------------

from utils import (
    # legacy (short script A) names
    compute_pixel_accuracy as _compute_pixel_accuracy,
    compute_binary_cross_entropy as _compute_binary_cross_entropy,
    create_dataloader as _create_dataloader,
    log_metrics_to_files as _log_metrics_to_files,
    log_model_statistics as _log_model_statistics,
    save_batch_visualization as _save_batch_visualization,  # may remain unused
)


_HAVE_NEW_UTILS = False

# Unify surface API for the rest of the script
compute_pixel_accuracy_fn = _calculate_pixel_accuracy if _HAVE_NEW_UTILS else _compute_pixel_accuracy
get_dataloader_fn = _get_dataloader if _HAVE_NEW_UTILS else _create_dataloader

def _log_results(epoch: int,
                 train_dict: Dict[str, float],
                 test_dict: Dict[str, float],
                 exp_name: str,
                 run: int,
                 *,
                 model_config: Optional[Dict[str, Any]] = None) -> None:
    """Compat wrapper to log metrics regardless of utils version."""
    if _HAVE_NEW_UTILS:
        _log_loss(
            epoch,
            train_dict,
            test_dict,
            exp_name,
            run,
            model_config=model_config or {},
            print_results=True,
        )
    else:
        _log_metrics_to_files(
            epoch,
            train_dict,
            test_dict,
            exp_name,
            run,
            model_config=model_config or {},
            verbose=True,
        )

def _log_hw(run_id: int, model_name: str, model_stats: Dict[str, Any]) -> None:
    """Compat wrapper to log HW/model stats for both utils variants."""
    if _HAVE_NEW_UTILS:
        _log_hw_metrics(run_id=run_id, model_name=model_name, model_stats=model_stats)
    else:
        _log_model_statistics(run_id=run_id, model_name=model_name, stats=model_stats)

# ---------------------------------------------------------------------------
# Seeding & multiprocessing configuration
# ---------------------------------------------------------------------------

# Set random seeds for reproducibility across Python, NumPy and PyTorch.
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Ensure that the multiprocessing start method is explicitly set.
mp.set_start_method("spawn", force=True)

# ---------------------------------------------------------------------------
# Metric definitions and helper functions
# ---------------------------------------------------------------------------

METRIC_TAGS: Tuple[str, ...] = (
    "Loss",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "FM",
    "p-FM",
    "PSNR",
    "DRD",
)

# 5×5 DRD weights (Manhattan distance decay)
_DRD_WEIGHTS = np.array(
    [[1.0 / (abs(i - 2) + abs(j - 2) + 1) for j in range(5)] for i in range(5)]
)

WNM = _DRD_WEIGHTS

def count_nubn(gt: np.ndarray, block_size: int = 8) -> int:
    """
    Spočítá NUBN – počet neuniformních (tj. obsahujících alespoň jeden pixel 0 i 1) bloků block_size×block_size v GT.
    Pokud by GT neobsahovala žádný takový blok, vrací 1, aby se předešlo dělení nulou.
    """
    h, w = gt.shape
    nubn = 0
    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            block = gt[by:by + block_size, bx:bx + block_size]
            if block.size == 0:
                continue
            # blok je neuniformní, pokud obsahuje alespoň jednu 0 i jednu 1
            if np.any(block == 0) and np.any(block == 1):
                nubn += 1
    return nubn if nubn > 0 else 1

def compute_drd_standard(pred: np.ndarray, gt: np.ndarray, weights: np.ndarray = WNM) -> float:
    """
    Implementace DRD podle definice v DIBCO.
    pred a gt musí být binární masky (0/1).
    1. Pro každý chybný pixel zjistíme jeho DRD_k: součet vah v 5×5 okně podle GT.
    2. Součet všech DRD_k dělíme počtem neuniformních bloků (NUBN).
    """
    pred = np.squeeze(pred).astype(np.uint8)
    gt   = np.squeeze(gt).astype(np.uint8)
    diff = (pred != gt).astype(np.uint8)

    # padneme GT, aby bylo možné vyříznout 5×5 okno kolem okrajů
    pad_gt = np.pad(gt, 2, mode="edge")
    error_coords = np.argwhere(diff == 1)

    drd_total = 0.0
    for y, x in error_coords:
        # 5×5 okno v GT
        window_gt = pad_gt[y:y + 5, x:x + 5]
        # GT v centru chyby (0 nebo 1)
        gt_center = gt[y, x]
        # DRD_k: sum_j W(j) * |window_gt(j) - gt_center|
        drd_k = np.sum(weights * np.abs(window_gt - gt_center))
        drd_total += drd_k

    nubn = count_nubn(gt, block_size=8)
    return drd_total / nubn


def estimate_conv2d_macs(conv: torch.nn.Module, input_shape: Tuple[int, int, int]) -> int:
    """Estimate multiply–accumulate (MAC) operations for a single Conv2d layer."""
    if not isinstance(conv, torch.nn.Conv2d):
        return 0
    C_in, H, W = input_shape
    C_out = conv.out_channels
    K_h, K_w = conv.kernel_size
    stride_h, stride_w = conv.stride
    pad_h, pad_w = conv.padding
    H_out = (H + 2 * pad_h - K_h) // stride_h + 1
    W_out = (W + 2 * pad_w - K_w) // stride_w + 1
    return H_out * W_out * C_out * K_h * K_w * C_in

def estimate_model_macs(model: torch.nn.Module, input_shape: Tuple[int, int, int] = (4, 200, 200)) -> int:
    """Estimate total MACs across Conv2d layers using a simple shape propagation."""
    total_macs = 0
    current_shape = input_shape
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            total_macs += estimate_conv2d_macs(layer, current_shape)
            H, W = current_shape[1], current_shape[2]
            K_h, K_w = layer.kernel_size
            stride_h, stride_w = layer.stride
            pad_h, pad_w = layer.padding
            H_out = (H + 2 * pad_h - K_h) // stride_h + 1
            W_out = (W + 2 * pad_w - K_w) // stride_w + 1
            current_shape = (layer.out_channels, H_out, W_out)
    return total_macs

def compute_psnr_batch(
    pred_batch: np.ndarray,
    tgt_batch: np.ndarray,
    *,
    threshold: float = 0.5,
    eps: float = 1e-10,
) -> float:
    """
    Spočítá PSNR mezi binárními maskami.
    
    - pred_batch: (B, H, W) – výstupy modelu v rozmezí [0, 1]
    - tgt_batch:  (B, H, W) – binární ground‑truth
    - threshold:  Prahovací hodnota; pixely > threshold jsou považovány za 1
    """
    psnr_values: List[float] = []
    for pred, tgt in zip(pred_batch, tgt_batch):
        pred_bin = (pred > threshold).astype(np.float32)
        tgt_bin = (tgt > threshold).astype(np.float32)
        mse = max(np.mean((pred_bin - tgt_bin) ** 2), eps)
        psnr_values.append(10.0 * math.log10(1.0 / mse))
    return float(np.mean(psnr_values))

def compute_drd(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute distance reciprocal distortion (DRD) for a single pair."""
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    diff = np.abs(pred - gt)
    conv = convolve2d(diff, _DRD_WEIGHTS, mode="same", boundary="symm")
    total_drd = np.sum(conv * diff)
    num_error_pixels = np.sum(diff)
    return 0.0 if num_error_pixels == 0 else total_drd / num_error_pixels

def compute_drd_batch(pred_batch: np.ndarray, tgt_batch: np.ndarray) -> float:
    return float(np.mean([
        compute_drd_standard(p, t)
        for p, t in zip(pred_batch, tgt_batch)
    ]))


def compute_p_fm(precision: float, recall: float, beta_squared: float = 0.3) -> float:
    """Compute pseudo-F measure (p-FM) from precision and recall."""
    if (precision + recall) == 0:
        return 0.0
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)

def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach and move to CPU numpy array."""
    return t.detach().cpu().numpy()

def convert_to_numpy(output: torch.Tensor, target: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Convert model outputs and targets to (B, H, W) numpy arrays."""
    if output.ndim == 4 and output.shape[1] == 1:
        output_np = _tensor_to_numpy(output.squeeze(1))
    else:
        output_np = _tensor_to_numpy(output)
    return output_np, _tensor_to_numpy(target)

# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------

def _forward_nca(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    pixel_weight: float,
    mode: str,
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """Forward pass for an NCA model with iterative state updates.

    Uses composite_loss with pixel-accuracy weighting when available,
    otherwise falls back to BCE from the legacy utilities.
    """
    loss = 0.0
    for _ in range(CAMODEL_CONFIG["steps"]):
        x = model(x)
        logits = x[:, 3]
        target = y.squeeze(1)
        if _HAVE_NEW_UTILS:
            loss += _composite_loss(logits, target, pixel_accuracy_weight=pixel_weight)
        else:
            loss += _compute_binary_cross_entropy(logits, target)
    correct, _ = compute_pixel_accuracy_fn(x[:, 3], y)
    return loss, correct, x[:, 3]

def _forward_generic(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_function: Optional[torch.nn.Module],
    activation_needed: bool = True,
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """Forward pass for segmentation models that output logits (or dicts)."""
    out = model(x)
    out = out["out"] if isinstance(out, dict) else out
    assert loss_function is not None, "loss_function must be provided for non-NCA models."
    loss = loss_function(out, y)
    correct, _ = compute_pixel_accuracy_fn(torch.sigmoid(out), y)
    return loss, correct, torch.sigmoid(out) if activation_needed else out

def forward_pass(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    model_name: str,
    loss_function: Optional[torch.nn.Module],
    mode: str = "train",
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """Dispatch to the appropriate forward pass and return (loss, acc, output)."""
    if model_name == "nca":
        pixel_weight = 1.0 if mode == "train" else 0.1
        return _forward_nca(model, x, y, pixel_weight, mode)
    if model_name == "deeplabv3":
        out_dict = model(x)
        logits = out_dict["out"]
        assert loss_function is not None, "loss_function must be provided for non-NCA models."
        loss = loss_function(logits, y)
        correct, _ = compute_pixel_accuracy_fn(torch.sigmoid(logits), y)
        return loss, correct, torch.sigmoid(logits)
    return _forward_generic(model, x, y, loss_function)

# ---------------------------------------------------------------------------
# Training/evaluation loop
# ---------------------------------------------------------------------------

def _run_epoch(
    *,
    model: torch.nn.Module,
    dataloader: Iterable,
    optimizer: Optional[torch.optim.Optimizer],
    loss_function: Optional[torch.nn.Module],
    device: torch.device,
    model_name: str,
    mode: str,
    epoch_idx: int,
    run_idx: int,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Run one pass over the dataset and compute metrics."""
    is_train = mode == "train"
    model.train() if is_train else model.eval()

    totals = {tag: 0.0 for tag in METRIC_TAGS}
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    # Show one visualization on first eval epoch of first run
    enable_plot = (not is_train) and (epoch_idx == 0) and (run_idx == 0)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in dataloader:
            x, y = x.float().to(device), y.float().to(device)

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad()

            loss, correct, output = forward_pass(
                model, x, y, model_name, loss_function, mode=mode
            )

            output_np, y_np = convert_to_numpy(output, y)
            batch_size = output_np.shape[0]

            totals["Loss"] += loss.item()
            totals["Accuracy"] += correct
            totals["PSNR"] += compute_psnr_batch(output_np, y_np) * batch_size

            binary_preds = (output_np > 0.5).astype(np.uint8)
            binary_targets = (y_np > 0.5).astype(np.uint8)
            totals["DRD"] += compute_drd_batch(binary_preds, binary_targets) * batch_size

            all_preds.extend(binary_preds.flatten())
            all_targets.extend(binary_targets.flatten())
            total_samples += batch_size

            if is_train:
                loss.backward()
                optimizer.step()

            if enable_plot:
                enable_plot = False
                inp = x[0]  # (C, H, W)
                gt = y[0, 0].cpu().numpy()
                raw_pred = output[0, 0].detach().cpu().numpy() if output.ndim == 4 else output[0].detach().cpu().numpy()
                bin_pred = (raw_pred > 0.5).astype(np.float32)
                diff = np.abs(bin_pred - gt)

                if inp.shape[0] == 3:
                    inp_img = inp.cpu().permute(1, 2, 0).numpy()
                    cmap_input = None
                else:
                    inp_img = inp[0].cpu().numpy()
                    cmap_input = "gray"

                fig, axs = plt.subplots(1, 5, figsize=(20, 4))

                axs[0].imshow(inp_img, cmap=cmap_input)
                axs[0].set_title("Input")

                axs[1].imshow(gt, cmap="gray")
                axs[1].set_title("Ground Truth")

                axs[2].imshow(raw_pred, cmap="gray")
                axs[2].set_title("Raw Prediction")

                axs[3].imshow(bin_pred, cmap="gray")
                axs[3].set_title("Binary Prediction (>0.5)")

                axs[4].imshow(diff, cmap="hot")
                axs[4].set_title("Difference (|Pred - GT|)")

                for ax in axs:
                    ax.axis("off")

                plt.tight_layout()
                plt.show()


            if enable_plot:
                enable_plot = False
                inp = x[0]
                gt = y[0, 0]
                pred_mask = output[0, 0] if output.ndim == 4 else output[0]
                if inp.shape[0] == 3:
                    inp_img = inp.cpu().permute(1, 2, 0).numpy()
                else:
                    inp_img = inp[0].cpu().numpy()
                gt_mask = gt.cpu().numpy()
                pred_img = pred_mask.detach().cpu().numpy()
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(inp_img if inp.shape[0] == 3 else inp_img, cmap=None if inp.shape[0] == 3 else "gray")
                ax[0].set_title("Input")
                ax[1].imshow(gt_mask, cmap="gray")
                ax[1].set_title("Ground Truth")
                ax[2].imshow(pred_img, cmap="gray")
                ax[2].set_title("Prediction")
                for a in ax:
                    a.axis("off")
                plt.tight_layout()
                plt.show()

    preds_array = np.array(all_preds)
    targets_array = np.array(all_targets)

    precision = precision_score(targets_array, preds_array, zero_division=0)
    recall = recall_score(targets_array, preds_array, zero_division=0)
    f1 = f1_score(targets_array, preds_array, zero_division=0)
    p_fm = compute_p_fm(precision, recall)

    totals.update(
        {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "FM": f1,  # to keep original contract
            "p-FM": p_fm,
            "PSNR": totals["PSNR"] / max(total_samples, 1),
            "DRD": totals["DRD"] / max(total_samples, 1),
        }
    )

    num_batches = max(len(dataloader), 1)
    totals["Loss"] /= num_batches
    totals["Accuracy"] /= num_batches

    return tuple(totals[tag] for tag in METRIC_TAGS)

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_function: Optional[torch.nn.Module],
    device: torch.device,
    model_name: str,
    epoch_idx: int,
    run_idx: int,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Public wrapper for a training epoch."""
    return _run_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        model_name=model_name,
        mode="train",
        epoch_idx=epoch_idx,
        run_idx=run_idx,
    )

def evaluate_model(
    model: torch.nn.Module,
    dataloader: Iterable,
    loss_function: Optional[torch.nn.Module],
    device: torch.device,
    model_name: str,
    epoch_idx: int,
    run_idx: int,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Public wrapper for an evaluation pass."""
    return _run_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=None,
        loss_function=loss_function,
        device=device,
        model_name=model_name,
        mode="eval",
        epoch_idx=epoch_idx,
        run_idx=run_idx,
    )

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _init_unet_tiny() -> torch.nn.Module:
    return UNetTiny(**UNET_CONFIG).to(DEVICE)

def _init_unet_normal() -> torch.nn.Module:
    return UNetNormal(**UNET_CONFIG).to(DEVICE)

def _init_nca() -> torch.nn.Module:
    return CAModel(**CAMODEL_CONFIG).to(DEVICE)

def _init_segnet() -> torch.nn.Module:
    return SegNet(**SEGNET_CONFIG).to(DEVICE)

def _init_deeplabv3() -> torch.nn.Module:
    model = deeplabv3_resnet50(**DEEPLABV3_CONFIG)
    model.backbone.conv1 = torch.nn.Conv2d(**DEEPLABV3_CONFIG_CONV1)
    return model.to(DEVICE)

def _init_pspnet() -> torch.nn.Module:
    return PSPNet(**PSPNET_CONFIG).to(DEVICE)

_MODEL_FACTORY: Dict[str, Any] = {
    "unet_tiny": _init_unet_tiny,
    "unet_normal": _init_unet_normal,
    "nca": _init_nca,
    "segnet": _init_segnet,
    "deeplabv3": _init_deeplabv3,
    "pspnet": _init_pspnet,
}

def initialize_model(model_name: str) -> torch.nn.Module:
    """Instantiate a model given its name."""
    try:
        return _MODEL_FACTORY[model_name]()
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc

def _resolve_shape_and_classes(model_name: str) -> Tuple[int, int]:
    """Determine the number of input channels and classes for a model."""
    if model_name in {"unet_tiny", "unet_normal"}:
        cfg = UNET_CONFIG
    elif model_name == "nca":
        cfg = {"n_channels": CAMODEL_CONFIG["n_channels"], "n_classes": 1}
    elif model_name == "segnet":
        cfg = SEGNET_CONFIG
    elif model_name == "deeplabv3":
        cfg = {
            "n_channels": DEEPLABV3_CONFIG["n_channels"],
            "n_classes": DEEPLABV3_CONFIG["num_classes"],
        }
    elif model_name == "pspnet":
        cfg = {
            "n_channels": PSPNET_CONFIG["in_channels"],
            "n_classes": PSPNET_CONFIG["classes"],
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return cfg["n_channels"], cfg["n_classes"]

def _prepare_loaders(
    *,
    n_channels: int,
    batch_size: int,
    crop_size: int,
    filter_masks: bool,
) -> Tuple[Iterable, Iterable]:
    """Create train and test dataloaders with optional dataset splitting."""
    cfg = TRAINING_CONFIG
    train_images_dir = cfg["train_images_dir"]
    train_masks_dir = cfg["train_masks_dir"]
    test_images_dir = cfg["test_images_dir"]
    test_masks_dir = cfg["test_masks_dir"]
    dataset_name = cfg.get("dataset_name", "")

    train_loader = get_dataloader_fn(
        train_images_dir,
        train_masks_dir,
        n_channels,
        batch_size,
        crop_size,
        filter_masks,
    )

    # Special 50/50 split for certain datasets
    if dataset_name in {"dibco", "trees"}:
        images_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []
        for imgs, msks in train_loader:
            images_list.append(imgs)
            masks_list.append(msks)
        images_all = torch.cat(images_list)
        masks_all = torch.cat(masks_list)
        total_samples = len(images_all)
        indices = torch.randperm(total_samples)
        split = int(0.5 * total_samples)
        train_subset = TensorDataset(
            images_all[indices[:split]], masks_all[indices[:split]]
        )
        test_subset = TensorDataset(
            images_all[indices[split:]], masks_all[indices[split:]]
        )
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = get_dataloader_fn(
            test_images_dir,
            test_masks_dir,
            n_channels,
            batch_size,
            crop_size,
            filter_masks,
        )
    return train_loader, test_loader

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(
    *,
    model_name: str = "nca",
    run: int = 0,
    steps: int = 1,
) -> Tuple[Tuple[Any, Any, int], Dict[str, Dict[str, List[float]]]]:
    """Top-level training routine for a single run."""
    start_time = time.time()

    if model_name == "nca":
        CAMODEL_CONFIG["steps"] = steps

    n_channels, _ = _resolve_shape_and_classes(model_name)
    cfg = TRAINING_CONFIG
    model = initialize_model(model_name)

    # Print model summary for non-Deeplab models
    if model_name != "deeplabv3":
        try:
            from torchsummary import summary
            summary(model, (n_channels, 200, 200))
        except Exception:
            pass

    optimizer = cfg["optimizer"](model.parameters(), lr=cfg["learning_rate"])
    scheduler = cfg["scheduler"](
        optimizer,
        step_size=cfg["scheduler_step_size"],
        gamma=cfg["scheduler_gamma"],
    )
    loss_function = (
        cfg["loss_function"]
        if model_name in {"unet_tiny", "unet_normal", "segnet", "deeplabv3", "pspnet"}
        else None
    )

    filter_masks = cfg.get("dataset_name", "") == "dibco"

    train_loader, test_loader = _prepare_loaders(
        n_channels=n_channels,
        batch_size=cfg["batch_size"],
        crop_size=200,
        filter_masks=filter_masks,
    )

    run_tag = f"{model_name}_steps_{steps}_run_{run + 1}"
    writer = SummaryWriter(Path("runs") / run_tag)

    epoch_curve: Dict[str, Dict[str, List[float]]] = {
        "Train": {tag: [] for tag in METRIC_TAGS},
        "Test": {tag: [] for tag in METRIC_TAGS},
    }
    best_loss = float("inf")
    best_train = best_test = None
    best_epoch = -1

    for epoch in range(cfg["epochs"]):
        print(f"Run {run + 1} | Epoch {epoch + 1}/{cfg['epochs']} | steps: {steps}")

        if epoch == 0:
            init_metrics = evaluate_model(
                model, test_loader, loss_function, DEVICE, model_name, epoch, run
            )
            for i, tag in enumerate(METRIC_TAGS):
                writer.add_scalar(f"Test/{tag}", init_metrics[i], epoch)
            print(f"Initial Test: Loss={init_metrics[0]:.4f}, Acc={init_metrics[1]:.2f}%")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            DEVICE,
            model_name,
            epoch,
            run,
        )

        test_metrics = evaluate_model(
            model, test_loader, loss_function, DEVICE, model_name, epoch, run
        )

        if test_metrics[0] < best_loss:
            best_loss = test_metrics[0]
            best_train = train_metrics
            best_test = test_metrics
            best_epoch = epoch

        for i, tag in enumerate(METRIC_TAGS):
            writer.add_scalar(f"Train/{tag}", train_metrics[i], epoch)
            writer.add_scalar(f"Test/{tag}", test_metrics[i], epoch)
            epoch_curve["Train"][tag].append(train_metrics[i])
            epoch_curve["Test"][tag].append(test_metrics[i])

        model_config: Dict[str, Any] = {}
        if model_name == "nca":
            model_config = {
                "n_channels": CAMODEL_CONFIG["n_channels"],
                "hidden_channels": CAMODEL_CONFIG["hidden_channels"],
                "deep_perceive": CAMODEL_CONFIG["deep_perceive"],
                "deep_update": CAMODEL_CONFIG["deep_update"],
                "use_residual": CAMODEL_CONFIG["use_residual"],
                "fire_rate": CAMODEL_CONFIG["fire_rate"],
                "neighbour": CAMODEL_CONFIG["neighbour"],
                "steps": CAMODEL_CONFIG["steps"],
            }

        # Persist metrics using compat wrapper
        _log_results(
            epoch,
            dict(zip(METRIC_TAGS, train_metrics)),
            dict(zip(METRIC_TAGS, test_metrics)),
            f"{model_name}_steps_{steps}",
            run,
            model_config=model_config,
        )

        scheduler.step()

    writer.close()

    # Save model weights
    os.makedirs("models", exist_ok=True)
    model_path = f"models/mynet_weights_small_run{run}.pt"
    torch.save(model.state_dict(), model_path)

    # HW/model stats
    macs = estimate_model_macs(model, input_shape=(CAMODEL_CONFIG.get("n_channels", n_channels), 200, 200))
    model_stats = {
        "model_name": model_name,
        "macs": macs,
        "adds": 0,
        "total_ops": macs,
        "model_size_MB": os.path.getsize(model_path) / (1024 ** 2),
        "training_time": f"{(time.time() - start_time):.2f}s",
        "hardware_target": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    _log_hw(run_id=run, model_name=model_name, model_stats=model_stats)

    return (best_test, best_train, best_epoch), epoch_curve

# ---------------------------------------------------------------------------
# Standalone experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Run experiments with varying NCA steps and multiple runs."""
    STEPS_VARIANTS: List[int] = [1]
    N_RUNS = 2
    final_results: Dict[int, Dict[str, List[Any]]] = {
        s: {"final": [], "curves": []} for s in STEPS_VARIANTS
    }
    print("UTILS :" + str(_HAVE_NEW_UTILS))
    for steps in STEPS_VARIANTS:
        CAMODEL_CONFIG["steps"] = steps
        for run in range(N_RUNS):
            print(f"\nRunning experiment: steps={steps}, run={run + 1}")
            (best_test, best_train, best_epoch), curve = main(model_name="nca", run=run, steps=steps)
            final_results[steps]["final"].append((best_train, best_test, best_epoch))
            final_results[steps]["curves"].append(curve)

    writer_avg = SummaryWriter(Path("runs") / "avg_nca")
    for steps, data in final_results.items():
        curves_list = data["curves"]
        total_epochs = len(curves_list[0]["Test"]["Loss"])
        for epoch in range(total_epochs):
            for tag in METRIC_TAGS:
                avg_train = float(np.mean([c["Train"][tag][epoch] for c in curves_list]))
                avg_test = float(np.mean([c["Test"][tag][epoch] for c in curves_list]))
                writer_avg.add_scalar(f"Avg/steps_{steps}/Train/{tag}", avg_train, epoch)
                writer_avg.add_scalar(f"Avg/steps_{steps}/Test/{tag}", avg_test, epoch)
    writer_avg.close()

    for steps, data in final_results.items():
        run_results = data["final"]
        best_idx = int(np.argmin([res[1][0] for res in run_results]))  # min Test Loss
        best_train, best_test, best_epoch = run_results[best_idx]
        print(f"\n=== BEST RESULTS for steps={steps} (run {best_idx + 1}) ===")
        print(
            (
                "Test Loss: {0:.4f}, Acc: {1:.2f}%, Precision: {2:.4f}, "
                "Recall: {3:.4f}, F1: {4:.4f}, PSNR: {5:.4f}, DRD: {6:.4f}"
            ).format(
                best_test[0],
                best_test[1],
                best_test[2],
                best_test[3],
                best_test[4],
                best_test[7],
                best_test[8],
            )
        )
        print(
            (
                "Train Loss at best epoch: {0:.4f}, Acc: {1:.2f}% (epoch {2})"
            ).format(best_train[0], best_train[1], best_epoch)
        )
