"""
Combined training script that merges the functionality of ``train.py`` and
``train_steps.py`` from the original project.  This script preserves all
capabilities from both sources: it supports a range of segmentation and
neural cellular automata (NCA) models, implements both the simple
training/evaluation loops found in ``train.py`` as well as the richer
metric tracking and hardware logging introduced in ``train_steps.py``, and
offers flexible experiment orchestration (multiple runs, varying NCA
steps, optional dataset splitting for specific datasets, tensorboard
logging, MACs/parameter estimation, etc.).

The core idea behind the merge is to unify common logic (model
initialisation, forward passes, metric computation, data loading) and
expose a single ``main`` entrypoint that can be used for simple
training runs or more elaborate experiments.  Where the two original
scripts diverged—such as in the handling of additional metrics like
PSNR/DRD, or the special splitting behaviour for certain datasets—those
branches are now incorporated into a single training loop so that
nothing is lost.  The script remains self contained: it depends only
on ``config.py``, ``model.py`` and ``utils.py`` from the original
repository, and can be executed as a standalone module.
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
from utils import (
    calculate_pixel_accuracy,
    composite_loss,
    get_dataloader,
    log_loss,
    log_hw_metrics,
    plot_batch_x_channels,
)

# ---------------------------------------------------------------------------
# Seeding & multiprocessing configuration
# ---------------------------------------------------------------------------

# Set random seeds for reproducibility across Python, NumPy and PyTorch.
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Ensure that the multiprocessing start method is explicitly set.  Without
# this call, PyTorch may default to 'fork' on Unix which can cause
# subtle bugs when spawning worker processes.
mp.set_start_method("spawn", force=True)

# ---------------------------------------------------------------------------
# Metric definitions and helper functions
# ---------------------------------------------------------------------------

# Names of the metrics returned from ``_run_epoch``.  The order here
# matches the order in which the metrics are computed and returned.
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

# Pre-compute a 5×5 weight matrix used for DRD (distance reciprocal
# distortion) computation.  The centre of the matrix has the largest
# weight and the contribution decays with Manhattan distance from the
# centre.  This definition matches the one found in the original
# ``train_steps.py``.
_DRD_WEIGHTS = np.array(
    [[1.0 / (abs(i - 2) + abs(j - 2) + 1) for j in range(5)] for i in range(5)]
)

def estimate_conv2d_macs(conv: torch.nn.Module, input_shape: Tuple[int, int, int]) -> int:
    """Estimate multiply–accumulate (MAC) operations for a single Conv2D layer.

    Args:
        conv: The convolutional layer to analyse.
        input_shape: Shape of the input tensor (C_in, H, W).

    Returns:
        Estimated number of MACs performed by the convolution.
    """
    if not isinstance(conv, torch.nn.Conv2d):
        return 0
    C_in, H, W = input_shape
    C_out = conv.out_channels
    K_h, K_w = conv.kernel_size
    # Output spatial dimensions
    H_out = (H + 2 * conv.padding[0] - K_h) // conv.stride[0] + 1
    W_out = (W + 2 * conv.padding[1] - K_w) // conv.stride[1] + 1
    # MACs = number of output elements × kernel area × C_in
    return H_out * W_out * C_out * K_h * K_w * C_in

def estimate_model_macs(model: torch.nn.Module, input_shape: Tuple[int, int, int] = (4, 200, 200)) -> int:
    """Estimate the total MACs for all Conv2D layers in a model.

    Args:
        model: The neural network whose MACs should be estimated.
        input_shape: Shape of the input tensor passed to the first layer.

    Returns:
        Total number of MACs.
    """
    total_macs = 0
    current_shape = input_shape
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            macs = estimate_conv2d_macs(layer, current_shape)
            total_macs += macs
            # Update shape for the next layer.  This is a simplified
            # propagation that assumes stride and padding remain
            # constant across width/height.
            H, W = current_shape[1], current_shape[2]
            H_out = (H + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
            W_out = (W + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
            current_shape = (layer.out_channels, H_out, W_out)
    return total_macs

def compute_psnr_batch(pred_batch: np.ndarray, tgt_batch: np.ndarray, eps: float = 1e-10) -> float:
    """Compute average PSNR for a batch of predictions and targets.

    Args:
        pred_batch: Predicted images with values in [0, 1].  Shape (B, H, W).
        tgt_batch: Ground truth images with values in [0, 1].  Shape (B, H, W).
        eps: Small constant to avoid division by zero.

    Returns:
        The average PSNR over the batch.
    """
    psnr_values: List[float] = []
    for pred, tgt in zip(pred_batch, tgt_batch):
        mse = max(np.mean((pred - tgt) ** 2), eps)
        psnr_values.append(10 * math.log10(1.0 / mse))
    return float(np.mean(psnr_values))

def compute_drd(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute the distance reciprocal distortion (DRD) for a single pair.

    Args:
        pred: Binary prediction mask (H, W).
        gt: Binary ground truth mask (H, W).

    Returns:
        The DRD metric value.
    """
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    diff = np.abs(pred - gt)
    conv = convolve2d(diff, _DRD_WEIGHTS, mode="same", boundary="symm")
    total_drd = np.sum(conv * diff)
    num_error_pixels = np.sum(diff)
    return 0.0 if num_error_pixels == 0 else total_drd / num_error_pixels

def compute_drd_batch(pred_batch: np.ndarray, tgt_batch: np.ndarray) -> float:
    """Compute the average DRD across a batch.

    Args:
        pred_batch: Batch of binary predictions (B, H, W).
        tgt_batch: Batch of binary ground truth masks (B, H, W).

    Returns:
        The mean DRD over the batch.
    """
    return float(np.mean([compute_drd(p, t) for p, t in zip(pred_batch, tgt_batch)]))

def compute_p_fm(precision: float, recall: float, beta_squared: float = 0.3) -> float:
    """Compute the pseudo-F measure (p-FM) given precision and recall.

    When precision and recall are both zero, returns zero to avoid
    division by zero.  Otherwise computes the weighted harmonic mean.

    Args:
        precision: Precision value.
        recall: Recall value.
        beta_squared: Weighting factor (beta^2) controlling the trade-off
            between precision and recall.  A lower beta_squared places more
            emphasis on precision.

    Returns:
        The p-FM score.
    """
    if (precision + recall) == 0:
        return 0.0
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)

def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach and transfer a torch Tensor to CPU NumPy array."""
    return t.detach().cpu().numpy()

def convert_to_numpy(output: torch.Tensor, target: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Convert network outputs and targets to NumPy arrays for metric computations.

    For single-channel outputs (e.g. segmentation masks), the channel
    dimension is squeezed to produce a (B, H, W) array.

    Args:
        output: Output tensor from the model (B, C, H, W) or (B, H, W).
        target: Target tensor (B, 1, H, W) or (B, H, W).

    Returns:
        A tuple of (output_np, target_np) both shaped (B, H, W).
    """
    if output.ndim == 4 and output.shape[1] == 1:
        output_np = _tensor_to_numpy(output.squeeze(1))
    else:
        output_np = _tensor_to_numpy(output)
    return output_np, _tensor_to_numpy(target)

def _forward_nca(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    pixel_weight: float,
    mode: str,
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """Forward pass for an NCA model with iterative state updates.

    Args:
        model: The NCA model implementing a step-based update.
        x: Input tensor representing the current state.  Shape (B, C, H, W).
        y: Target tensor.  Shape (B, 1, H, W).
        pixel_weight: Weight applied to the pixel-accuracy term in the loss.
        mode: Either "train" or "eval", used to gate optional visualisation.

    Returns:
        A tuple of (loss, correct_pixels, activated_output), where
        ``activated_output`` is passed through a sigmoid.
    """
    loss = 0.0
    for _ in range(CAMODEL_CONFIG["steps"]):
        x = model(x)
        loss += composite_loss(x[:, 3], y.squeeze(1), pixel_accuracy_weight=pixel_weight)
    correct, _ = calculate_pixel_accuracy(x[:, 3], y)
    return loss, correct, torch.sigmoid(x[:, 3])

def _forward_generic(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_function: torch.nn.Module,
    activation_needed: bool = True,
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """Forward pass for segmentation models that output raw logits.

    Some segmentation models (e.g. torchvision's DeepLabV3) return a
    dictionary with the logits under the key ``"out"``.  This helper
    abstracts over that difference and always returns a tensor of logits.

    Args:
        model: The segmentation model.
        x: Input tensor.  Shape (B, C, H, W).
        y: Target tensor.  Shape (B, 1, H, W).
        loss_function: Loss function to apply to the logits and targets.
        activation_needed: Whether to apply a sigmoid activation to the
            output before returning it.  This should generally be ``True``
            for segmentation models producing per-pixel logits.

    Returns:
        A tuple of (loss, correct_pixels, activated_output_or_logits).
    """
    out = model(x)
    # Handle models returning a dict (DeepLabV3)
    out = out["out"] if isinstance(out, dict) else out
    loss = loss_function(out, y)
    correct, _ = calculate_pixel_accuracy(torch.sigmoid(out), y)
    return loss, correct, torch.sigmoid(out) if activation_needed else out

def forward_pass(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    model_name: str,
    loss_function: torch.nn.Module | None,
    mode: str = "train",
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """Compute a single forward pass and return loss, accuracy and output.

    Dispatches to the appropriate helper depending on the model type.

    Args:
        model: The neural network.
        x: Input tensor.
        y: Target tensor.
        model_name: A string identifying the model type.  Must be one of
            ``"nca"``, ``"deeplabv3"`` or any other supported model.
        loss_function: Loss function used by models other than NCA.
        mode: Either "train" or "eval".

    Returns:
        A tuple (loss, correct_pixels, activated_output).
    """
    if model_name == "nca":
        # For NCA we modulate the pixel_accuracy_weight depending on
        # training/evaluation.  During training we place more weight on
        # pixel accuracy; during evaluation we reduce it as done in
        # ``train.py``.
        pixel_weight = 1.0 if mode == "train" else 0.1
        return _forward_nca(model, x, y, pixel_weight, mode)
    if model_name == "deeplabv3":
        # torchvision models return a dict
        out_dict = model(x)
        loss = loss_function(out_dict["out"], y)
        correct, _ = calculate_pixel_accuracy(torch.sigmoid(out_dict["out"]), y)
        return loss, correct, torch.sigmoid(out_dict["out"])
    # All other models: raw logits are exposed directly
    return _forward_generic(model, x, y, loss_function)

def _run_epoch(
    *,
    model: torch.nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer | None,
    loss_function: torch.nn.Module | None,
    device: torch.device,
    model_name: str,
    mode: str,
    epoch_idx: int,
    run_idx: int,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Generic routine for running one pass over the dataset.

    Whether called for training or evaluation, this function computes all
    metrics defined in ``METRIC_TAGS``, accumulates them across batches,
    and returns a tuple with the values in the exact order of
    ``METRIC_TAGS``.  When training, it also performs backpropagation and
    optimizer updates.  The ``epoch_idx`` and ``run_idx`` arguments are
    used to gate optional debug visualisations so that they occur on a
    predictable schedule instead of never being triggered.

    Args:
        model: The neural network.
        dataloader: Iterable yielding batches of (images, masks).
        optimizer: Optimizer to update the model parameters.  Should be
            ``None`` when ``mode`` is "eval".
        loss_function: Loss function used for non-NCA models.  Pass
            ``None`` for NCA.
        device: Target device (CPU or CUDA).
        model_name: Name of the model.
        mode: Either "train" or "eval".
        epoch_idx: Index of the current epoch (0-based).
        run_idx: Index of the current run (0-based).

    Returns:
        A tuple containing the metrics in the order specified by
        ``METRIC_TAGS``.
    """
    is_train = mode == "train"
    if is_train:
        model.train()
    else:
        model.eval()
    totals = {tag: 0.0 for tag in METRIC_TAGS}
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []
    # Choose whether to enable debug plotting on this epoch.  In the original
    # ``train.py`` evaluate_model would show plots when a global counter
    # ``PICA`` equalled 2.  Here we reproduce similar behaviour by
    # enabling plotting on the second evaluation epoch of the first run.
    enable_plot = (not is_train) and (epoch_idx == 0) and (run_idx == 0)
    # Disable gradient tracking when evaluating
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in dataloader:
            x, y = x.float().to(device), y.float().to(device)
            if is_train:
                optimizer.zero_grad()
            loss, correct, output = forward_pass(
                model, x, y, model_name, loss_function, mode=mode
            )
            output_np, y_np = convert_to_numpy(output, y)
            batch_size = output_np.shape[0]
            # Aggregate scalar metrics
            totals["Loss"] += loss.item()
            totals["Accuracy"] += correct
            totals["PSNR"] += compute_psnr_batch(output_np, y_np) * batch_size
            # Binarize predictions/targets for classification-style metrics
            binary_preds = (output_np > 0.5).astype(np.uint8)
            binary_targets = (y_np > 0.5).astype(np.uint8)
            totals["DRD"] += compute_drd_batch(binary_preds, binary_targets) * batch_size
            all_preds.extend(binary_preds.flatten())
            all_targets.extend(binary_targets.flatten())
            total_samples += batch_size
            if is_train:
                loss.backward()
                optimizer.step()
            # Optional visualisation: on a fixed schedule plot the first
            # sample in the batch.  This reproduces the behaviour of the
            # ``train.py`` script without requiring an always-false random
            # condition.
            if enable_plot:
                # Only plot once per epoch
                enable_plot = False
                inp = x[0]
                gt = y[0, 0]
                pred_mask = output[0, 0] if output.ndim == 4 else output[0]
                # Detach & convert to NumPy
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
    # Compute classification metrics once per epoch
    preds_array = np.array(all_preds)
    targets_array = np.array(all_targets)
    precision = precision_score(targets_array, preds_array)
    recall = recall_score(targets_array, preds_array)
    f1 = f1_score(targets_array, preds_array)
    p_fm = compute_p_fm(precision, recall)
    # Update totals with classification metrics (FM duplicates F1 to keep original contract)
    totals.update(
        {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "FM": f1,
            "p-FM": p_fm,
            "PSNR": totals["PSNR"] / total_samples,
            "DRD": totals["DRD"] / total_samples,
        }
    )
    # Normalize accumulated loss and accuracy across batches
    num_batches = len(dataloader)
    totals["Loss"] /= num_batches
    totals["Accuracy"] /= num_batches
    return tuple(totals[tag] for tag in METRIC_TAGS)

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module | None,
    device: torch.device,
    model_name: str,
    epoch_idx: int,
    run_idx: int,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Public wrapper for a training epoch.  Delegates to ``_run_epoch``."""
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
    loss_function: torch.nn.Module | None,
    device: torch.device,
    model_name: str,
    epoch_idx: int,
    run_idx: int,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Public wrapper for an evaluation pass.  Delegates to ``_run_epoch``."""
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
    # Replace the first convolution to support arbitrary input channels
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
    """Instantiate a model given its name.

    Args:
        model_name: Key in ``_MODEL_FACTORY``.

    Returns:
        The initialised model moved to ``DEVICE``.

    Raises:
        ValueError: If the model name is not recognised.
    """
    try:
        return _MODEL_FACTORY[model_name]()
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc

def _resolve_shape_and_classes(model_name: str) -> Tuple[int, int]:
    """Determine the number of input channels and classes for a model.

    Returns a tuple ``(n_channels, n_classes)`` appropriate for creating
    synthetic inputs when calling ``torchsummary.summary``.  For NCA we
    treat it as a single-class problem even though the CAModel itself
    maintains multiple state channels.
    """
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
    """Create train and test dataloaders with optional dataset splitting.

    For certain datasets (e.g. ``dibco`` or ``trees``) the original
    ``train.py`` script loaded the entire training set and then split it
    into 50% train and 50% test.  To preserve this behaviour the
    function checks the ``dataset_name`` entry in ``TRAINING_CONFIG`` and
    performs the split if necessary; otherwise it creates separate
    dataloaders using the train/test directories from ``TRAINING_CONFIG``.

    Args:
        n_channels: Number of channels in the input images.
        batch_size: Batch size for loading.
        crop_size: Crop size used by ``get_dataloader``.
        filter_masks: Whether to filter out empty masks.

    Returns:
        A tuple (train_loader, test_loader).
    """
    cfg = TRAINING_CONFIG
    train_images_dir = cfg["train_images_dir"]
    train_masks_dir = cfg["train_masks_dir"]
    test_images_dir = cfg["test_images_dir"]
    test_masks_dir = cfg["test_masks_dir"]
    dataset_name = cfg.get("dataset_name", "")
    # Primary train loader
    train_loader = get_dataloader(
        train_images_dir,
        train_masks_dir,
        n_channels,
        batch_size,
        crop_size,
        filter_masks,
    )
    # For certain datasets we split the training set ourselves.  This
    # behaviour originated in ``train.py`` and is retained here.  We
    # accumulate all samples from the loader, shuffle and split 50/50.
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
        test_loader = get_dataloader(
            test_images_dir,
            test_masks_dir,
            n_channels,
            batch_size,
            crop_size,
            filter_masks,
        )
    return train_loader, test_loader

def main(
    *,
    model_name: str = "nca",
    run: int = 0,
    steps: int = 1,
) -> Tuple[Tuple[Any, Any], Dict[str, Dict[str, List[float]]]]:
    """Top-level training routine for a single run.

    This function orchestrates the complete training and evaluation cycle
    for one configuration of model, run index and number of NCA steps.
    It mirrors the logic found in both original scripts: it instantiates
    the model, prepares the data loaders (including dataset splitting
    where appropriate), runs through the specified number of epochs
    collecting detailed metrics, logs results via TensorBoard and
    ``log_loss``, saves model weights and logs hardware metrics.

    Args:
        model_name: Name of the model architecture to train.
        run: Index of the current run (0-based).  Used for naming
            output directories and files.
        steps: Number of steps to update the NCA (only relevant when
            ``model_name == 'nca'``).

    Returns:
        A tuple ``((best_test, best_train), epoch_curve)`` where
        ``best_test`` and ``best_train`` are the metric tuples
        corresponding to the best test loss encountered during training,
        and ``epoch_curve`` is a dictionary mapping ``"Train"`` and
        ``"Test"`` to per-epoch metric histories.
    """
    start_time = time.time()
    # Adjust global configuration if running an NCA experiment
    if model_name == "nca":
        CAMODEL_CONFIG["steps"] = steps
    n_channels, _ = _resolve_shape_and_classes(model_name)
    cfg = TRAINING_CONFIG
    model = initialize_model(model_name)
    # Print model summary for non-Deeplab models to aid debugging
    if model_name != "deeplabv3":
        try:
            from torchsummary import summary
            summary(model, (n_channels, 200, 200))
        except Exception:
            # ``torchsummary`` might not be available; ignore if so
            pass
    optimizer = cfg["optimizer"](model.parameters(), lr=cfg["learning_rate"])
    scheduler = cfg["scheduler"](
        optimizer,
        step_size=cfg["scheduler_step_size"],
        gamma=cfg["scheduler_gamma"],
    )
    # Loss function is only needed for non-NCA models
    loss_function = (
        cfg["loss_function"]
        if model_name in {"unet_tiny", "unet_normal", "segnet", "deeplabv3", "pspnet"}
        else None
    )
    # Determine whether to filter empty masks
    filter_masks = cfg.get("dataset_name", "") == "dibco"
    # Prepare loaders, possibly splitting the dataset
    train_loader, test_loader = _prepare_loaders(
        n_channels=n_channels,
        batch_size=cfg["batch_size"],
        crop_size=200,
        filter_masks=filter_masks,
    )
    # TensorBoard writer
    run_tag = f"{model_name}_steps_{steps}_run_{run + 1}"
    writer = SummaryWriter(Path("runs") / run_tag)
    # Data structure to store per-epoch metrics for plotting later
    epoch_curve: Dict[str, Dict[str, List[float]]] = {
        "Train": {tag: [] for tag in METRIC_TAGS},
        "Test": {tag: [] for tag in METRIC_TAGS},
    }
    best_loss = float("inf")
    best_train = best_test = None
    best_epoch = None
    for epoch in range(cfg["epochs"]):
        print(f"Run {run + 1} | Epoch {epoch + 1}/{cfg['epochs']} | steps: {steps}")
        # Initial evaluation on the test set before any training occurs
        if epoch == 0:
            init_metrics = evaluate_model(
                model, test_loader, loss_function, DEVICE, model_name, epoch, run
            )
            for i, tag in enumerate(METRIC_TAGS):
                writer.add_scalar(f"Test/{tag}", init_metrics[i], epoch)
            print(
                f"Initial Test: Loss={init_metrics[0]:.4f}, Acc={init_metrics[1]:.2f}%"
            )
        # Training pass
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
        # Evaluation pass
        test_metrics = evaluate_model(
            model, test_loader, loss_function, DEVICE, model_name, epoch, run
        )
        # Track best model according to test loss
        if test_metrics[0] < best_loss:
            best_loss = test_metrics[0]
            best_train = train_metrics
            best_test = test_metrics
            best_epoch = epoch
        # Log metrics to TensorBoard and in-memory curves
        for i, tag in enumerate(METRIC_TAGS):
            writer.add_scalar(f"Train/{tag}", train_metrics[i], epoch)
            writer.add_scalar(f"Test/{tag}", test_metrics[i], epoch)
            epoch_curve["Train"][tag].append(train_metrics[i])
            epoch_curve["Test"][tag].append(test_metrics[i])
        # Construct a model_config dict only for NCA models to log
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
        # Persist results using the provided ``log_loss`` helper.  The
        # helper expects dictionaries mapping metric names to values.
        log_loss(
            epoch,
            dict(zip(METRIC_TAGS, train_metrics)),
            dict(zip(METRIC_TAGS, test_metrics)),
            f"{model_name}_steps_{steps}",
            run,
            model_config=model_config,
            print_results=True,
        )
        scheduler.step()
    writer.close()
    # Save model weights.  Use the same filename pattern as in the
    # original scripts.
    os.makedirs("models", exist_ok=True)
    model_path = f"models/mynet_weights_small_run{run}.pt"
    torch.save(model.state_dict(), model_path)
    # Compute MACs and other hardware statistics
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
    log_hw_metrics(run_id=run, model_name=model_name, model_stats=model_stats)
    # Return best metrics and entire curve for further analysis
    return (best_test, best_train), epoch_curve

if __name__ == "__main__":
    """Run experiments with varying NCA steps and multiple runs.

    The main block here mirrors the standalone invocation logic from
    ``train_steps.py``.  It iterates over a list of step values and a
    number of runs per step, aggregates final results, averages
    epoch-wise curves and prints a summary of the best results.  For
    simplicity the step variants and number of runs can be adjusted
    directly in the lists below.
    """
    # Define the range of NCA step counts to experiment with.  Feel free
    # to modify this list to explore different step counts.  For non-NCA
    # models the ``steps`` argument has no effect.
    STEPS_VARIANTS: List[int] = [1]
    # Number of independent runs per step count.  Increasing this can
    # smooth out random variation in training outcomes.
    N_RUNS = 2
    final_results: Dict[int, Dict[str, List[Any]]] = {
        s: {"final": [], "curves": []} for s in STEPS_VARIANTS
    }
    for steps in STEPS_VARIANTS:
        CAMODEL_CONFIG["steps"] = steps
        for run in range(N_RUNS):
            print(f"\nRunning experiment: steps={steps}, run={run + 1}")
            (best_test, best_train), curve = main(model_name="nca", run=run, steps=steps)
            final_results[steps]["final"].append((best_train, best_test))
            final_results[steps]["curves"].append(curve)
    # Average curves across runs and log to a separate writer
    writer_avg = SummaryWriter(Path("runs") / "avg_nca")
    for steps, data in final_results.items():
        curves_list = data["curves"]
        total_epochs = len(curves_list[0]["Test"]["Loss"])
        for epoch in range(total_epochs):
            for tag in METRIC_TAGS:
                avg_train = float(
                    np.mean([c["Train"][tag][epoch] for c in curves_list])
                )
                avg_test = float(
                    np.mean([c["Test"][tag][epoch] for c in curves_list])
                )
                writer_avg.add_scalar(f"Avg/steps_{steps}/Train/{tag}", avg_train, epoch)
                writer_avg.add_scalar(f"Avg/steps_{steps}/Test/{tag}", avg_test, epoch)
    writer_avg.close()
    # Display best results per steps value
    for steps, data in final_results.items():
        run_results = data["final"]
        # Index of the run with minimum test loss
        best_idx = int(np.argmin([res[1][0] for res in run_results]))
        best_train, best_test = run_results[best_idx]
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
            ).format(best_train[0], best_train[1], best_epoch if 'best_epoch' in locals() else '?')
        )