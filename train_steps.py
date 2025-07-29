"""
 train_steps_refactored.py
 Refactored version of the original *train_steps.py* for improved readability and maintainability.
 Functionality is **identical** to the original script.
"""

from __future__ import annotations

import math
import os
import csv
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.signal import convolve2d
from sklearn.metrics import f1_score, precision_score, recall_score
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

def estimate_conv2d_macs(conv, input_shape):
    """
    Estimate MACs for a Conv2D layer given input shape (C_in, H, W).
    Returns MACs as an integer.
    """
    if not isinstance(conv, torch.nn.Conv2d):
        return 0
    C_in, H, W = input_shape
    C_out = conv.out_channels
    K_h, K_w = conv.kernel_size
    # MACs = H_out * W_out * C_out * K_h * K_w * C_in
    H_out = (H + 2 * conv.padding[0] - K_h) // conv.stride[0] + 1
    W_out = (W + 2 * conv.padding[1] - K_w) // conv.stride[1] + 1
    macs = H_out * W_out * C_out * K_h * K_w * C_in
    return macs

def estimate_model_macs(model: torch.nn.Module, input_shape=(4, 200, 200)):
    """
    Go through all Conv2D layers and estimate total MACs.
    """
    total_macs = 0
    print(input_shape)
    current_shape = input_shape  # Start with (C_in, H, W)

    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            macs = estimate_conv2d_macs(layer, current_shape)
            total_macs += macs
            # Update shape for next layer (simplified)
            H, W = current_shape[1], current_shape[2]
            H_out = (H + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
            W_out = (W + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
            current_shape = (layer.out_channels, H_out, W_out)
    return total_macs


# ---------------------------------------------------------------------------
# Reproducibility & multiprocessing setup
# ---------------------------------------------------------------------------

torch.manual_seed(SEED)
# Ensures deterministic behavior for each GPU
torch.cuda.manual_seed_all(SEED)
mp.set_start_method("spawn", force=True)

# ---------------------------------------------------------------------------
# Metrics & helpers
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

# Pre‑compute DRD weight matrix once
_DRD_WEIGHTS = np.array(
    [[1.0 / (abs(i - 2) + abs(j - 2) + 1) for j in range(5)] for i in range(5)]
)

def compute_psnr_batch(pred_batch: np.ndarray, tgt_batch: np.ndarray, eps: float = 1e-10) -> float:
    """Compute average PSNR for a batch."""
    psnr_values: List[float] = []
    for pred, tgt in zip(pred_batch, tgt_batch):
        mse = max(np.mean((pred - tgt) ** 2), eps)
        psnr_values.append(10 * math.log10(1.0 / mse))
    return float(np.mean(psnr_values))


def compute_drd(pred, gt):
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    diff = np.abs(pred - gt)
    # váhová matice pro DRD
    w = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            w[i, j] = 1.0 / (abs(i - 2) + abs(j - 2) + 1)
    conv = convolve2d(diff, w, mode='same', boundary='symm')
    total_drd = np.sum(conv * diff)
    num_error_pixels = np.sum(diff)
    return 0 if num_error_pixels == 0 else total_drd / num_error_pixels


def compute_drd_batch(pred_batch: np.ndarray, tgt_batch: np.ndarray) -> float:
    return float(np.mean([compute_drd(p, t) for p, t in zip(pred_batch, tgt_batch)]))


def compute_p_fm(precision: float, recall: float, beta_squared: float = 0.3) -> float:
    return (
        0.0
        if (precision + recall) == 0
        else (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)
    )

# ---------------------------------------------------------------------------
# Forward passes & model utilities
# ---------------------------------------------------------------------------

def _forward_nca(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    pixel_weight: float,
    mode: str,
):
    loss = 0.0
    for _ in range(CAMODEL_CONFIG["steps"]):
        x = model(x)
        #print(x.shape)
        #print(x[:, 3].shape)
        #print(y.squeeze(1).shape)
        """
        # starting from a PyTorch tensor of shape (1, 200, 200)
        img1_np = x[:, 3].squeeze()          # (200, 200)        ––> grayscale OK
        img1_np = img1_np.detach().cpu().numpy()

        img2_np = y.squeeze().detach().cpu().numpy()   # same

        fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=120)
        axes[0].imshow(img1_np, cmap='gray')
        axes[0].axis('off')
        axes[1].imshow(img2_np, cmap='gray')
        axes[1].axis('off')
        plt.tight_layout(); plt.show()"""

        loss += composite_loss(x[:, 3], y.squeeze(1), pixel_accuracy_weight=pixel_weight)
    #if mode == "train":
    #    plot_batch_x_channels(x.detach().cpu(), y.detach().cpu(), "post_model")
    correct, _ = calculate_pixel_accuracy(x[:, 3], y)
    return loss, correct, torch.sigmoid(x[:, 3])


def _forward_generic(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_function: torch.nn.Module,
    activation_needed: bool = True,
):
    out = model(x) if not isinstance(out := model(x), dict) else model(x)["out"]
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
):
    """Single forward & loss computation depending on *model_name*."""
    if model_name == "nca":
        pixel_weight = 1.0 if mode == "train" else 0.1
        return _forward_nca(model, x, y, pixel_weight, mode)
    if model_name == "deeplabv3":
        out_dict = model(x)
        loss = loss_function(out_dict["out"], y)
        correct, _ = calculate_pixel_accuracy(torch.sigmoid(out_dict["out"]), y)
        return loss, correct, torch.sigmoid(out_dict["out"])
    # All other models expose the raw logits directly
    return _forward_generic(model, x, y, loss_function)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def convert_to_numpy(output: torch.Tensor, target: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    if output.ndim == 4 and output.shape[1] == 1:
        output_np = _tensor_to_numpy(output.squeeze(1))
    else:
        output_np = _tensor_to_numpy(output)
    return output_np, _tensor_to_numpy(target)

# ---------------------------------------------------------------------------
# Training & evaluation loops
# ---------------------------------------------------------------------------

def _run_epoch(
    *,
    model: torch.nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer | None,
    loss_function: torch.nn.Module | None,
    device: torch.device,
    model_name: str,
    mode: str,
):
    is_train = mode == "train"
    if is_train:
        model.train()
    else:
        model.eval()

    totals = {tag: 0.0 for tag in METRIC_TAGS}
    total_samples = 0
    all_preds, all_targets = [], []

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
            binary_preds = (output_np > 0.5).astype(np.uint8)
            binary_targets = (y_np > 0.5).astype(np.uint8)
            totals["DRD"] += compute_drd_batch(binary_preds, binary_targets) * batch_size

            all_preds.extend(binary_preds.flatten())
            all_targets.extend(binary_targets.flatten())
            total_samples += batch_size

            if is_train:
                loss.backward()
                optimizer.step()

    # Compute classification‑style metrics once per epoch
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    p_fm = compute_p_fm(precision, recall)

    totals.update({
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FM": f1,  # duplicate to keep original contract
        "p-FM": p_fm,
        "PSNR": totals["PSNR"] / total_samples,
        "DRD": totals["DRD"] / total_samples,
    })

    # Normalize accumulated loss & accuracy
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
):
    return _run_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        model_name=model_name,
        mode="train",
    )


def evaluate_model(
    model: torch.nn.Module,
    dataloader: Iterable,
    loss_function: torch.nn.Module | None,
    device: torch.device,
    model_name: str,
):
    # Optimizer is irrelevant during evaluation
    return _run_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=None,
        loss_function=loss_function,
        device=device,
        model_name=model_name,
        mode="eval",
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
    try:
        return _MODEL_FACTORY[model_name]()
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc

# ---------------------------------------------------------------------------
# Top‑level experiment routine
# ---------------------------------------------------------------------------

def _resolve_shape_and_classes(model_name: str) -> Tuple[int, int]:
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


def main(*, model_name: str = "nca", run: int = 1, steps: int = 1):
    start_time = time.time()
    # Update global CAMODEL_CONFIG steps if required
    if model_name == "nca":
        CAMODEL_CONFIG["steps"] = steps

    n_channels, _ = _resolve_shape_and_classes(model_name)
    cfg = TRAINING_CONFIG

    model = initialize_model(model_name)
    if model_name != "deeplabv3":
        from torchsummary import summary

        summary(model, (n_channels, 200, 200))  # input size is fixed 50×50

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

    filter_masks = cfg["dataset_name"] == "dibco"

    def _get_loader(images_dir: str, masks_dir: str):
        return get_dataloader(
            images_dir,
            masks_dir,
            n_channels,
            cfg["batch_size"],
            200,  # crop size fixed
            filter_masks,
        )

    train_loader = _get_loader(cfg["train_images_dir"], cfg["train_masks_dir"])
    test_loader = _get_loader(cfg["test_images_dir"], cfg["test_masks_dir"])

    writer = SummaryWriter(Path("runs") / f"{model_name}_steps_{steps}_run_{run + 1}")
    epoch_curve: Dict[str, Dict[str, List[float]]] = {
        "Train": {tag: [] for tag in METRIC_TAGS},
        "Test": {tag: [] for tag in METRIC_TAGS},
    }

    best_loss = float("inf")
    best_train = best_test = best_epoch = None

    for epoch in range(cfg["epochs"]):
        print(f"Run {run + 1} | Epoch {epoch + 1}/{cfg['epochs']} | steps: {steps}")

        # Initial evaluation before any training
        if epoch == 0:
            init_metrics = evaluate_model(model, test_loader, loss_function, DEVICE, model_name)
            for i, tag in enumerate(METRIC_TAGS):
                writer.add_scalar(f"Test/{tag}", init_metrics[i], epoch)
            print(
                f"Initial Test: Loss={init_metrics[0]:.4f}, Acc={init_metrics[1]:.2f}%"
            )

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_function, DEVICE, model_name
        )
        test_metrics = evaluate_model(model, test_loader, loss_function, DEVICE, model_name)

        # Remember best model according to test loss
        if test_metrics[0] < best_loss:
            best_loss, best_train, best_test, best_epoch = test_metrics[0], train_metrics, test_metrics, epoch

        # Log epoch metrics
        for i, tag in enumerate(METRIC_TAGS):
            writer.add_scalar(f"Train/{tag}", train_metrics[i], epoch)
            writer.add_scalar(f"Test/{tag}", test_metrics[i], epoch)
            epoch_curve["Train"][tag].append(train_metrics[i])
            epoch_curve["Test"][tag].append(test_metrics[i])

        model_config = {}
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

    training_time = f"{(time.time() - start_time):.2f}s"


    torch.save(model.state_dict(), "models/mynet_weights_small_run" + str(run) + ".pt")
    model_path = "models/mynet_weights_small_run" + str(run) + ".pt"
    # Compute MACs, params
    macs = estimate_model_macs(model, input_shape=(CAMODEL_CONFIG["n_channels"], 200, 200))


    # Add other metadata
    model_stats = {
        "model_name": model_name,
        "macs": macs,
        "adds": 0,
        "total_ops": macs,
        "model_size_MB": os.path.getsize(model_path) / (1024 ** 2),
        "training_time": training_time,
        "hardware_target": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }

    log_hw_metrics(run_id=run, model_name=model_name, model_stats=model_stats)



    return (best_test, best_train), epoch_curve

# ---------------------------------------------------------------------------
# Stand‑alone execution for experiments with varying NCA steps
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    STEPS_VARIANTS: List[int] = [1]
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

    # Average curves across runs
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

    # Display best results per *steps* value
    for steps, data in final_results.items():
        run_results = data["final"]
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
