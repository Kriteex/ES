"""
Size Invarience Experiment
==========================

This script demonstrates the **size invariance** property of a Neural Cellular
Automata (NCA) used for binary segmentation.  The core idea behind a
cellular automaton is that every update depends only on a local neighbourhood
of pixels.  Consequently, once an NCA has learned to transform a small
patch correctly, the same learned rules should generalise to arbitrarily
large images.  In other words, whether your input is **20×20** or
**2000×2000**, the NCA will behave identically on each local neighbourhood.

The implementation below takes inspiration from the existing training code in
the repository (`train.py` and `utils.py`).  It trains a tiny CA to copy a
binary mask from its input channel to its output channel.  After training
on small 64×64 patches, the script evaluates the model on two very
different input sizes (20×20 and 2000×2000).  At each simulation step it
computes the Peak Signal‑to‑Noise Ratio (PSNR) and pixel accuracy of the
predicted mask relative to the ground truth.  Finally, it plots PSNR and
accuracy curves for both sizes.

**Note:** This script relies on PyTorch and Matplotlib.  It will not run
inside environments without these dependencies.  To execute the experiment
simply run

```bash
python size_invarience.py
```

in the project root.  The resulting graphs will be saved in the local
directory.

"""

import os
import math
import random
from typing import List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError(
        "PyTorch is required to run the size_invarience experiment. Please install "
        "torch in your Python environment."
    )

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError(
        "Matplotlib is required to plot the PSNR and accuracy curves. Please install "
        "matplotlib in your Python environment."
    )

# Import the Cellular Automata model and configuration from the repository.
from config import CAMODEL_CONFIG, DEVICE
from model import CAModel


def compute_psnr(pred: np.ndarray, tgt: np.ndarray, threshold: float = 0.5, eps: float = 1e-10) -> float:
    """Compute the PSNR between two binary masks.

    The inputs are expected to be 2‑D arrays with values in `[0, 1]`.  They are
    thresholded at `threshold` to obtain binary masks.  PSNR is then
    calculated in decibels using the mean squared error (MSE) between the
    binary predictions and the ground truth.  A small `eps` is added to
    avoid division by zero.

    Args:
        pred: Predicted probability map of shape `(H, W)`.
        tgt: Ground truth binary mask of shape `(H, W)`.
        threshold: Binarisation threshold for the prediction.
        eps: Numerical stability term.

    Returns:
        The PSNR value in decibels.
    """
    pred_bin = (pred > threshold).astype(np.float32)
    tgt_bin = (tgt > threshold).astype(np.float32)
    mse = max(np.mean((pred_bin - tgt_bin) ** 2), eps)
    return 10.0 * math.log10(1.0 / mse)


def compute_pixel_accuracy(pred: np.ndarray, tgt: np.ndarray, threshold: float = 0.5) -> float:
    """Compute the pixel accuracy between a prediction and a target mask.

    Args:
        pred: Predicted probability map of shape `(H, W)`.
        tgt: Ground truth binary mask of shape `(H, W)`.
        threshold: Binarisation threshold for the prediction.

    Returns:
        The percentage of correctly classified pixels (0–100).
    """
    pred_bin = (pred > threshold).astype(np.uint8)
    tgt_bin = (tgt > threshold).astype(np.uint8)
    correct = (pred_bin == tgt_bin).sum()
    total = pred_bin.size
    return float(correct) / float(total) * 100.0


def generate_random_mask(size: int, p: float = 0.5) -> torch.Tensor:
    """Generate a random binary mask of shape `(size, size)`.

    Each pixel is independently sampled from a Bernoulli distribution with
    probability `p` of being 1.

    Args:
        size: The height and width of the mask.
        p: Probability that a pixel is 1.

    Returns:
        A torch tensor of shape `(1, size, size)` with dtype `torch.float32`.
    """
    mask = torch.bernoulli(torch.full((size, size), p, dtype=torch.float32))
    return mask.unsqueeze(0)


def train_nca(model: CAModel, num_iterations: int = 500, patch_size: int = 64, lr: float = 1e-3) -> None:
    """Train the NCA to copy a binary mask on small patches.

    This training loop uses a very simple objective: after a single update
    step, the NCA should reproduce its input mask on the fourth channel.  A
    binary cross‑entropy loss (with logits) is used to drive the learning.

    Args:
        model: The cellular automata model to be trained.
        num_iterations: Number of training iterations to run.
        patch_size: Size of the square training patches.
        lr: Learning rate for the Adam optimiser.
    """
    # We override the number of internal steps to 1 during training so that
    # gradients propagate cleanly through a single update.  If you wish to
    # train over multiple internal steps, adjust this value accordingly.
    model.steps = 1
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for it in range(num_iterations):
        # Generate a random binary mask and use it as both input and target
        target_mask = generate_random_mask(patch_size).to(model.device)
        # Initial CA state: first three channels are zeros, fourth channel holds the mask
        x = torch.zeros(1, model.perceive_conv[0].in_channels, patch_size, patch_size, device=model.device)
        x[:, 3] = target_mask
        # Forward through the CA
        x = model(x)
        logits = x[:, 3]
        loss = loss_fn(logits, target_mask)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (it + 1) % 100 == 0:
            print(f"[Training] Iteration {it + 1}/{num_iterations}, Loss: {loss.item():.4f}")


def evaluate_size_invarience(
    model: CAModel,
    sizes: List[int],
    steps: int = 10,
    p: float = 0.5,
    seed: int = 42,
    results_dir: str = "size_invarience_results",
) -> None:
    """Evaluate the trained NCA on masks of different sizes and plot PSNR/ACC curves.

    For each size in `sizes`, a random binary mask is generated, placed in the
    fourth channel of an initial CA state, and then evolved for `steps`
    iterations.  After each iteration the PSNR and pixel accuracy are
    computed against the ground truth.  Two plots (PSNR vs steps and
    accuracy vs steps) are saved in `results_dir`.

    Args:
        model: The trained cellular automata model.
        sizes: A list of image sizes (e.g., [20, 2000]).
        steps: Number of CA update steps during evaluation.
        p: Probability for random mask generation.
        seed: Random seed for reproducibility.
        results_dir: Directory where plots will be saved.
    """
    os.makedirs(results_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_psnr: List[List[float]] = []
    all_acc: List[List[float]] = []

    for size in sizes:
        print(f"[Evaluation] Evaluating on size {size}×{size}...")
        target = generate_random_mask(size, p=p).to(model.device)
        # Build initial CA state with zeros in the first three channels and target in the fourth
        x = torch.zeros(1, model.perceive_conv[0].in_channels, size, size, device=model.device)
        x[:, 3] = target
        # Run the CA for the specified number of steps and record metrics
        psnr_curve: List[float] = []
        acc_curve: List[float] = []
        with torch.no_grad():
            for step in range(steps):
                x = model(x)
                logits = x[:, 3]
                preds = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                gt = target.squeeze(0).cpu().numpy()
                psnr_curve.append(compute_psnr(preds, gt))
                acc_curve.append(compute_pixel_accuracy(preds, gt))
        all_psnr.append(psnr_curve)
        all_acc.append(acc_curve)

        # Plot PSNR and ACC curves for this size
        steps_range = list(range(1, steps + 1))
        plt.figure()
        plt.plot(steps_range, psnr_curve, marker="o")
        plt.title(f"PSNR vs Steps for {size}×{size}")
        plt.xlabel("CA Update Step")
        plt.ylabel("PSNR (dB)")
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"psnr_curve_{size}.png"))
        plt.close()

        plt.figure()
        plt.plot(steps_range, acc_curve, marker="o")
        plt.title(f"Pixel Accuracy vs Steps for {size}×{size}")
        plt.xlabel("CA Update Step")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"accuracy_curve_{size}.png"))
        plt.close()

    # Optionally, overlay curves from different sizes on a single plot
    plt.figure()
    for size, psnr_curve in zip(sizes, all_psnr):
        plt.plot(range(1, steps + 1), psnr_curve, marker="o", label=f"{size}×{size}")
    plt.title("PSNR vs Steps (All Sizes)")
    plt.xlabel("CA Update Step")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "psnr_curves_comparison.png"))
    plt.close()

    plt.figure()
    for size, acc_curve in zip(sizes, all_acc):
        plt.plot(range(1, steps + 1), acc_curve, marker="o", label=f"{size}×{size}")
    plt.title("Pixel Accuracy vs Steps (All Sizes)")
    plt.xlabel("CA Update Step")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "accuracy_curves_comparison.png"))
    plt.close()


def main() -> None:
    """Main entry point for the size invariance experiment."""
    # Instantiate the CA model with the base configuration.  The CAModel
    # constructor takes keyword arguments, so we unpack the config here.  The
    # config includes parameters such as number of channels, hidden channels,
    # neighbourhood size and update depth.  Feel free to experiment with
    # different configurations (e.g., CAMODEL_CONFIG_05_BASELINE) to observe
    # their behaviour.
    model = CAModel(**CAMODEL_CONFIG)
    model.to(DEVICE)

    # Train the model on small patches.  If you have a pre‑trained model you
    # can skip this step and load your weights instead.
    print("[Info] Starting training...")
    train_nca(model, num_iterations=500, patch_size=64, lr=1e-3)
    print("[Info] Training complete. Starting evaluation...")

    # Evaluate on two vastly different sizes.  Here we choose 20×20 and
    # 2000×2000 to emphasise the contrast.  You can add more sizes to the
    # list if desired.
    evaluate_size_invarience(
        model,
        sizes=[20, 2000],
        steps=10,
        p=0.5,
        seed=42,
        results_dir="size_invarience_results",
    )
    print("[Info] Evaluation complete. Plots saved in 'size_invarience_results'.")


if __name__ == "__main__":
    main()