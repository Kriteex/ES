"""
Experimentation script for neural cellular automata (NCA)
-----------------------------------------------------------------

This module provides a flexible entry point for training a
`CAModel` with one number of update steps (``train_steps``) and
evaluating the same trained model with a different number of steps
(``run_steps``).  During training the model is unrolled
``train_steps`` times per forward pass so that gradients flow
through ``train_steps`` state updates.  At test time we can set
``run_steps`` to a larger or smaller value to observe how the
learned update rules generalise when iterated more or fewer times
than during training.

The script is intentionally designed to integrate seamlessly with
the existing project structure.  It reuses the ``main`` function
from ``train.py`` for training, initialises and loads models via
``initialize_model`` and ``CAModel``, constructs dataloaders via
``_prepare_loaders`` and runs evaluation via ``evaluate_model``.
Hyperparameters such as the dataset location, optimiser, batch size
and learning rate are read from ``config.py``.  The only values
modified on the fly are ``CAMODEL_CONFIG['steps']`` and the
checkpoint naming scheme.

Usage
===== 

Run from the project root with your desired number of steps.  For
example, to train an NCA for one update step and evaluate it when
unrolled for four steps:

.. code-block:: bash

    python trained_for_n_run_for_m.py --train_steps 1 --run_steps 4 --epochs 30

Multiple runs can be launched by increasing ``--n_runs``; runs are
identified in the checkpoint filenames.  If a checkpoint for a
given ``train_steps`` and ``run`` already exists, training is
skipped and the file is reused.  This makes it cheap to sweep
across many evaluation step counts without retraining.

Notes
-----

* The script assumes that the working directory contains the
  existing codebase with ``config.py`` and ``train.py``.  It uses
  functions and global variables from those modules to remain
  consistent with the rest of the project.
* Training and evaluation always operate on the test loader defined
  in ``config.py``.  No custom dataset splitting is performed here.
* Because NCA training can be sensitive to the number of unrolled
  steps, be cautious when evaluating far beyond ``train_steps``.  In
  many cases the model will converge to a fixed point after a
  handful of iterations; measuring this behaviour can still be
  illuminating.
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch

# Local imports from this project.  These modules must be present in the
# same repository.  They expose configuration dictionaries and helper
# functions used throughout the existing training pipeline.
from config import CAMODEL_CONFIG, TRAINING_CONFIG, DEVICE  # type: ignore
from train import (
    main as _train_main,
    initialize_model,
    evaluate_model,
    _prepare_loaders,
    _resolve_shape_and_classes,
)  # type: ignore
from model import CAModel  # type: ignore


def train_nca(train_steps: int, run: int, epochs: Optional[int] = None) -> Path:
    """Train an NCA for a fixed number of unrolled steps.

    This function wraps the existing ``main`` function from ``train.py``.
    It sets ``CAMODEL_CONFIG['steps']`` to ``train_steps``, kicks off
    training and returns the path to the saved checkpoint.  If a
    checkpoint matching the requested ``train_steps`` and ``run`` exists
    already, training is skipped and the existing file is returned.

    Args:
        train_steps: Number of unrolled update steps during training.
        run: Index of the run (used in checkpoint naming).
        epochs: Optionally override the default number of epochs defined in
            ``TRAINING_CONFIG``.  When provided, ``TRAINING_CONFIG['epochs']``
            will be temporarily patched for the duration of training.

    Returns:
        A ``Path`` to the model checkpoint that contains the trained
        weights.
    """
    # Determine checkpoint path based on the requested configuration.  This
    # naming scheme avoids collisions between different numbers of steps
    # and run indices.
    ckpt_dir = Path("models")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_name = f"nca_train{train_steps}_run{run}.pt"
    ckpt_path = ckpt_dir / ckpt_name

    # If the checkpoint already exists, skip training to save time.
    if ckpt_path.exists():
        print(f"[train_nca] Checkpoint {ckpt_path} already exists; skipping training.")
        return ckpt_path

    # Temporarily override the global number of epochs if requested.  Use
    # context management so that TRAINING_CONFIG is restored after training.
    original_epochs = TRAINING_CONFIG.get("epochs")
    if epochs is not None:
        TRAINING_CONFIG["epochs"] = epochs

    # Set the number of steps on the global CA config.  The ``main``
    # function will propagate this value into the model constructor.
    CAMODEL_CONFIG["steps"] = train_steps

    # Kick off training.  The returned values include metrics but are
    # intentionally ignored here because we only need the saved model.
    print(f"[train_nca] Training NCA with {train_steps} unrolled steps (run {run}).")
    _train_main(model_name="nca", run=run, steps=train_steps)

    # Restore the original epoch count if it was overridden.
    if epochs is not None and original_epochs is not None:
        TRAINING_CONFIG["epochs"] = original_epochs

    # The ``main`` function from ``train.py`` writes its checkpoint under
    # ``models/mynet_weights_small_run{run}.pt`` by default.  Rename this
    # file to our custom naming scheme to avoid clobbering it on subsequent
    # runs and to embed the training step count in the filename.  If the
    # default file does not exist, raise an error to alert the user.
    default_ckpt = ckpt_dir / f"mynet_weights_small_run{run}.pt"
    if default_ckpt.exists():
        default_ckpt.rename(ckpt_path)
    else:
        raise FileNotFoundError(
            f"Expected checkpoint {default_ckpt} not found after training; did the training succeed?"
        )
    print(f"[train_nca] Training complete.  Checkpoint saved to {ckpt_path}.")
    return ckpt_path


def evaluate_nca(model_path: Path, run_steps: int) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Evaluate a trained NCA with a different number of unrolled steps.

    The function sets ``CAMODEL_CONFIG['steps']`` to ``run_steps``,
    instantiates a fresh ``CAModel`` with this configuration, loads
    ``model_path`` into it and evaluates the model on the test set.  The
    evaluation metrics mirror those returned by ``train.evaluate_model``:
    ``Loss``, ``Accuracy``, ``Precision``, ``Recall``, ``F1``, ``FM``,
    ``p-FM``, ``PSNR`` and ``DRD``.

    Args:
        model_path: Path to the trained model checkpoint.
        run_steps: Number of update steps during evaluation.

    Returns:
        A tuple of nine floats corresponding to the evaluation metrics.
    """
    # Update the global CA configuration to reflect the desired number of
    # update steps at inference time.  This ensures that the model
    # constructor builds the appropriate number of perception and update
    # layers and that the forward pass unrolls ``run_steps`` times.
    CAMODEL_CONFIG["steps"] = run_steps

    # Reconstruct the model and load the trained weights.  We rely on
    # ``CAModel`` directly here instead of ``initialize_model`` because we
    # want to ensure the architecture matches the training run (e.g. number
    # of channels, depth, neighbour size, use of residual connections).
    model = CAModel(**CAMODEL_CONFIG).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # Build dataloaders.  We reuse the dataset configuration from
    # ``TRAINING_CONFIG``.  The call to ``_resolve_shape_and_classes``
    # determines how many channels the dataloader should produce.  Note
    # that for NCA models the number of classes is always one (binary
    # segmentation), so we ignore the second return value.
    n_channels, _ = _resolve_shape_and_classes("nca")
    filter_masks = TRAINING_CONFIG.get("dataset_name", "") == "dibco"
    _, test_loader = _prepare_loaders(
        n_channels=n_channels,
        batch_size=TRAINING_CONFIG["batch_size"],
        crop_size=200,
        filter_masks=filter_masks,
    )

    # Evaluate.  Loss function is ``None`` for NCA models because the
    # training loop computes the loss internally via ``composite_loss`` or
    # BCE.  The evaluation function gracefully handles ``loss_function=None``.
    metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        loss_function=None,
        device=DEVICE,
        model_name="nca",
        epoch_idx=0,
        run_idx=0,
    )
    return metrics


def run_experiments(train_steps_list: Iterable[int], run_steps_list: Iterable[int], n_runs: int, epochs: Optional[int]) -> None:
    """Execute a grid of (train_steps, run_steps) experiments.

    For each number of training steps in ``train_steps_list`` and each number
    of evaluation steps in ``run_steps_list`` this function trains (if
    necessary) and evaluates ``n_runs`` independent replicates.  Results
    are printed to stdout in a tabular form and stored in nested
    dictionaries for convenience.

    Args:
        train_steps_list: Iterable of integers specifying numbers of
            unrolled steps during training.
        run_steps_list: Iterable of integers specifying numbers of
            unrolled steps during evaluation.
        n_runs: Number of independent runs per (train_steps, run_steps)
            configuration.
        epochs: Optional override for the number of training epochs.  If
            ``None``, the default from ``TRAINING_CONFIG`` is used.
    """
    results = {}
    for train_steps in train_steps_list:
        results[train_steps] = {}
        for run_steps in run_steps_list:
            results[train_steps][run_steps] = []
            for run in range(n_runs):
                # Train (or reuse) the checkpoint for this number of steps
                ckpt_path = train_nca(train_steps=train_steps, run=run, epochs=epochs)
                # Evaluate on the desired number of steps
                metrics = evaluate_nca(model_path=ckpt_path, run_steps=run_steps)
                results[train_steps][run_steps].append(metrics)
                # Unpack metrics into readable variables
                loss, acc, prec, recall, f1, fm, pfm, psnr, drd = metrics
                print(
                    f"TrainSteps={train_steps}, RunSteps={run_steps}, Run={run}: "
                    f"Loss={loss:.4f}, Acc={acc:.2f}%, Prec={prec:.2f}, "
                    f"Recall={recall:.2f}, F1={f1:.2f}, PSNR={psnr:.2f}, DRD={drd:.2f}"
                )
    # After finishing, summarise results by reporting averages over runs
    print("\nSummary of results (averaged over runs):")
    for train_steps in train_steps_list:
        for run_steps in run_steps_list:
            all_metrics = results[train_steps][run_steps]
            # Compute mean metrics across runs
            means = [float(sum(x[i] for x in all_metrics) / len(all_metrics)) for i in range(9)]
            loss, acc, prec, recall, f1, fm, pfm, psnr, drd = means
            print(
                f"TrainSteps={train_steps}, RunSteps={run_steps}: "
                f"Loss={loss:.4f}, Acc={acc:.2f}%, Prec={prec:.2f}, "
                f"Recall={recall:.2f}, F1={f1:.2f}, PSNR={psnr:.2f}, DRD={drd:.2f}"
            )


def parse_int_list(arg: str) -> Tuple[int, ...]:
    """Parse a comma-separated string of integers into a tuple.

    Allows users to specify multiple training or run steps on the command
    line.  For example, ``--train_steps 1,2,4`` becomes ``(1, 2, 4)``.

    Args:
        arg: Input string from the command line.

    Returns:
        Tuple of parsed integers.
    """
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("List of ints cannot be empty.")
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {arg}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an NCA for N steps and evaluate for M steps.")
    parser.add_argument(
        "--train_steps",
        type=parse_int_list,
        required=True,
        help="Comma-separated list of unrolled steps during training.",
    )
    parser.add_argument(
        "--run_steps",
        type=parse_int_list,
        required=True,
        help="Comma-separated list of unrolled steps during evaluation.",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of independent runs per configuration (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Override the default number of epochs defined in TRAINING_CONFIG.",
    )

    args = parser.parse_args()
    run_experiments(
        train_steps_list=args.train_steps,
        run_steps_list=args.run_steps,
        n_runs=args.n_runs,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()