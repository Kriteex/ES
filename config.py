"""
Configuration settings for segmentation and neural cellular automata experiments.

This module centralises all hyperparameters and directory paths in one place
so that experiments can be easily adjusted without touching the rest of the
codebase.  Settings are organised into sections: device, model configs,
cellular automaton (CA) configs, training defaults, and data loading settings.
"""

import torch
from typing import Tuple

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model configuration templates
# ---------------------------------------------------------------------------

# U‑Net variants
UNET_CONFIG: dict = {"n_channels": 3, "n_classes": 1}
SEGNET_CONFIG: dict = {"n_channels": 3, "n_classes": 1}

# DeepLabV3 requires additional fields to override the initial conv layer
DEEPLABV3_CONFIG: dict = {"n_channels": 3, "num_classes": 1, "pretrained": False}
DEEPLABV3_CONFIG_CONV1: dict = {
    "in_channels": 3,
    "out_channels": 64,
    "kernel_size": 7,
    "stride": 2,
    "padding": 3,
    "bias": False,
}

# PSPNet uses a backbone; specify encoder
PSPNET_CONFIG: dict = {"in_channels": 3, "classes": 1, "encoder_name": "resnet50"}

# ---------------------------------------------------------------------------
# Cellular Automata model configs
# ---------------------------------------------------------------------------

# Each CA configuration is a dictionary of hyperparameters.  Use the one that
# best suits your experiment; see README for guidance.
CAMODEL_CONFIG = { #_01_MINIMAL = {
    "n_channels": 4,
    "hidden_channels": 1,
    "fire_rate": 0.5,
    "device": DEVICE,
    "neighbour": 3,
    "deep_perceive": 1,
    "deep_update": 1,
    "use_residual": False,
    "steps": 1,
}

CAMODEL_CONFIG_02_SHALLOW = {
    "n_channels": 6,
    "hidden_channels": 8,
    "fire_rate": 0.5,
    "device": DEVICE,
    "neighbour": 3,
    "deep_perceive": 1,
    "deep_update": 1,
    "use_residual": False,
    "steps": 2,
}

CAMODEL_CONFIG_03_WIDE = {
    "n_channels": 8,
    "hidden_channels": 16,
    "fire_rate": 0.5,
    "device": DEVICE,
    "neighbour": 3,
    "deep_perceive": 1,
    "deep_update": 1,
    "use_residual": False,
    "steps": 2,
}

CAMODEL_CONFIG_04_RESIDUAL_LIGHT = {
    "n_channels": 4,
    "hidden_channels": 32,
    "fire_rate": 0.5,
    "device": DEVICE,
    "neighbour": 5,
    "deep_perceive": 1,
    "deep_update": 2,
    "use_residual": True,
    "steps": 1,
}

CAMODEL_CONFIG_05_BASELINE = {
    "n_channels": 8,
    "hidden_channels": 32,
    "fire_rate": 0.5,
    "device": DEVICE,
    "neighbour": 7,
    "deep_perceive": 2,
    "deep_update": 2,
    "use_residual": True,
    "steps": 2,
}

CAMODEL_CONFIG_06_INTERMEDIATE = {
    "n_channels": 12,
    "hidden_channels": 48,
    "fire_rate": 0.5,
    "device": DEVICE,
    "neighbour": 5,
    "deep_perceive": 2,
    "deep_update": 2,
    "use_residual": True,
    "steps": 2,
}

CAMODEL_CONFIG_07_NEIGHBOR_EXPANDED = {
    "n_channels": 12,
    "hidden_channels": 48,
    "fire_rate": 0.5,
    "device": DEVICE,
    "neighbour": 7,
    "deep_perceive": 2,
    "deep_update": 2,
    "use_residual": True,
    "steps": 3,
}

CAMODEL_CONFIG_08_DEEP = {
    "n_channels": 16,
    "hidden_channels": 64,
    "fire_rate": 0.5,
    "device": DEVICE,
    "neighbour": 7,
    "deep_perceive": 3,
    "deep_update": 2,
    "use_residual": True,
    "steps": 3,
}

CAMODEL_CONFIG_09_DEEPER = {
    "n_channels": 20,
    "hidden_channels": 100,
    "fire_rate": 0.5,
    "device": DEVICE,
    "neighbour": 7,
    "deep_perceive": 3,
    "deep_update": 3,
    "use_residual": True,
    "steps": 3,
}

CAMODEL_CONFIG_10_ULTRA = {
    "n_channels": 24,
    "hidden_channels": 120,
    "fire_rate": 0.5,
    "device": DEVICE,
    "neighbour": 17,
    "deep_perceive": 5,
    "deep_update": 2,
    "use_residual": True,
    "steps": 4,
}

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

TRAINING_CONFIG = {
    "dataset_name": "dibco",
    "batch_size": 5,
    "learning_rate": 1e-2,
    "epochs": 30,
    "loss_function": torch.nn.BCEWithLogitsLoss(),
    "optimizer": torch.optim.Adam,
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_step_size": 10,
    "scheduler_gamma": 0.5,
    "train_images_dir": "data/data-DIBCO/datasets/DIPCO2016_dataset_cropsz",
    "train_masks_dir": "data/data-DIBCO/datasets/DIPCO2016_dataset_cropsz",
    "test_images_dir":  "data/data-DIBCO/datasets/DIPCO2016_dataset_cropsz_test",
    "test_masks_dir":   "data/data-DIBCO/datasets/DIPCO2016_dataset_cropsz_test",
    "default_model_path": "nca_0/nca_0_epoch_0.pt",
    "window_title": "NCA Visualizer",
    "default_threshold": 127,
    "crop_size": 800,
}

# ---------------------------------------------------------------------------
# Seeds and DataLoader settings
# ---------------------------------------------------------------------------

SEED: int = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATALOADER_CONFIG = {
    "shuffle": True,
    "num_workers": 2 if torch.cuda.is_available() else 0,
    "pin_memory": torch.cuda.is_available(),
}

# ---------------------------------------------------------------------------
# Image constants
# ---------------------------------------------------------------------------

TILE_SIZE: Tuple[int, int] = (200, 200)
ADDITIONAL_CHANNELS: int = 1
ALPHA: bool = True
