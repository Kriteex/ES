# nc.py
"""
Interactive visualization of a Neural Cellular Automata (NCA) model using PyQt5.

This module provides a GUI for loading a trained CA model and an image, dividing
the image into tiles, stepping the model to evolve the state, and displaying
the reconstructed result.  The GUI displays both the RGB image and a binary
mask derived from the alpha channel.
"""

import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from config import CAMODEL_CONFIG, DEVICE
from model import CAModel

# Global references to loaded models (for convenience).
loaded_model: Optional[CAModel] = None

def reconstruct_image(
    tiles: List[torch.Tensor],
    original_size: Tuple[int, int],
    tile_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Stitch a list of tiles back into a single image tensor.

    Args:
        tiles: List of tile tensors of shape (1, C, tile_height, tile_width).
        original_size: Original full image height and width (H, W).
        tile_size: Height and width of each tile (h, w).

    Returns:
        A tensor of shape (C, H, W) representing the stitched image.
    """
    tiles_x = original_size[1] // tile_size[1]
    tiles_y = original_size[0] // tile_size[0]
    channels = tiles[0].shape[1]
    reconstructed = torch.zeros((channels, original_size[0], original_size[1]))
    for i in range(tiles_y):
        for j in range(tiles_x):
            idx = i * tiles_x + j
            reconstructed[:, i * tile_size[0] : (i + 1) * tile_size[0], j * tile_size[1] : (j + 1) * tile_size[1]] = (
                tiles[idx].squeeze(0)
            )
    return reconstructed

def load_model(file_path: str) -> CAModel:
    """
    Instantiate and load a CA model from a given checkpoint file.

    Args:
        file_path: Path to a .pth file containing state_dict.

    Returns:
        A CAModel instance loaded with the checkpoint weights.
    """
    model = CAModel(**CAMODEL_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model

def load_and_tile_image(
    file_path: str,
    tile_size: Tuple[int, int] = (200, 200),
    additional_channels: int = CAMODEL_CONFIG["n_channels"] - 3,
) -> Tuple[List[torch.Tensor], int, int]:
    """
    Read an image file, normalize it, and split it into tiles.

    Args:
        file_path: Path to the input image.
        tile_size: Desired size of each tile (height, width).
        additional_channels: Number of extra channels to append to each tile.

    Returns:
        A tuple of (list of tile tensors, original height, original width).
    """
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {file_path}")

    image_tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).unsqueeze(0)
    # Drop alpha channel if present
    if image_tensor.shape[1] == 4:
        image_tensor = image_tensor[:, :3, :, :]
    height, width = image_tensor.shape[2], image_tensor.shape[3]
    tiles_x = width // tile_size[1]
    tiles_y = height // tile_size[0]

    tiles: List[torch.Tensor] = []
    for i in range(tiles_y):
        for j in range(tiles_x):
            tile = image_tensor[
                :, :, i * tile_size[0] : (i + 1) * tile_size[0], j * tile_size[1] : (j + 1) * tile_size[1]
            ]
            if additional_channels > 0:
                mean_channel = tile.mean(dim=1, keepdim=True).expand(-1, additional_channels, -1, -1)
                tile = torch.cat((tile, mean_channel), dim=1)
            tiles.append(tile)
    return tiles, height, width

def step(
    model: CAModel,
    state: torch.Tensor,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    Perform a single iteration of the CA model.

    Args:
        model: Loaded CA model.
        state: Current tile state.
        device: Device on which to run the model.

    Returns:
        The next state as a CPU tensor.
    """
    with torch.no_grad():
        next_state = model(state.float().to(device))
    return next_state.cpu().detach()

class NCAVisualizer(QWidget):
    """
    Minimal PyQt5 interface for loading a CA model, loading an image,
    running simulation steps, and visualizing results.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tiles: Optional[List[torch.Tensor]] = None
        self.reconstructed: Optional[torch.Tensor] = None
        self.height: Optional[int] = None
        self.width: Optional[int] = None
        self.steps: int = 0
        self.threshold: int = 127
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialise UI components and layout."""
        self.setWindowTitle("NCA Visualizer")
        self.setGeometry(100, 100, 1200, 600)
        layout = QVBoxLayout()
        button_row = QHBoxLayout()

        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self._on_load_model)
        button_row.addWidget(load_model_btn)

        load_img_btn = QPushButton("Load Image")
        load_img_btn.clicked.connect(self._on_load_image)
        button_row.addWidget(load_img_btn)

        step_btn = QPushButton("Step")
        step_btn.clicked.connect(self._on_step)
        button_row.addWidget(step_btn)

        self.step_label = QLabel("Steps: 0")
        button_row.addWidget(self.step_label)
        layout.addLayout(button_row)

        # Visualisation canvases
        grid = QGridLayout()
        self.fig_rgb = FigureCanvas(Figure(figsize=(4, 4)))
        self.ax_rgb = self.fig_rgb.figure.subplots()
        self.ax_rgb.axis("off")
        self.fig_mask = FigureCanvas(Figure(figsize=(4, 4)))
        self.ax_mask = self.fig_mask.figure.subplots()
        self.ax_mask.axis("off")
        grid.addWidget(self.fig_rgb, 0, 0)
        grid.addWidget(self.fig_mask, 0, 1)
        layout.addLayout(grid)
        self.setLayout(layout)

    def _on_load_model(self) -> None:
        """Prompt user to select a model checkpoint and load it."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pth);;All Files (*)")
        if path:
            global loaded_model
            loaded_model = load_model(path)
            self.steps = 0
            self.step_label.setText("Steps: 0")
            if self.tiles is not None:
                self._update_visualisation(initial=True)

    def _on_load_image(self) -> None:
        """Prompt user to select an image and prepare tiles."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.bmp *.tiff *.tif);;All Files (*)"
        )
        if path:
            self.tiles, self.height, self.width = load_and_tile_image(path)
            self.reconstructed = None
            self.steps = 0
            self.step_label.setText("Steps: 0")
            self._update_visualisation(initial=True)

    def _on_step(self) -> None:
        """Advance the CA by one step on all tiles and update display."""
        if self.tiles is None or loaded_model is None:
            return
        for idx, tile in enumerate(self.tiles):
            self.tiles[idx] = step(loaded_model, tile)
        tile_h, tile_w = self.tiles[0].shape[2], self.tiles[0].shape[3]
        self.reconstructed = reconstruct_image(self.tiles, (self.height, self.width), (tile_h, tile_w))
        self.steps += 1
        self.step_label.setText(f"Steps: {self.steps}")
        self._update_visualisation()

    def _update_visualisation(self, initial: bool = False) -> None:
        """Refresh the canvas with the latest reconstructed state."""
        if self.reconstructed is not None:
            rgb = self.reconstructed[:3].permute(1, 2, 0).numpy()
            self.ax_rgb.clear()
            self.ax_rgb.imshow(rgb)
            self.ax_rgb.axis("off")

            alpha_mask = (self.reconstructed[3].numpy() * 255) > self.threshold
            self.ax_mask.clear()
            self.ax_mask.imshow(alpha_mask, cmap="viridis")
            self.ax_mask.axis("off")

            self.fig_rgb.draw()
            self.fig_mask.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    vis = NCAVisualizer()
    vis.show()
    sys.exit(app.exec_())
