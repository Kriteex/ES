"""
Cellular Automata (CA) visualizer using PyQt5.
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
from utils import get_device

# Global models (can be later refactored as instance attributes)
model1: Optional[CAModel] = None
model2: Optional[CAModel] = None


def reconstruct_image(tiles: List[torch.Tensor], original_size: Tuple[int, int], tile_size: Tuple[int, int]) -> torch.Tensor:
    """
    Reconstruct the full image from tiles.

    Args:
        tiles (List[torch.Tensor]): List of tile tensors.
        original_size (Tuple[int, int]): Original image dimensions (height, width).
        tile_size (Tuple[int, int]): Tile dimensions (height, width).

    Returns:
        torch.Tensor: Reconstructed image.
    """
    tiles_x = original_size[1] // tile_size[1]
    tiles_y = original_size[0] // tile_size[0]
    reconstructed = torch.zeros((tiles[0].shape[1], original_size[0], original_size[1]))
    for i in range(tiles_y):
        for j in range(tiles_x):
            reconstructed[:, i * tile_size[0]:(i + 1) * tile_size[0], j * tile_size[1]:(j + 1) * tile_size[1]] = tiles[i * tiles_x + j].squeeze(0)
    return reconstructed


def load_model(file_path: str) -> CAModel:
    """
    Load a cellular automata model from a file.

    Args:
        file_path (str): Path to the model file.

    Returns:
        CAModel: Loaded model.
    """
    model = CAModel(**CAMODEL_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model


def load_and_tile_image(file_path: str, tile_size: Tuple[int, int] = (200, 200), additional_channels: int = CAMODEL_CONFIG["n_channels"] - 3) -> Tuple[List[torch.Tensor], int, int]:
    """
    Load an image and split it into tiles.

    Args:
        file_path (str): Path to the image file.
        tile_size (Tuple[int, int], optional): Tile dimensions. Defaults to (200, 200).
        additional_channels (int, optional): Number of additional channels to add.

    Returns:
        Tuple[List[torch.Tensor], int, int]: List of tile tensors, image height, and width.
    """
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"File not found: {file_path}")
    
    image_tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).unsqueeze(0)
    if (image_tensor.shape[1] == 4):
        image_tensor = image_tensor[:,:3,:,:]

    print(image_tensor.shape)
    height, width = image_tensor.shape[2], image_tensor.shape[3]
    tiles_x = width // tile_size[1]
    tiles_y = height // tile_size[0]

    tile_list = []
    for i in range(tiles_y):
        for j in range(tiles_x):
            tile = image_tensor[:, :, i * tile_size[0]:(i + 1) * tile_size[0], j * tile_size[1]:(j + 1) * tile_size[1]]
            if additional_channels > 0:
                #extra_channels = torch.randn((additional_channels, tile.shape[2], tile.shape[3])).unsqueeze(0)
                extra_channels = tile.mean(dim=1, keepdim=True).expand(-1, additional_channels, -1, -1)
                tile = torch.cat((tile, extra_channels), dim=1)
            tile_list.append(tile)
    return tile_list, height, width


def step(model: CAModel, state_tensor: torch.Tensor, device: torch.device = DEVICE) -> torch.Tensor:
    """
    Execute one simulation step using the model.

    Args:
        model (CAModel): Cellular automata model.
        state_tensor (torch.Tensor): Current state tensor.
        device (torch.device, optional): Device for computation.

    Returns:
        torch.Tensor: Next state tensor.
    """
    with torch.no_grad():
        next_state_tensor = model(state_tensor.float().to(device))
    return next_state_tensor.cpu().detach()


class NCAVisualizer(QWidget):
    """
    Cellular Automata Visualizer GUI.
    """

    def __init__(self) -> None:
        super().__init__()
        self.state: Optional[torch.Tensor] = None
        self.tiles: Optional[List[torch.Tensor]] = None
        self.steps = 0
        self.threshold = 127
        self.reconstructed_image: Optional[torch.Tensor] = None
        self.height: Optional[int] = None
        self.width: Optional[int] = None
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle("NCA Visualizer")
        self.setGeometry(100, 100, 1200, 600)

        self.layout = QVBoxLayout()

        self.canvas1 = FigureCanvas(Figure(figsize=(4, 4)))
        self.ax1 = self.canvas1.figure.subplots()
        self.ax1.axis("off")

        self.canvas2 = FigureCanvas(Figure(figsize=(4, 4)))
        self.ax2 = self.canvas2.figure.subplots()
        self.ax2.axis("off")

        self.button_layout = QHBoxLayout()
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        self.button_layout.addWidget(self.load_model_button)

        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        self.button_layout.addWidget(self.load_image_button)

        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(self.single_step)
        self.button_layout.addWidget(self.step_button)

        self.step_label = QLabel("Steps: 0")
        self.button_layout.addWidget(self.step_label)

        self.layout.addLayout(self.button_layout)

        self.grid_layout = QGridLayout()
        self.grid_layout.addWidget(self.canvas1, 0, 0)
        self.grid_layout.addWidget(self.canvas2, 0, 1)

        self.layout.addLayout(self.grid_layout)
        self.setLayout(self.layout)

    def load_model(self) -> None:
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Model File", "", "Model Files (*.pth);;All Files (*)", options=options
        )
        if file_path:
            global model1
            model1 = load_model(file_path)
            self.steps = 0
            self.step_label.setText("Steps: 0")
            if self.state is not None:
                self.update_plot(initial=True)

    def load_image(self) -> None:
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp *.tiff *.tif);;All Files (*)", options=options
        )
        if file_path:
            self.tiles, self.height, self.width = load_and_tile_image(file_path)
            self.state = self.tiles[0]
            self.update_plot(initial=True)

    def single_step(self) -> None:
        if self.state is not None and model1 is not None and self.tiles is not None:
            for idx, tile in enumerate(self.tiles):
                self.tiles[idx] = step(model1, tile)
            original_size = (self.height, self.width)
            tile_size = (self.tiles[0].shape[2], self.tiles[0].shape[3])
            self.reconstructed_image = reconstruct_image(self.tiles, original_size, tile_size)
            self.steps += 1
            self.step_label.setText(f"Steps: {self.steps}")
            self.update_plot(initial=False)

    def update_plot(self, initial: bool = False) -> None:
        if self.reconstructed_image is not None:
            self.ax1.clear()
            # Display the RGB channels of the reconstructed image
            self.ax1.imshow(self.reconstructed_image[:3, :, :].permute(1, 2, 0).numpy(), cmap="viridis")
            self.ax1.axis("off")

            self.ax2.clear()
            # Display a binary mask from the alpha channel
            alpha_channel = (self.reconstructed_image[3, :, :].numpy() * 255) > self.threshold
            self.ax2.imshow(alpha_channel, cmap="viridis")
            self.ax2.axis("off")
        self.canvas1.draw()
        self.canvas2.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    visualizer = NCAVisualizer()
    visualizer.show()
    sys.exit(app.exec_())
