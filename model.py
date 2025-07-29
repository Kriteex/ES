"""
Model definitions for various segmentation and cellular automata architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetTiny(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = True) -> None:
        """
        A tiny U-Net implementation.

        Args:
            n_channels (int): Number of input channels.
            n_classes (int): Number of output classes.
            bilinear (bool): Whether to use bilinear upsampling.
        """
        super().__init__()
        self.bilinear = bilinear
        self.inc = self._double_conv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), self._double_conv(64, 128))
        self.up1 = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            if bilinear
            else nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        )
        self.up_conv = self._double_conv(128 + 64, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def _double_conv(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Apply two consecutive convolution layers with batch normalization and ReLU.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: Sequential container of layers.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2_up = self.up1(x2)
        diffY = x1.size(2) - x2_up.size(2)
        diffX = x1.size(3) - x2_up.size(3)
        x2_up = F.pad(x2_up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = self.up_conv(torch.cat([x2_up, x1], dim=1))
        return self.outc(x)


class UNetNormal(nn.Module):
    def __init__(self, n_channels: int, n_classes: int) -> None:
        """
        A normal U-Net implementation with deeper architecture.

        Args:
            n_channels (int): Number of input channels.
            n_classes (int): Number of output classes.
        """
        super().__init__()
        self.inc = self._double_conv(n_channels, 64)
        self.down1 = self._down(64, 128)
        self.down2 = self._down(128, 256)
        self.down3 = self._down(256, 512)
        self.down4 = self._down(512, 1024)
        self.up1 = self._up(1024, 512)
        self.up2 = self._up(512, 256)
        self.up3 = self._up(256, 128)
        self.up4 = self._up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def _double_conv(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _down(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(nn.MaxPool2d(2), self._double_conv(in_channels, out_channels))

    def _up(self, in_channels: int, out_channels: int) -> nn.ModuleDict:
        upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        conv = self._double_conv(in_channels, out_channels)
        return nn.ModuleDict({"upsample": upsample, "conv": conv})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self._up_concat(x5, x4, self.up1)
        x = self._up_concat(x, x3, self.up2)
        x = self._up_concat(x, x2, self.up3)
        x = self._up_concat(x, x1, self.up4)
        return self.outc(x)

    def _up_concat(self, x1: torch.Tensor, x2: torch.Tensor, up_block: nn.ModuleDict) -> torch.Tensor:
        x1 = up_block["upsample"](x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return up_block["conv"](torch.cat([x2, x1], dim=1))


class SegNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int) -> None:
        """
        SegNet implementation for segmentation tasks.

        Args:
            n_channels (int): Number of input channels.
            n_classes (int): Number of output classes.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=True),
        )
        self.decoder = nn.Sequential(
            nn.MaxUnpool2d(2, stride=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, indices = self.encoder(x)
        x = self.decoder[0](x, indices)
        return self.decoder[1:](x)


class CAModel(nn.Module):
    def __init__(
        self,
        n_channels: int = 16,
        hidden_channels: int = 128,
        fire_rate: float = 0.5,
        device: torch.device = None,
        neighbour: int = 3,
        deep_perceive: int = 1,
        deep_update: int = 1,
        use_residual: bool = True,
        steps: int = 1,
    ) -> None:
        """
        Cellular Automata (CA) Model.

        Args:
            n_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            fire_rate (float): Stochastic update probability.
            device (torch.device, optional): Device to use.
            neighbour (int): Neighbourhood size.
            deep_perceive (int): Depth of perception layers.
            deep_update (int): Depth of update layers.
            use_residual (bool): Use residual connections.
            steps (int): Number of simulation steps.
        """
        super().__init__()
        self.fire_rate = fire_rate
        self.steps = steps
        self.device = device or torch.device("cpu")
        self.use_residual = use_residual
        self.perceive_conv = self._build_perception_layers(n_channels, neighbour, deep_perceive)
        self.update_module = self._build_update_layers(n_channels, hidden_channels, neighbour, deep_update)
        if use_residual:
            self.residual_conv = nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False)
        self.to(self.device)

    def _build_perception_layers(self, n_channels: int, neighbour: int, depth: int) -> nn.Sequential:
        layers = []
        for i in range(depth):
            kernel_size = max(1, neighbour - i * 2)
            padding = max(0, (neighbour - 1 - i * 2) // 2)
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding, groups=n_channels, bias=False))
            if i != depth - 1:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _build_update_layers(self, n_channels: int, hidden_channels: int, neighbour: int, depth: int) -> nn.Sequential:
        layers = []
        for i in range(depth):
            in_channels = n_channels if i == 0 else hidden_channels
            out_channels = n_channels if i == depth - 1 else hidden_channels
            groups = n_channels if i == 0 else 1
            kernel_size = max(1, neighbour - i * 2)
            padding = max(0, (neighbour - 1 - i * 2) // 2)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=groups, bias=False))
            if i != depth - 1:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        """Apply perception layers."""
        return self.perceive_conv(x)

    def update(self, x: torch.Tensor) -> torch.Tensor:
        """Update the state with optional residual connection."""
        residual = x
        x = self.update_module(x)
        return x + self.residual_conv(residual) if self.use_residual else x

    @staticmethod
    def stochastic_update(x: torch.Tensor, fire_rate: float) -> torch.Tensor:
        """Apply stochastic update based on fire_rate."""
        mask = (torch.rand(x[:, :1].shape, device=x.device) <= fire_rate).float()
        return x * mask

    @staticmethod
    def get_living_mask(x: torch.Tensor) -> torch.Tensor:
        """Compute a living mask based on the first channel."""
        return nn.functional.max_pool2d(x[:, :1], kernel_size=3, stride=1, padding=1) > 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x_fixed, x_mod = x[:3], x[3:]
        else:
            x_fixed, x_mod = x[:, :3], x[:, 3:]
        pre_mask = self.get_living_mask(x_mod)
        dx = self.stochastic_update(self.update(self.perceive(x)), self.fire_rate)
        if x.dim() == 3:
            x_mod = x_mod + dx[3:]
            out = torch.cat([x_fixed, x_mod], dim=0)
        else:
            x_mod = x_mod + dx[:, 3:]
            out = torch.cat([x_fixed, x_mod], dim=1)
        life_mask = (pre_mask & self.get_living_mask(x_mod)).float()
        return torch.cat([x_fixed, x_mod * life_mask], dim=0 if x.dim() == 3 else 1)
