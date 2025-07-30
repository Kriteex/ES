# flexible_dataset.py
"""
Custom PyTorch Dataset that loads image/mask pairs with optional alpha and
additional channels.

Images and masks are collected recursively from their respective directories.
If `filter_masks` is enabled, only files ending with typical GT suffixes are
treated as masks and any unmatched image file in the masks directory is
reassigned to the images set.

During access, each sample is optionally transformed (e.g., random crop),
normalized to [0,1], extended with an alpha channel computed from the image,
and padded with replicated mean channels to reach the desired number of channels.
"""

import os
from typing import Callable, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class FlexibleDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        number_of_channels: int = 4,
        transform: Optional[Callable] = None,
        alpha: bool = True,
        alpha_init: Optional[int] = None,
        filter_masks: bool = False,
    ) -> None:
        """
        Initialize dataset by indexing all image and mask files.

        Args:
            images_dir: Root directory containing input images.
            masks_dir: Root directory containing mask images.
            number_of_channels: Desired number of output channels (≥3).
            transform: Optional Albumentations transform applied to both image and mask.
            alpha: If True, append an alpha channel computed from the image.
            alpha_init: Optional scheme for alpha initialization.  None uses mean of RGB.
            filter_masks: If True, treat the mask directory as superset and filter by suffix.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.number_of_channels = number_of_channels
        self.alpha = alpha
        self.alpha_init = alpha_init

        # Build sorted lists of image and mask file paths
        self.images: List[str] = []
        self.masks: List[str] = []
        mask_suffixes = ("gt.tiff", "gt.tif", "gt.bmp", "gt.png")

        for root, _, files in os.walk(masks_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                if not filter_masks or fname.lower().endswith(mask_suffixes):
                    self.masks.append(full_path)

        for root, _, files in os.walk(images_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                if filter_masks:
                    if fname.lower().endswith((".bmp", ".png", ".tif", ".jpg", ".tiff")) and \
                            full_path not in self.masks:
                        self.images.append(full_path)
                else:
                    self.images.append(full_path)

        self.images.sort()
        self.masks.sort()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load an image/mask pair, apply transforms, normalize, and pad channels.

        Args:
            idx: Index of the sample to fetch.

        Returns:
            A tuple of (image_tensor, mask_tensor) with shapes
            (C, H, W) and (1, H, W), respectively.
        """
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load image as RGB and mask as single‑channel
        image = np.array(Image.open(img_path))[:, :, :3]
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = mask.mean(axis=2)
        mask = mask[:, :, None]

        # Apply transform if provided
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image_np = augmented["image"]
            mask_np = augmented["mask"]
        else:
            image_np, mask_np = image, mask

        # Convert to tensors and normalize
        image_tensor = torch.from_numpy(image_np / 255.0).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_np).permute(2, 0, 1).float()
        mask_tensor = (mask_tensor > 0).float()  # binarize

        # Append alpha channel if requested
        if self.alpha:
            shape = (1, *image_tensor.shape[1:])
            if self.alpha_init == 1:
                alpha_channel = torch.randn(shape)
            elif self.alpha_init == 2:
                alpha_channel = torch.ones(shape)
            elif self.alpha_init == 3:
                alpha_channel = torch.normal(mean=0.5, std=0.1, size=shape)
            elif self.alpha_init == 4:
                alpha_channel = torch.bernoulli(torch.full(shape, 0.5))
            else:
                alpha_channel = image_tensor[:1].mean(dim=0, keepdim=True)
            image_tensor = torch.cat((image_tensor, alpha_channel), dim=0)

        # Pad with replicated mean channels to reach desired channel count
        if self.number_of_channels > image_tensor.size(0):
            extra_channels = self.number_of_channels - image_tensor.size(0)
            mean_channel = image_tensor[:3].mean(dim=0, keepdim=True)
            image_tensor = torch.cat(
                (image_tensor, mean_channel.repeat(extra_channels, 1, 1)), dim=0
            )

        return image_tensor, mask_tensor
