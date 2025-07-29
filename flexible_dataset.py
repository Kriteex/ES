# flexible_dataset.py
"""
A flexible dataset class for image a mask loading s optional transformacemi.
"""

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.number_of_channels = number_of_channels
        self.alpha = alpha
        self.alpha_init = alpha_init

        self.images: List[str] = []
        self.masks: List[str] = []

        # sběr masek
        for root, _, files in os.walk(masks_dir):
            for file in files:
                if not filter_masks or file.lower().endswith(("gt.tiff", "gt.tif", "gt.bmp", "gt.png")):
                    self.masks.append(os.path.join(root, file))
        # sběr obrázků
        for root, _, files in os.walk(images_dir):
            for file in files:
                if filter_masks:
                    if file.lower().endswith((".bmp", ".png", ".tif", ".jpg", ".tiff")) and \
                       os.path.join(root, file) not in self.masks:
                        self.images.append(os.path.join(root, file))
                else:
                    self.images.append(os.path.join(root, file))

        self.images.sort()
        self.masks.sort()

        self.images = self.images
        self.masks = self.masks


    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        image = np.array(Image.open(img_path))[:, :, :3]
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = mask.mean(axis=2)
        mask = np.expand_dims(mask, axis=2)

        data = self.transform(image=image, mask=mask) if self.transform else {"image": image, "mask": mask}
        image_tensor = torch.from_numpy(data["image"] / 255.0).permute(2, 0, 1)
        mask_tensor  = torch.from_numpy(data["mask"]).permute(2, 0, 1).float()
        mask_tensor  = (mask_tensor > 0).float()

        if self.alpha:
            shape = (1, *image_tensor.shape[1:])
            if self.alpha_init == 1:
                alpha = torch.randn(shape)
            elif self.alpha_init == 2:
                alpha = torch.ones(shape)
            elif self.alpha_init == 3:
                alpha = torch.normal(mean=0.5, std=0.1, size=shape)
            elif self.alpha_init == 4:
                alpha = torch.bernoulli(torch.full(shape, 0.5))
            else:
                alpha = image_tensor[:1].mean(dim=0, keepdim=True)
            image_tensor = torch.cat((image_tensor, alpha), dim=0)

        if self.number_of_channels > image_tensor.shape[0]:
            extra = self.number_of_channels - image_tensor.shape[0]
            mean_ch = image_tensor[:3].mean(dim=0, keepdim=True)
            image_tensor = torch.cat((image_tensor, mean_ch.repeat(extra, 1, 1)), dim=0)

        return image_tensor, mask_tensor
