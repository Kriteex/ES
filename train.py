"""
Training script for various segmentation and CA models.
"""

import os
import random
from typing import Tuple

import albumentations as A
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset
from segmentation_models_pytorch import PSPNet
from torchvision.models.segmentation import deeplabv3_resnet50

from config import (
    CAMODEL_CONFIG,
    DEEPLABV3_CONFIG,
    DEEPLABV3_CONFIG_CONV1,
    PSPNET_CONFIG,
    TRAINING_CONFIG,
    UNET_CONFIG,
    SEGNET_CONFIG,
    DEVICE,
    SEED,
)
from flexible_dataset import FlexibleDataset
from model import CAModel, UNetTiny, UNetNormal, SegNet
from utils import (
    calculate_pixel_accuracy,
    composite_loss,
    get_dataloader,
    log_loss,
    plot_batch_x_channels,
)

# Set seeds for reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)




def initialize_model(model_name: str) -> torch.nn.Module:
    """Initialize the model based on the provided model name."""
    if model_name == "unet_tiny":
        return UNetTiny(**UNET_CONFIG).to(DEVICE)
    elif model_name == "unet_normal":
        return UNetNormal(**UNET_CONFIG).to(DEVICE)
    elif model_name == "nca":
        return CAModel(**CAMODEL_CONFIG).to(DEVICE)
    elif model_name == "segnet":
        return SegNet(**SEGNET_CONFIG).to(DEVICE)
    elif model_name == "deeplabv3":
        model = deeplabv3_resnet50(**DEEPLABV3_CONFIG)
        model.backbone.conv1 = torch.nn.Conv2d(**DEEPLABV3_CONFIG_CONV1)
        return model.to(DEVICE)
    elif model_name == "pspnet":
        return PSPNet(**PSPNET_CONFIG).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function,
    device: torch.device,
    model_name: str,
) -> Tuple[float, float, float, float, float]:
    """Train the model for one epoch and return training metrics."""
    model.train()
    total_loss, total_accuracy = 0.0, 0.0
    all_preds, all_targets = [], []

    for x, y in dataloader:
        x, y = x.float().to(device), y.float().to(device)
        optimizer.zero_grad()

        if model_name == "nca":
            loss = 0.0
            for _ in range(CAMODEL_CONFIG["steps"]):
                x = model(x)
                loss += composite_loss(x[:, 3], y, pixel_accuracy_weight=1)
            #Uncomment to plot batch 
            #plot_batch_x_channels(x.detach().cpu(), y.detach().cpu(), "post_model")
            correct, _ = calculate_pixel_accuracy(x[:, 3], y)
            preds = torch.sigmoid(x[:, 3]).detach().cpu().numpy().flatten()
        elif model_name == "deeplabv3":
            output = model(x)["out"]
            loss = loss_function(output, y)
            correct, _ = calculate_pixel_accuracy(torch.sigmoid(output), y)
            preds = torch.sigmoid(output).detach().cpu().numpy().flatten()
        else:
            output = model(x)
            #print(output.shape) #(5,1,400,400)

            import matplotlib.pyplot as plt
            if (random.random() > 1.99):
                # ─── Vyber jeden vzorek z batche ───────────────────────────────
                inp  = x[0]                  # (C, H, W)
                gt   = y[0, 0]               # (H, W)
                pred = torch.sigmoid(output[0, 0])   # (H, W) – pravděpodobnosti

                # ─── Na CPU & NumPy ────────────────────────────────────────────
                if inp.shape[0] == 3:        # RGB obraz
                    inp_img = inp.cpu().permute(1, 2, 0).numpy()        # (H, W, 3)
                else:                        # jednokanálový (např. čb)
                    inp_img = inp[0].cpu().numpy()                      # (H, W)

                gt_mask   = gt.cpu().numpy()
                pred_mask = pred.detach().cpu().numpy()

                # ─── Plot ──────────────────────────────────────────────────────
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))

                # Input
                ax[0].imshow(inp_img if inp.shape[0] == 3 else inp_img, cmap=None if inp.shape[0] == 3 else 'gray')
                ax[0].set_title("Input")

                # Ground-truth
                ax[1].imshow(gt_mask, cmap='gray')
                ax[1].set_title("Ground Truth")

                # Prediction
                ax[2].imshow(pred_mask, cmap='gray')
                ax[2].set_title("Prediction")

                for a in ax:
                    a.axis('off')

                plt.tight_layout()
                plt.show()

            loss = loss_function(output, y)
            correct, _ = calculate_pixel_accuracy(torch.sigmoid(output), y)
            preds = torch.sigmoid(output).detach().cpu().numpy().flatten()

        targets = y.cpu().numpy().flatten()
        all_preds.extend(preds)
        all_targets.extend(targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += correct

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    all_preds = (np.array(all_preds) > 0.5).astype(int)
    all_targets = (np.array(all_targets) > 0.5).astype(int)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    return avg_loss, avg_accuracy, precision, recall, f1


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_function,
    device: torch.device,
    model_name: str,
) -> Tuple[float, float, float, float, float]:
    """Evaluate the model and return evaluation metrics."""
    model.eval()
    total_loss, total_accuracy = 0.0, 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.float().to(device), y.float().to(device)
            if model_name == "nca":
                loss = 0.0
                for _ in range(CAMODEL_CONFIG["steps"]):
                    x = model(x)
                    loss += composite_loss(x[:, 3], y, pixel_accuracy_weight=0.1)
                correct, _ = calculate_pixel_accuracy(x[:, 3], y)
                preds = torch.sigmoid(x[:, 3]).cpu().numpy().flatten()
            elif model_name == "deeplabv3":
                output = model(x)["out"]
                loss = loss_function(output, y)
                correct, _ = calculate_pixel_accuracy(torch.sigmoid(output), y)
                preds = torch.sigmoid(output).cpu().numpy().flatten()
            else:
                output = model(x)
                loss = loss_function(output, y)
                correct, _ = calculate_pixel_accuracy(torch.sigmoid(output), y)
                preds = torch.sigmoid(output).cpu().numpy().flatten()

                import matplotlib.pyplot as plt
                if (PICA == 2):
                    # ─── Vyber jeden vzorek z batche ───────────────────────────────
                    inp  = x[0]                  # (C, H, W)
                    gt   = y[0, 0]               # (H, W)
                    pred = torch.sigmoid(output[0, 0])   # (H, W) – pravděpodobnosti

                    # ─── Na CPU & NumPy ────────────────────────────────────────────
                    if inp.shape[0] == 3:        # RGB obraz
                        inp_img = inp.cpu().permute(1, 2, 0).numpy()        # (H, W, 3)
                    else:                        # jednokanálový (např. čb)
                        inp_img = inp[0].cpu().numpy()                      # (H, W)

                    gt_mask   = gt.cpu().numpy()
                    pred_mask = pred.detach().cpu().numpy()

                    # ─── Plot ──────────────────────────────────────────────────────
                    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

                    # Input
                    ax[0].imshow(inp_img if inp.shape[0] == 3 else inp_img, cmap=None if inp.shape[0] == 3 else 'gray')
                    ax[0].set_title("Input")

                    # Ground-truth
                    ax[1].imshow(gt_mask, cmap='gray')
                    ax[1].set_title("Ground Truth")

                    # Prediction
                    ax[2].imshow(pred_mask, cmap='gray')
                    ax[2].set_title("Prediction")

                    for a in ax:
                        a.axis('off')

                    plt.tight_layout()
                    plt.show()

            targets = y.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(targets)
            total_loss += loss.item()
            total_accuracy += correct

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    all_preds = (np.array(all_preds) > 0.5).astype(int)
    all_targets = (np.array(all_targets) > 0.5).astype(int)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    return avg_loss, avg_accuracy, precision, recall, f1


def main(model_name: str = "nca", run: int = 1) -> None:
    """
    Main training loop.

    Args:
        model_name (str): Name of the model to train.
        run (int): Run identifier for logging.
    """
    global PICA
    PICA = 0
    if model_name in ["unet_tiny", "unet_normal"]:
        n_channels, n_classes = UNET_CONFIG["n_channels"], UNET_CONFIG["n_classes"]
    elif model_name == "nca":
        n_channels, n_classes = CAMODEL_CONFIG["n_channels"], 1
    elif model_name == "segnet":
        n_channels, n_classes = SEGNET_CONFIG["n_channels"], SEGNET_CONFIG["n_classes"]
    elif model_name == "deeplabv3":
        n_channels, n_classes = DEEPLABV3_CONFIG["n_channels"], DEEPLABV3_CONFIG["num_classes"]
    elif model_name == "pspnet":
        n_channels, n_classes = PSPNET_CONFIG["in_channels"], PSPNET_CONFIG["classes"]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    batch_size = TRAINING_CONFIG["batch_size"]
    epochs = TRAINING_CONFIG["epochs"]
    crop_size = 200

    model = initialize_model(model_name)
    if model_name != "deeplabv3":
        summary(model, (n_channels, crop_size, crop_size))

    optimizer = TRAINING_CONFIG["optimizer"](model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    scheduler = TRAINING_CONFIG["scheduler"](
        optimizer,
        step_size=TRAINING_CONFIG["scheduler_step_size"],
        gamma=TRAINING_CONFIG["scheduler_gamma"],
    )
    loss_function = TRAINING_CONFIG["loss_function"] if model_name in [
        "unet_tiny",
        "unet_normal",
        "segnet",
        "deeplabv3",
        "pspnet",
    ] else None
    filter_masks = TRAINING_CONFIG["dataset_name"] == "dibco"

    dataloader_train = get_dataloader(
        TRAINING_CONFIG["train_images_dir"],
        TRAINING_CONFIG["train_masks_dir"],
        n_channels,
        batch_size,
        crop_size,
        filter_masks,
    )
    if TRAINING_CONFIG["dataset_name"] in ["dibco", "trees"]:
        images_list, masks_list = [], []
        for imgs, msks in dataloader_train:
            images_list.append(imgs)
            masks_list.append(msks)
        images_all = torch.cat(images_list)
        masks_all = torch.cat(masks_list)
        total_samples = len(images_all)
        indices = torch.randperm(total_samples)
        split = int(0.5 * total_samples)
        train_subset = TensorDataset(images_all[indices[:split]], masks_all[indices[:split]])
        test_subset = TensorDataset(images_all[indices[split:]], masks_all[indices[split:]])
        dataloader_train = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        dataloader_test = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    else:
        dataloader_test = get_dataloader(
            TRAINING_CONFIG["test_images_dir"],
            TRAINING_CONFIG["test_masks_dir"],
            n_channels,
            batch_size,
            crop_size,
            filter_masks,
        )

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        if epoch == 0:
            test_metrics = evaluate_model(model, dataloader_test, loss_function, DEVICE, model_name)
            print(f"Test: Loss={test_metrics[0]:.4f}, Acc={test_metrics[1]:.2f}%, Prec={test_metrics[2]:.4f}, Recall={test_metrics[3]:.4f}, F1={test_metrics[4]:.4f}")
        train_metrics = train_one_epoch(model, dataloader_train, optimizer, loss_function, DEVICE, model_name)
        test_metrics = evaluate_model(model, dataloader_test, loss_function, DEVICE, model_name)
        log_loss(
            epoch,
            {"Loss": train_metrics[0], "Accuracy": train_metrics[1], "Precision": train_metrics[2], "Recall": train_metrics[3], "F1 Score": train_metrics[4]},
            {"Loss": test_metrics[0], "Accuracy": test_metrics[1], "Precision": test_metrics[2], "Recall": test_metrics[3], "F1 Score": test_metrics[4]},
            model_name,
            run,
            print_results=True,
        )
        scheduler.step()
        PICA += 1


if __name__ == "__main__":
    for i in range(1):
        main(model_name="nca", run=i)
