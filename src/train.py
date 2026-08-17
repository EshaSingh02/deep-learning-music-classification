import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from .dataset import MultiInputImageDataset, get_transforms
from .model import CNN_Scratch


def train_and_validate(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=80,
    checkpoint_path="best_model.pth",
):
    train_losses, val_losses, val_f1s = [], [], []
    best_f1 = 0.0

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        threshold=1e-3,
        min_lr=5e-7,
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp,
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=use_amp,
                ):
                    outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        f1 = f1_score(all_labels, all_preds, average="macro")
        val_f1s.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_f1s": val_f1s,
                "best_f1": best_f1,
            }
            torch.save(checkpoint, checkpoint_path)

        scheduler.step(f1)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {f1:.4f}"
        )

    return train_losses, val_losses, val_f1s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--train-csv",
        default=None,
        help="Path to training metadata.csv.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint", default="best_model.pth")
    args = parser.parse_args()

    train_csv = args.train_csv or os.path.join(
        args.data_root, "dataset_music", "train", "metadata.csv"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform, val_transform = get_transforms()

    train_full = MultiInputImageDataset(
        root_dir=args.data_root,
        csv_file=train_csv,
        train=True,
        transform=train_transform,
    )

    val_full = MultiInputImageDataset(
        root_dir=args.data_root,
        csv_file=train_csv,
        train=True,
        transform=val_transform,
    )

    np.random.seed(42)
    indices = np.random.permutation(len(train_full))
    train_indices = indices[:20000]
    val_indices = indices[20000:]

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CNN_Scratch(num_classes=16).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-4,
    )

    train_and_validate(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
