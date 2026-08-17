import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Subset

from .dataset import MultiInputImageDataset, get_transforms
from .model import CNN_Scratch


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    print("\n--- Validation Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        zero_division=0,
    ))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix: Validation Dataset")
    plt.tight_layout()
    plt.show()

    return f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_transform = get_transforms()

    train_csv = f"{args.data_root}/dataset_music/train/metadata.csv"

    dataset = MultiInputImageDataset(
        root_dir=args.data_root,
        csv_file=train_csv,
        train=True,
        transform=val_transform,
    )

    # Same deterministic split used in the supplied notebook.
    import numpy as np
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    val_indices = indices[20000:]

    val_dataset = Subset(dataset, val_indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CNN_Scratch(num_classes=16).to(device)
    load_checkpoint(model, args.checkpoint, device)
    evaluate_model(model, val_loader, device)


if __name__ == "__main__":
    main()
