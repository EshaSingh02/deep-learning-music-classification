import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MultiInputImageDataset(Dataset):
    """Dataset for samples represented by three RGB images.

    The three transformed RGB images are concatenated along the channel
    dimension to form a 9-channel tensor.
    """

    def __init__(self, root_dir, csv_file, train=True, transform=None):
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.train = train
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def load_image(self, image_path):
        image_path = image_path.lstrip("./")
        full_path = os.path.join(self.root_dir, image_path)
        return Image.open(full_path).convert("RGB")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img1 = self.load_image(
            row["input_1"] if "input_1" in row else row["input_1_path"]
        )
        img2 = self.load_image(row["input_2"])
        img3 = self.load_image(row["input_3"])

        if self.transform is not None:
            if self.train:
                # Use one random seed so the same spatial augmentation
                # is applied to all three inputs.
                seed = np.random.randint(2147483647)

                def apply_sync(img):
                    random.seed(seed)
                    torch.manual_seed(seed)
                    return self.transform(img)

                img1 = apply_sync(img1)
                img2 = apply_sync(img2)
                img3 = apply_sync(img3)
            else:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
        else:
            img1 = self.to_tensor(img1)
            img2 = self.to_tensor(img2)
            img3 = self.to_tensor(img3)

        image = torch.cat([img1, img2, img3], dim=0)

        if self.train:
            label = torch.tensor(row["target"], dtype=torch.long)
            return image, label

        return image, row["id"]


def get_transforms():
    """Return the training and validation transforms used in the notebook."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform
