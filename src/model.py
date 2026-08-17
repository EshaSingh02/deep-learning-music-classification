import torch.nn as nn

from .layers import (
    AdaptiveAvgPool2D_Scratch,
    BatchNorm2D_Scratch,
    Conv2D_Scratch,
    Linear_Scratch,
    MaxPool2D_Scratch,
    ReLU_Scratch,
)


class CNN_Scratch(nn.Module):
    """Six-block CNN used in the supplied music-classification notebook."""

    def __init__(self, num_classes=16):
        super().__init__()

        self.conv1 = Conv2D_Scratch(9, 16, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2D_Scratch(16)
        self.pool1 = MaxPool2D_Scratch(2, 2)

        self.conv2 = Conv2D_Scratch(16, 32, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2D_Scratch(32)
        self.pool2 = MaxPool2D_Scratch(2, 2)

        self.conv3 = Conv2D_Scratch(32, 64, kernel_size=3, padding=1)
        self.bn3 = BatchNorm2D_Scratch(64)
        self.pool3 = MaxPool2D_Scratch(2, 2)

        self.conv4 = Conv2D_Scratch(64, 96, kernel_size=3, padding=1)
        self.bn4 = BatchNorm2D_Scratch(96)
        self.pool4 = MaxPool2D_Scratch(2, 2)

        self.conv5 = Conv2D_Scratch(96, 128, kernel_size=3, padding=1)
        self.bn5 = BatchNorm2D_Scratch(128)
        self.pool5 = MaxPool2D_Scratch(2, 2)

        self.conv6 = Conv2D_Scratch(128, 192, kernel_size=3, padding=1)
        self.bn6 = BatchNorm2D_Scratch(192)
        self.pool6 = MaxPool2D_Scratch(2, 2)

        self.relu = ReLU_Scratch()
        self.adaptive_pool = AdaptiveAvgPool2D_Scratch((1, 1))

        self.fc1 = Linear_Scratch(192, 128)
        self.fc2 = Linear_Scratch(128, num_classes)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = self.pool5(self.relu(self.bn5(self.conv5(x))))
        x = self.pool6(self.relu(self.bn6(self.conv6(x))))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)
