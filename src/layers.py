import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D_Scratch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size,
            )
            * math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x_unf = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        weights = self.weight.view(self.weight.size(0), -1)
        out = weights @ x_unf
        out = out + self.bias.view(1, -1, 1)

        height_out = (
            height + 2 * self.padding - self.kernel_size
        ) // self.stride + 1
        width_out = (
            width + 2 * self.padding - self.kernel_size
        ) // self.stride + 1

        return out.view(batch_size, -1, height_out, width_out)


class MaxPool2D_Scratch(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x_unf = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

        x_unf = x_unf.view(
            batch_size,
            channels,
            self.kernel_size * self.kernel_size,
            -1,
        )
        out, _ = x_unf.max(dim=2)

        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1

        return out.view(batch_size, channels, height_out, width_out)


class ReLU_Scratch(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0.0)


class BatchNorm2D_Scratch(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)

            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * var
            )
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (
            x - mean.view(1, -1, 1, 1)
        ) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)

        return (
            self.gamma.view(1, -1, 1, 1) * x_hat
            + self.beta.view(1, -1, 1, 1)
        )


class Linear_Scratch(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features)
            * math.sqrt(2.0 / in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.t() + self.bias


class AdaptiveAvgPool2D_Scratch(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.out_h, self.out_w = output_size

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        stride_h = height // self.out_h
        stride_w = width // self.out_w

        kernel_h = height - (self.out_h - 1) * stride_h
        kernel_w = width - (self.out_w - 1) * stride_w

        patches = F.unfold(
            x,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
        )

        patches = patches.view(
            batch_size,
            channels,
            kernel_h * kernel_w,
            self.out_h * self.out_w,
        )

        out = patches.mean(dim=2)
        return out.view(batch_size, channels, self.out_h, self.out_w)
