"""Compact classifier definitions used in the IS-ML paper."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfigurableSimpleCNN(nn.Module):
    """Three-block convolutional classifier with configurable padding."""

    def __init__(
        self,
        num_classes: int = 15,
        in_channels: int = 1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding="same", padding_mode=padding_mode),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same", padding_mode=padding_mode),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding="same", padding_mode=padding_mode),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ConfigurableBasicBlock(nn.Module):
    """Residual block with configurable convolution padding."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            padding_mode=padding_mode,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding="same",
            bias=False,
            padding_mode=padding_mode,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class ConfigurableResNet(nn.Module):
    """Minimal ResNet-18 with configurable boundary convention."""

    def __init__(
        self,
        block: type[ConfigurableBasicBlock],
        layers: list[int],
        num_classes: int = 15,
        in_channels: int = 1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.in_planes = 64
        self.padding_mode = padding_mode
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            padding_mode=padding_mode,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0], padding_mode=padding_mode)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, padding_mode=padding_mode)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, padding_mode=padding_mode)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, padding_mode=padding_mode)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self,
        block: type[ConfigurableBasicBlock],
        planes: int,
        num_blocks: int,
        stride: int = 1,
        padding_mode: str = "zeros",
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample, padding_mode=padding_mode)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, padding_mode=padding_mode))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        if self.padding_mode == "circular":
            x = F.pad(x, (1, 1, 1, 1), mode="circular")
        else:
            x = F.pad(x, (1, 1, 1, 1), mode="constant", value=0.0)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def ConfigurableResNet18(num_classes: int = 15, in_channels: int = 1, padding_mode: str = "zeros") -> nn.Module:
    """Build the ResNet-18 classifier used in the paper."""

    return ConfigurableResNet(
        ConfigurableBasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        in_channels=in_channels,
        padding_mode=padding_mode,
    )


MODEL_REGISTRY = {
    "SimpleCNN": ConfigurableSimpleCNN,
    "ResNet18": ConfigurableResNet18,
}
