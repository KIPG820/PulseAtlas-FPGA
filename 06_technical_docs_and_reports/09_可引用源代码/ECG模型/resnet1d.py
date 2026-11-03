"""Small fixed-shape multi-label 1D ResNet for 12-lead ECG."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7, stride=stride,
            padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=7, stride=1,
            padding=3, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.projection = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class ECGResNet1D(nn.Module):
    """Input [batch, 12, 5000], output five independent class logits."""

    def __init__(self, in_channels: int = 12, num_labels: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, stride=4,
                      padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock1D(16, 16),
            ResidualBlock1D(16, 16),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock1D(16, 32, stride=2),
            ResidualBlock1D(32, 32),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 64),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(64, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(self.dropout(x))
