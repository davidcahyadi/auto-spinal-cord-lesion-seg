from typing import Optional

import torch
from torch import nn
from torchvision.transforms.functional import center_crop

from src.model.base import Model
from src.schema import SegmentationConfig


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self._conv_1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False)
        self._conv_2 = nn.Conv2d(
            out_channels, out_channels, 3, 1, padding=1, bias=False
        )
        self._relu = nn.ReLU(inplace=True)
        if use_batch_norm:
            self._bn_1 = nn.BatchNorm2d(out_channels)
            self._bn_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self._conv_1(x)
        if self.use_batch_norm:
            x = self._bn_1(x)
        x = self._relu(x)
        x = self._conv_2(x)
        if self.use_batch_norm:
            x = self._bn_2(x)
        x = self._relu(x)
        return x


class DownBlock(Block):
    def __init__(self, in_channels, out_channels, use_batch_norm=True):
        super().__init__(in_channels, out_channels, use_batch_norm)
        self._pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = super().forward(x)
        return self._pool(x), x


class UpBlock(Block):
    def __init__(self, in_channels, out_channels, use_batch_norm=True):
        super().__init__(in_channels, out_channels, use_batch_norm)
        self._conv_up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )

    def forward(self, x, skip):
        x = self._conv_up(x)
        skip = center_crop(skip, [x.shape[-2], x.shape[-1]])
        x = torch.cat([x, skip], dim=1)
        return super().forward(x)


class UNETConfig(SegmentationConfig):
    in_channels: int = 1
    out_channels: int = 1
    depth: int = 4
    features: int = 64
    out_activation: Optional[str] = "sigmoid"
    use_batch_norm: bool = False


class UNET(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.input_layer = nn.Conv2d(
            self.hparams.in_channels,
            self.hparams.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # down part
        in_feature = self.hparams.in_channels
        out_feature = self.hparams.features
        for _ in range(self.hparams.depth):
            self.downs.append(
                DownBlock(in_feature, out_feature, self.hparams.use_batch_norm)
            )
            in_feature = out_feature
            out_feature = out_feature * 2

        # middle
        self.bottleneck = Block(in_feature, out_feature, self.hparams.use_batch_norm)

        # up part
        for _ in range(self.hparams.depth):
            in_feature = out_feature
            out_feature = in_feature // 2
            self.ups.append(
                UpBlock(in_feature, out_feature, self.hparams.use_batch_norm)
            )

        self.output_layer = nn.Conv2d(
            self.hparams.features, self.hparams.out_channels, kernel_size=1
        )
        self.progression = []

        if self.hparams.out_activation is None:
            self.out_activation = nn.Identity()
        elif self.hparams.out_activation.lower() == "sigmoid":
            self.out_activation = nn.Sigmoid()
        elif self.hparams.out_activation.lower() == "softmax":
            self.out_activation = nn.Softmax(dim=1)

    def forward(self, x):
        skip_connections = []
        x = self.input_layer(x)

        for down in self.downs:
            x, x_skip = down(x)
            skip_connections.append(x_skip)

        x = self.bottleneck(x)

        for up in self.ups:
            x = up(x, skip_connections.pop())

        x = self.output_layer(x)
        x = self.out_activation(x)
        return x
