from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


def _pool(kind: str, kernel: tuple[int, int], stride: tuple[int, int]) -> nn.Module:
    normalized = kind.lower()
    if normalized == "mean":
        return nn.AvgPool2d(kernel_size=kernel, stride=stride)
    if normalized == "max":
        return nn.MaxPool2d(kernel_size=kernel, stride=stride)
    raise ValueError(f"Unsupported pooling mode: {kind}")


def _group_count(channels: int, first_block: bool = False) -> int:
    if first_block:
        return max(1, channels // 2 if channels % 10 == 0 else channels // 5)
    return max(1, channels // 2)


class TemporalSpatialBranch(nn.Module):
    def __init__(
        self,
        n_channels: int,
        filters: int,
        temporal_kernel: int,
        pool_length: int,
        pool_stride: int,
        activation: str,
        normalization: str,
        pool_mode: str,
    ) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(
            1,
            filters,
            kernel_size=(temporal_kernel, 1),
            stride=(1, 1),
            padding="same",
            bias=True,
        )
        self.spatial = nn.Conv2d(
            filters,
            filters,
            kernel_size=(1, n_channels),
            stride=(1, 1),
            bias=True,
        )
        if normalization.lower() != "group":
            raise ValueError(
                "The published final Multi-BK-Net configuration uses GroupNorm."
            )
        self.norm = nn.GroupNorm(_group_count(filters, first_block=True), filters)
        self.activation = _activation(activation)
        self.pool = _pool(pool_mode, (pool_length, 1), (pool_stride, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.norm(x)
        x = self.activation(x)
        return self.pool(x)


class LaterConvPoolBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        filter_length: int,
        conv_stride: int,
        pool_length: int,
        pool_stride: int,
        dropout: float,
        activation: str,
        normalization: str,
        pool_mode: str,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(
            in_filters,
            out_filters,
            kernel_size=(filter_length, 1),
            stride=(conv_stride, 1),
            bias=False,
        )
        if normalization.lower() != "group":
            raise ValueError(
                "The published final Multi-BK-Net configuration uses GroupNorm."
            )
        self.norm = nn.GroupNorm(_group_count(out_filters), out_filters)
        self.activation = _activation(activation)
        self.pool = _pool(pool_mode, (pool_length, 1), (pool_stride, 1))
        self.post_pool_activation = _activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return self.post_pool_activation(x)


class MultiBKNet(nn.Module):
    """Published Multi-BK-Net architecture with a configurable spatial channel count."""

    def __init__(
        self,
        n_channels: int,
        input_samples: int = 6000,
        n_classes: int = 2,
        total_first_block_filters: int = 35,
        temporal_kernels: Iterable[int] = (200, 25, 13, 7, 3),
        first_pool_length: int = 50,
        first_pool_stride: int = 15,
        later_filter_length: int = 20,
        later_pool_length: int = 3,
        later_pool_stride: int = 1,
        stride_before_pool: int = 3,
        dropout: float = 0.5029593396661691,
        activation: str = "gelu",
        normalization: str = "group",
        first_pool_mode: str = "mean",
        later_pool_mode: str = "mean",
        fourth_block: bool = True,
        fourth_block_broader: bool = True,
    ) -> None:
        super().__init__()
        kernels = tuple(int(value) for value in temporal_kernels)
        if len(kernels) != 5:
            raise ValueError("Multi-BK-Net requires exactly five first-block branches.")
        if total_first_block_filters % len(kernels) != 0:
            raise ValueError("Total first-block filters must be divisible by five.")
        if not fourth_block or not fourth_block_broader:
            raise ValueError(
                "This adaptation targets the published final four-block broader model."
            )

        self.n_channels = int(n_channels)
        self.input_samples = int(input_samples)
        self.n_classes = int(n_classes)
        branch_filters = total_first_block_filters // len(kernels)

        self.branches = nn.ModuleList(
            [
                TemporalSpatialBranch(
                    n_channels=self.n_channels,
                    filters=branch_filters,
                    temporal_kernel=kernel,
                    pool_length=first_pool_length,
                    pool_stride=first_pool_stride,
                    activation=activation,
                    normalization=normalization,
                    pool_mode=first_pool_mode,
                )
                for kernel in kernels
            ]
        )

        filters_2 = total_first_block_filters * 2
        filters_3 = filters_2 * 2
        filters_4 = filters_3 * 2
        self.blocks = nn.ModuleList(
            [
                LaterConvPoolBlock(
                    total_first_block_filters,
                    filters_2,
                    later_filter_length,
                    stride_before_pool,
                    later_pool_length,
                    later_pool_stride,
                    dropout,
                    activation,
                    normalization,
                    later_pool_mode,
                ),
                LaterConvPoolBlock(
                    filters_2,
                    filters_3,
                    later_filter_length,
                    stride_before_pool,
                    later_pool_length,
                    later_pool_stride,
                    dropout,
                    activation,
                    normalization,
                    later_pool_mode,
                ),
                LaterConvPoolBlock(
                    filters_3,
                    filters_4,
                    later_filter_length,
                    stride_before_pool,
                    later_pool_length,
                    later_pool_stride,
                    dropout,
                    activation,
                    normalization,
                    later_pool_mode,
                ),
            ]
        )

        with torch.no_grad():
            feature_shape = self.forward_features(
                torch.zeros(1, self.n_channels, self.input_samples)
            ).shape
        self.final_feature_samples = int(feature_shape[2])
        self.classifier = nn.Conv2d(
            filters_4,
            self.n_classes,
            kernel_size=(self.final_feature_samples, 1),
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch, channels, samples), got {tuple(x.shape)}"
            )
        if x.shape[1] != self.n_channels or x.shape[2] != self.input_samples:
            raise ValueError(
                f"Expected (*, {self.n_channels}, {self.input_samples}), got {tuple(x.shape)}"
            )
        # (B, C, T) -> (B, 1, T, C): temporal convolution then spatial convolution.
        x = x.unsqueeze(1).permute(0, 1, 3, 2)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits.squeeze(-1).squeeze(-1)


def build_model(config: dict, n_channels: int | None = None) -> MultiBKNet:
    model_cfg = config["model"]
    preprocessing = config["preprocessing"]
    if n_channels is None:
        n_channels = len(config["dataset"]["channels"])
    input_samples = round(
        float(preprocessing["target_sfreq"]) * float(preprocessing["window_seconds"])
    )
    return MultiBKNet(
        n_channels=n_channels,
        input_samples=input_samples,
        n_classes=int(model_cfg["n_classes"]),
        total_first_block_filters=int(model_cfg["total_first_block_filters"]),
        temporal_kernels=model_cfg["temporal_kernel_samples"],
        first_pool_length=int(model_cfg["first_pool_length"]),
        first_pool_stride=int(model_cfg["first_pool_stride"]),
        later_filter_length=int(model_cfg["later_filter_length"]),
        later_pool_length=int(model_cfg["later_pool_length"]),
        later_pool_stride=int(model_cfg["later_pool_stride"]),
        stride_before_pool=int(model_cfg["stride_before_pool"]),
        dropout=float(model_cfg["dropout"]),
        activation=str(model_cfg["activation"]),
        normalization=str(model_cfg["normalization"]),
        first_pool_mode="mean",
        later_pool_mode="mean",
        fourth_block=bool(model_cfg["fourth_block"]),
        fourth_block_broader=bool(model_cfg["fourth_block_broader"]),
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
