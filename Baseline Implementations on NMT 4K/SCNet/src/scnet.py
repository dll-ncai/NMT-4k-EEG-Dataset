from __future__ import annotations

import torch
from torch import nn


class SILM(nn.Module):
    """Spatial Information Learning Module from Wu et al. (2023).

    For X in R^(C x T), computes channel-wise GAP, GMP and population
    standard-deviation pooling at every time point, applies independent dropout,
    concatenates the three 1 x T descriptors, then concatenates them with X.
    """

    def __init__(self, dropout: float = 0.05) -> None:
        super().__init__()
        self.drop_gap = nn.Dropout(dropout)
        self.drop_gmp = nn.Dropout(dropout)
        self.drop_gsp = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        gap = x.mean(dim=1, keepdim=True)
        gmp = x.amax(dim=1, keepdim=True)
        # Paper equation uses denominator C, i.e. population std (unbiased=False).
        gsp = x.std(dim=1, keepdim=True, unbiased=False)
        spatial = torch.cat(
            [self.drop_gap(gap), self.drop_gmp(gmp), self.drop_gsp(gsp)], dim=1
        )
        return torch.cat([x, spatial], dim=1)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, relu: bool = True) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size, stride=1,
            padding=padding, bias=True
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class MFFM(nn.Module):
    """Multi-level Feature Fusion Module.

    F'   = ReLU(BN(Conv1D(F)))            with 8 filters, k=5
    F''  = ReLU(BN(Conv1D(F' || F)))      with 16 filters, k=5
    out  = F'' || F' || F
    """

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, 8, kernel_size=5, relu=True)
        self.conv2 = ConvBNReLU(in_ch + 8, 16, kernel_size=5, relu=True)
        self.out_channels = in_ch + 8 + 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.conv1(x)
        f2 = self.conv2(torch.cat([f1, x], dim=1))
        return torch.cat([f2, f1, x], dim=1)


class SCNet(nn.Module):
    """Paper-faithful PyTorch reconstruction of SCNet.

    The architecture follows Fig. 2 and Secs. 3.1-3.3 of Wu et al. (2023).
    Conv padding is set to ``same`` (k//2), which is required by the MFFM skip
    concatenations but was not explicitly stated in the paper.
    """

    def __init__(self, input_channels: int, num_classes: int = 2) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.silm = SILM(dropout=0.05)

        aug_ch = input_channels + 3
        pooled_ch = 2 * aug_ch
        self.avg_pool3 = nn.AvgPool1d(kernel_size=3, stride=3)
        self.max_pool3 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool_bn = nn.BatchNorm1d(pooled_ch)

        self.mffm1a = MFFM(pooled_ch)
        self.mffm1b = MFFM(pooled_ch)
        stage1_ch = self.mffm1a.out_channels

        self.spatial_dropout = nn.Dropout1d(p=0.5)
        self.max_pool2a = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = ConvBNReLU(stage1_ch, 32, kernel_size=3, relu=True)

        self.mffm2a = MFFM(32)
        self.mffm2b = MFFM(32)
        stage2_ch = self.mffm2a.out_channels

        # The figure shows BN but no ReLU after this convolution.
        self.conv2 = ConvBNReLU(stage2_ch, 32, kernel_size=3, relu=False)

        self.mffm3 = MFFM(32)
        stage3_ch = self.mffm3.out_channels
        self.max_pool2b = nn.MaxPool1d(kernel_size=2, stride=2)
        # Again, Fig. 2 shows BN but no ReLU here.
        self.conv3 = ConvBNReLU(stage3_ch, 32, kernel_size=3, relu=False)

        self.classifier = nn.Linear(32, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, T]
        x = self.silm(x)

        x = torch.cat([self.avg_pool3(x), self.max_pool3(x)], dim=1)
        x = self.pool_bn(x)

        x = self.mffm1a(x) + self.mffm1b(x)
        x = self.spatial_dropout(x)
        x = self.max_pool2a(x)
        x = self.conv1(x)

        x = self.mffm2a(x) + self.mffm2b(x)
        x = self.conv2(x)

        x = self.mffm3(x)
        x = self.max_pool2b(x)
        x = self.conv3(x)

        # Global average pooling over time.
        x = x.mean(dim=-1)
        return self.classifier(x)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
