from __future__ import annotations

import torch
from torch import nn


class Deep4Net(nn.Module):
    """Independent PyTorch implementation of the Schirrmeister Deep ConvNet.

    Input shape: [batch, channels, time].
    Output: one binary logit per input window.
    """

    def __init__(
        self,
        n_chans: int = 19,
        n_times: int = 600,
        n_filters_time: int = 25,
        n_filters_spat: int = 25,
        filter_time_length: int = 10,
        pool_time_length: int = 3,
        pool_time_stride: int = 3,
        n_filters_2: int = 50,
        filter_length_2: int = 10,
        n_filters_3: int = 100,
        filter_length_3: int = 10,
        n_filters_4: int = 200,
        filter_length_4: int = 10,
        drop_prob: float = 0.5,
        batch_norm: bool = True,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times

        # Temporal filtering is applied independently to each electrode first,
        # followed by a spatial convolution spanning all EEG channels.
        self.conv_time = nn.Conv2d(
            1,
            n_filters_time,
            kernel_size=(filter_time_length, 1),
            stride=1,
            bias=True,
        )
        self.conv_spat = nn.Conv2d(
            n_filters_time,
            n_filters_spat,
            kernel_size=(1, n_chans),
            stride=1,
            bias=not batch_norm,
        )
        self.bn1 = (
            nn.BatchNorm2d(n_filters_spat, momentum=batch_norm_momentum, eps=1e-5)
            if batch_norm
            else nn.Identity()
        )
        self.act1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1))

        self.block2 = self._make_block(
            n_filters_spat, n_filters_2, filter_length_2, pool_time_length,
            pool_time_stride, drop_prob, batch_norm, batch_norm_momentum
        )
        self.block3 = self._make_block(
            n_filters_2, n_filters_3, filter_length_3, pool_time_length,
            pool_time_stride, drop_prob, batch_norm, batch_norm_momentum
        )
        self.block4 = self._make_block(
            n_filters_3, n_filters_4, filter_length_4, pool_time_length,
            pool_time_stride, drop_prob, batch_norm, batch_norm_momentum
        )

        final_time = self._infer_final_time(n_times)
        if final_time < 1:
            raise ValueError(
                f"n_times={n_times} is too short for Deep4Net. "
                "Use at least about 441 samples with the default kernels/pooling."
            )
        self.classifier = nn.Conv2d(n_filters_4, 1, kernel_size=(final_time, 1), bias=True)
        self._initialize_weights()

    @staticmethod
    def _make_block(
        in_filters: int,
        out_filters: int,
        filter_length: int,
        pool_length: int,
        pool_stride: int,
        drop_prob: float,
        batch_norm: bool,
        batch_norm_momentum: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Dropout(drop_prob),
            nn.Conv2d(
                in_filters,
                out_filters,
                kernel_size=(filter_length, 1),
                stride=1,
                bias=not batch_norm,
            ),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_filters, momentum=batch_norm_momentum, eps=1e-5))
        layers.extend(
            [
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(pool_length, 1), stride=(pool_stride, 1)),
            ]
        )
        return nn.Sequential(*layers)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, T] -> [B, 1, T, C]
        x = x.unsqueeze(1).permute(0, 1, 3, 2)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def _infer_final_time(self, n_times: int) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_chans, n_times)
            try:
                out = self._features(dummy)
            except RuntimeError as e:
                raise ValueError(
                    f"Input length {n_times} samples is not valid for Deep4Net default kernels."
                ) from e
        self.train(was_training)
        return int(out.shape[2])

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x)
        x = self.classifier(x)
        return x.flatten(1).squeeze(1)


if __name__ == "__main__":
    m = Deep4Net(n_chans=19, n_times=600)
    x = torch.randn(4, 19, 600)
    y = m(x)
    print(m)
    print("Output:", y.shape)
    print("Parameters:", sum(p.numel() for p in m.parameters()))
