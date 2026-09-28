from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """1-D causal convolution with output length equal to input length."""

    def __init__(self, *args, **kwargs):
        kernel_size = kwargs.get("kernel_size", args[2] if len(args) > 2 else None)
        dilation = kwargs.get("dilation", 1)
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(dilation, tuple):
            dilation = dilation[0]
        self._causal_pad = (int(kernel_size) - 1) * int(dilation)
        kwargs["padding"] = self._causal_pad
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._causal_pad > 0:
            y = y[..., :-self._causal_pad]
        return y


class WaveBlock(nn.Module):
    """
    Modified WaveNet block.

    The paper specifies gated dilated causal convolutions, residual accumulation,
    no separate skip-connection output, and dilation rates [1, 2, 4, ...].
    """

    def __init__(self, in_channels: int, filters: int, kernel_size: int, n_dilations: int):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, filters, kernel_size=1)
        dilations = [2**i for i in range(n_dilations)]
        self.filter_convs = nn.ModuleList(
            [CausalConv1d(filters, filters, kernel_size=kernel_size, dilation=d) for d in dilations]
        )
        self.gate_convs = nn.ModuleList(
            [CausalConv1d(filters, filters, kernel_size=kernel_size, dilation=d) for d in dilations]
        )
        self.residual_convs = nn.ModuleList(
            [nn.Conv1d(filters, filters, kernel_size=1) for _ in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        residual_sum = x
        for fconv, gconv, rconv in zip(self.filter_convs, self.gate_convs, self.residual_convs):
            gated = torch.tanh(fconv(x)) * torch.sigmoid(gconv(x))
            x = rconv(gated)
            residual_sum = residual_sum + x
        return residual_sum


class FeatureAttention(nn.Module):
    """Feature-wise attention over the TimeDistributed-LSTM representation."""

    def __init__(self, features: int):
        super().__init__()
        self.score = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=-1)
        return x * weights


class WaveNetLSTM(nn.Module):
    """
    Paper-faithful dual-path WaveNet-LSTM adaptation.

    Input shape: [B, T, 20]. At 250 Hz and 60 s, T=15000.

    Note: Figure 5 in the source paper labels some lower-path intermediate tensors
    as [N,30,20] even though it also specifies LSTM(64). To preserve the specified
    64-unit LSTMs and the unambiguous final [N,30,2] -> 60 flattening, this
    implementation uses [N,30,64] internally in that branch.
    """

    def __init__(
        self,
        input_channels: int = 20,
        input_samples: int = 15000,
        window_size_samples: int = 500,
        wave_blocks: Sequence[Sequence[int]] = ((16, 3, 8), (32, 3, 5), (64, 3, 3), (64, 2, 2)),
        pool_sizes: Sequence[int] = (10, 10, 10, 2),
        wavenet_lstm_hidden: int = 64,
        window_lstm_hidden: int = 64,
        sequence_lstm_hidden: int = 64,
        top_dropout: float = 0.5,
        bottom_dropout: float = 0.2,
    ):
        super().__init__()
        if len(wave_blocks) != len(pool_sizes):
            raise ValueError("wave_blocks and pool_sizes must have the same length")
        if input_samples % window_size_samples != 0:
            raise ValueError(
                f"input_samples={input_samples} must be divisible by window_size_samples={window_size_samples}"
            )
        self.input_channels = input_channels
        self.input_samples = input_samples
        self.window_size = window_size_samples
        self.num_windows = input_samples // window_size_samples

        top_modules = []
        in_ch = input_channels
        for (filters, kernel, n_dil), pool in zip(wave_blocks, pool_sizes):
            top_modules.append(WaveBlock(in_ch, int(filters), int(kernel), int(n_dil)))
            top_modules.append(nn.AvgPool1d(kernel_size=int(pool), stride=int(pool)))
            in_ch = int(filters)
        self.top_conv = nn.Sequential(*top_modules)
        self.top_lstm = nn.LSTM(input_size=in_ch, hidden_size=wavenet_lstm_hidden, batch_first=True)
        self.top_dropout = nn.Dropout(top_dropout)

        # Bottom path: reverse -> 30 windows of 500 samples -> shared LSTM per window.
        self.window_lstm = nn.LSTM(
            input_size=input_channels, hidden_size=window_lstm_hidden, batch_first=True
        )
        self.attention = FeatureAttention(window_lstm_hidden)
        self.sequence_lstm = nn.LSTM(
            input_size=window_lstm_hidden, hidden_size=sequence_lstm_hidden, batch_first=True
        )
        self.bottom_dropout = nn.Dropout(bottom_dropout)
        self.bottom_dense = nn.Linear(sequence_lstm_hidden, 2)

        concat_features = wavenet_lstm_hidden + self.num_windows * 2
        self.classifier = nn.Linear(concat_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,C], got {tuple(x.shape)}")
        if x.shape[1] != self.input_samples or x.shape[2] != self.input_channels:
            raise ValueError(
                f"Expected [B,{self.input_samples},{self.input_channels}], got {tuple(x.shape)}"
            )

        # Top WaveNet-LSTM path.
        top = x.transpose(1, 2)  # [B,C,T]
        top = self.top_conv(top)
        top = top.transpose(1, 2)  # [B,T',C']
        _, (h, _) = self.top_lstm(top)
        top = self.top_dropout(h[-1])  # [B,64]

        # Bottom attention-LSTM path.
        bottom = torch.flip(x, dims=[1])
        b, _, c = bottom.shape
        bottom = bottom.reshape(b, self.num_windows, self.window_size, c)
        bottom = bottom.reshape(b * self.num_windows, self.window_size, c)
        _, (h_win, _) = self.window_lstm(bottom)
        bottom = h_win[-1].reshape(b, self.num_windows, -1)
        bottom = self.attention(bottom)
        bottom, _ = self.sequence_lstm(bottom)
        bottom = self.bottom_dropout(bottom)
        bottom = torch.tanh(self.bottom_dense(bottom))  # [B,num_windows,2]
        bottom = bottom.flatten(start_dim=1)  # [B,num_windows*2]

        fused = torch.cat([top, bottom], dim=1)
        return self.classifier(fused)  # raw logits; CrossEntropyLoss applies softmax internally


def build_model(cfg: dict) -> WaveNetLSTM:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    input_samples = int(round(data_cfg["target_sfreq_hz"] * data_cfg["segment_seconds"]))
    return WaveNetLSTM(
        input_channels=int(model_cfg["input_channels"]),
        input_samples=input_samples,
        window_size_samples=int(model_cfg["window_size_samples"]),
        wave_blocks=model_cfg["wave_blocks"],
        pool_sizes=model_cfg["pool_sizes"],
        wavenet_lstm_hidden=int(model_cfg["wavenet_lstm_hidden"]),
        window_lstm_hidden=int(model_cfg["window_lstm_hidden"]),
        sequence_lstm_hidden=int(model_cfg["sequence_lstm_hidden"]),
        top_dropout=float(model_cfg["top_dropout"]),
        bottom_dropout=float(model_cfg["bottom_dropout"]),
    )
