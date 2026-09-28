"""EEGPT encoder plus a recording-level multiple-instance learning head."""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn as nn

from .constants import EEGPT_SCALP_CHANNELS, EEGPT_WINDOW_SAMPLES, NMT_INPUT_CHANNELS
from .eegpt_backbone import Conv1dWithConstraint, EEGTransformer


class EEGPTWindowEncoder(nn.Module):
    """Adapt 21 NMT channels and return one 512-D feature per four-second window."""

    def __init__(
        self,
        adapter: str = "spatial_temporal",
        temporal_kernel: int = 15,
        adapter_dropout: float = 0.25,
        feature_dropout: float = 0.25,
        max_norm_adapter: float = 1.0,
    ) -> None:
        super().__init__()
        if adapter not in {"spatial_only", "spatial_temporal"}:
            raise ValueError("adapter must be 'spatial_only' or 'spatial_temporal'.")
        if temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be odd so the time length remains exactly 1,024.")

        # This learnable 1x1 convolution maps the 21 NMT electrodes to the 19
        # named scalp positions used to look up EEGPT channel embeddings.
        self.chan_conv = Conv1dWithConstraint(
            len(NMT_INPUT_CHANNELS),
            len(EEGPT_SCALP_CHANNELS),
            kernel_size=1,
            max_norm=max_norm_adapter,
            doWeightNorm=max_norm_adapter > 0,
        )
        # Start as an exact pass-through for the 19 scalp channels and ignore
        # A1/A2 until supervised training finds evidence that they help.
        with torch.no_grad():
            self.chan_conv.weight.zero_()
            self.chan_conv.bias.zero_()
            for channel_index in range(len(EEGPT_SCALP_CHANNELS)):
                self.chan_conv.weight[channel_index, channel_index, 0] = 1.0
        self.adapter = adapter
        if adapter == "spatial_temporal":
            self.temporal_adapter = nn.Sequential(
                nn.Conv1d(
                    len(EEGPT_SCALP_CHANNELS),
                    len(EEGPT_SCALP_CHANNELS),
                    kernel_size=temporal_kernel,
                    padding=temporal_kernel // 2,
                    groups=len(EEGPT_SCALP_CHANNELS),
                    bias=False,
                ),
                nn.BatchNorm1d(len(EEGPT_SCALP_CHANNELS)),
                nn.GELU(),
                nn.Dropout(adapter_dropout),
            )
            # Zero starts the residual branch as an identity-safe spatial-only
            # baseline and lets training introduce temporal adaptation gradually.
            self.temporal_scale = nn.Parameter(torch.tensor(0.0))

        # Keep the module name target_encoder because that is the key prefix in
        # the released EEGPT pretraining checkpoint.
        self.target_encoder = EEGTransformer(
            img_size=(len(EEGPT_SCALP_CHANNELS), EEGPT_WINDOW_SAMPLES),
            patch_size=64,
            patch_stride=64,
            embed_dim=512,
            embed_num=4,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        channel_ids = self.target_encoder.prepare_chan_ids(EEGPT_SCALP_CHANNELS)
        self.register_buffer("chans_id", channel_ids, persistent=False)
        self.feature_norm = nn.LayerNorm(512)
        self.feature_dropout = nn.Dropout(feature_dropout)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Encode ``[number_of_windows, 21, 1024]`` into ``[..., 512]``."""
        if windows.ndim != 3:
            raise ValueError(f"Window encoder expects a 3-D tensor, received {tuple(windows.shape)}")
        if tuple(windows.shape[1:]) != (len(NMT_INPUT_CHANNELS), EEGPT_WINDOW_SAMPLES):
            raise ValueError(
                f"Expected [batch, 21, 1024], received {tuple(windows.shape)}. "
                "Temporal interpolation is intentionally forbidden."
            )
        adapted = self.chan_conv(windows)
        if self.adapter == "spatial_temporal":
            adapted = adapted + self.temporal_scale * self.temporal_adapter(adapted)
        tokens = self.target_encoder(adapted, self.chans_id.to(adapted))
        # EEGTransformer returns [windows, 16 temporal patches, 4 summary tokens, 512].
        features = tokens.flatten(1, 2).mean(dim=1)
        return self.feature_dropout(self.feature_norm(features))


class EEGPTMILClassifier(nn.Module):
    """Combine window evidence into one recording logit and one recording loss."""

    def __init__(
        self,
        aggregation: str = "attention",
        topk_fraction: float = 0.25,
        **encoder_options: Any,
    ) -> None:
        super().__init__()
        if aggregation not in {"mean_logit", "topk_mean", "attention"}:
            raise ValueError("aggregation must be mean_logit, topk_mean, or attention.")
        if not 0 < topk_fraction <= 1:
            raise ValueError("topk_fraction must be in (0, 1].")
        self.aggregation = aggregation
        self.topk_fraction = float(topk_fraction)
        self.window_encoder = EEGPTWindowEncoder(**encoder_options)
        # Aliases expose intuitive names without registering the encoder twice.
        self.window_head = nn.Linear(512, 1)
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(512, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )

    @property
    def target_encoder(self) -> EEGTransformer:
        """Convenient access used by freezing and checkpoint audit code."""
        return self.window_encoder.target_encoder

    def _encode_in_chunks(self, flat_windows: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Bound GPU memory when validation/test recordings contain many windows."""
        if chunk_size <= 0 or len(flat_windows) <= chunk_size:
            return self.window_encoder(flat_windows)
        features = [
            self.window_encoder(flat_windows[start : start + chunk_size])
            for start in range(0, len(flat_windows), chunk_size)
        ]
        return torch.cat(features, dim=0)

    def _topk_pool(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled: list[torch.Tensor] = []
        for recording_logits, recording_mask in zip(logits, mask):
            valid = recording_logits[recording_mask]
            number_to_keep = max(1, int(round(len(valid) * self.topk_fraction)))
            pooled.append(torch.topk(valid, k=number_to_keep).values.mean())
        return torch.stack(pooled)

    def forward(
        self,
        windows: torch.Tensor,
        mask: torch.Tensor,
        encode_chunk_size: int = 16,
    ) -> dict[str, torch.Tensor]:
        """Return recording logits, window logits, and optional attention weights."""
        if windows.ndim != 4:
            raise ValueError(f"MIL model expects [recordings, windows, channels, time], got {windows.shape}")
        batch_size, bag_size, channels, samples = windows.shape
        mask = mask.bool()
        if tuple(mask.shape) != (batch_size, bag_size):
            raise ValueError(f"Mask shape {tuple(mask.shape)} does not match bag shape {(batch_size, bag_size)}")
        if not mask.any(dim=1).all():
            raise ValueError("Every recording bag must contain at least one valid window.")
        flat_windows = windows.reshape(batch_size * bag_size, channels, samples)
        flat_mask = mask.reshape(-1)
        valid_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
        # Padding is never encoded, so it cannot affect adapter BatchNorm or
        # waste expensive EEGPT computation on all-zero pseudo-windows.
        valid_features = self._encode_in_chunks(flat_windows[valid_indices], int(encode_chunk_size))
        flat_features = valid_features.new_zeros((batch_size * bag_size, valid_features.shape[-1]))
        flat_features = flat_features.index_copy(0, valid_indices, valid_features)
        features = flat_features.reshape(batch_size, bag_size, -1)
        window_logits = self.window_head(features).squeeze(-1)

        attention_weights: torch.Tensor | None = None
        if self.aggregation == "mean_logit":
            recording_logits = (window_logits * mask).sum(dim=1) / mask.sum(dim=1)
        elif self.aggregation == "topk_mean":
            recording_logits = self._topk_pool(window_logits, mask)
        else:
            attention_scores = self.attention(features).squeeze(-1)
            attention_scores = attention_scores.masked_fill(~mask, torch.finfo(attention_scores.dtype).min)
            attention_weights = torch.softmax(attention_scores, dim=1)
            pooled_features = (features * attention_weights.unsqueeze(-1)).sum(dim=1)
            recording_logits = self.window_head(pooled_features).squeeze(-1)

        return {
            "recording_logits": recording_logits,
            "window_logits": window_logits,
            "attention_weights": attention_weights,
        }


def set_encoder_trainability(model: EEGPTMILClassifier, unfreeze_last_n: int) -> None:
    """Freeze all EEGPT weights, then optionally open only its highest blocks."""
    encoder = model.target_encoder
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if unfreeze_last_n < 0:
        for parameter in encoder.parameters():
            parameter.requires_grad_(True)
        return
    if unfreeze_last_n == 0:
        return
    number_of_blocks = len(encoder.blocks)
    if unfreeze_last_n > number_of_blocks:
        raise ValueError(f"Cannot unfreeze {unfreeze_last_n}; EEGPT has {number_of_blocks} encoder blocks.")
    for block in encoder.blocks[-unfreeze_last_n:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    for parameter in encoder.norm.parameters():
        parameter.requires_grad_(True)


def parameter_summary(model: nn.Module) -> dict[str, int]:
    """Count all and trainable scalar parameters for transparent run logs."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}
