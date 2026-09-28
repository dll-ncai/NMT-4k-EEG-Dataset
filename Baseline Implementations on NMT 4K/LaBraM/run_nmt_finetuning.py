#!/usr/bin/env python
"""Fine-tune LaBraM-base for recording-level NMT abnormality classification."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from nmt_finetuning.channels import NMT_CHANNELS, get_input_chans
from nmt_finetuning.data import (
    NMTWindowDataset,
    make_recording_balanced_sampler,
    read_manifest,
    select_split,
    summarize_entries,
)
from nmt_finetuning.metrics import (
    aggregate_recordings,
    binary_metrics,
    find_balanced_accuracy_threshold,
)

PROJECT_ROOT = Path(__file__).resolve().parent
LABRAM_ROOT = PROJECT_ROOT / "LaBraM"
if str(LABRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(LABRAM_ROOT))

import modeling_finetune  # noqa: F401 - registers LaBraM models with timm
from timm.models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/nmt_finetune_manifest.csv")
    parser.add_argument("--checkpoint", type=Path, default=LABRAM_ROOT / "checkpoints/labram-base.pth")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/nmt_labram_base")
    parser.add_argument("--resume", type=Path, default=None, help="Resume a fine-tuning checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--allow-unsafe-checkpoint", action="store_true")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--update-freq", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epoch-size", type=int, default=50000, help="Balanced windows drawn per epoch; 0 uses all")
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-val-windows-per-recording", type=int, default=64)
    parser.add_argument("--max-test-windows-per-recording", type=int, default=0)
    parser.add_argument(
        "--sampling-strategy",
        choices=("recording-balanced", "shuffle"),
        default="recording-balanced",
    )

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--layer-decay", type=float, default=0.65)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=3.0)
    parser.add_argument("--input-scale", type=float, default=100.0, help="Official LaBraM code divides microvolts by 100")
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "bfloat16", "float16", "none"),
        default="auto",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (optional; leave off for first run)")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=("balanced_accuracy", "roc_auc", "f1"),
        default="balanced_accuracy",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
    if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_amp_dtype(choice: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or choice == "none":
        return None
    if choice == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not report bfloat16 support")
        return torch.bfloat16
    if choice == "float16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def safe_torch_load(path: Path, allow_unsafe: bool = False) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # The official LaBraM checkpoint stores an argparse Namespace and NumPy
    # scalars alongside its tensor state. Explicitly allow only those known
    # harmless types while retaining weights_only=True.
    safe_types = [argparse.Namespace, np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float32))]
    try:
        with torch.serialization.safe_globals(safe_types):
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_unsafe:
            raise RuntimeError(
                f"Safe checkpoint loading failed for {path}. If you trust this file, rerun with "
                "--allow-unsafe-checkpoint. Original error: " + str(exc)
            ) from exc
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_pretrained_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "module"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary")

    keys = list(state)
    if any(key.startswith("student.") for key in keys):
        state = {key.removeprefix("student."): value for key, value in state.items() if key.startswith("student.")}
    else:
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_pretrained_weights(model: nn.Module, path: Path, allow_unsafe: bool) -> None:
    checkpoint = safe_torch_load(path, allow_unsafe)
    state = extract_pretrained_state(checkpoint)
    model_state = model.state_dict()
    usable: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    for key, value in state.items():
        if "relative_position_index" in key or key.startswith(("head.", "lm_head.")):
            continue
        if key in model_state and hasattr(value, "shape") and value.shape == model_state[key].shape:
            usable[key] = value
        elif key in model_state:
            skipped_shape.append(key)
    missing, unexpected = model.load_state_dict(usable, strict=False)
    non_head_keys = [key for key in model_state if not key.startswith("head.")]
    coverage = len(set(usable).intersection(non_head_keys)) / max(1, len(non_head_keys))
    print(
        f"Loaded {len(usable)} pretrained tensors from {path} "
        f"({coverage:.1%} of fine-tuning backbone state entries)."
    )
    if skipped_shape:
        print(f"Skipped {len(skipped_shape)} tensors with incompatible shapes: {skipped_shape[:8]}")
    if unexpected:
        print(f"Ignored unexpected tensors: {unexpected[:8]}")
    meaningful_missing = [key for key in missing if not key.startswith("head.")]
    if coverage < 0.70:
        raise RuntimeError(
            "Too little of the LaBraM backbone was loaded. "
            f"Missing backbone examples: {meaningful_missing[:12]}"
        )


def create_labram_model(drop_path: float) -> nn.Module:
    return create_model(
        "labram_base_patch200_200",
        pretrained=False,
        num_classes=1,
        drop_rate=0.0,
        drop_path_rate=drop_path,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
    )


def layer_id_for_parameter(name: str, depth: int) -> int:
    if name in {"cls_token", "pos_embed", "time_embed"} or name.startswith("patch_embed"):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return depth + 1


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    depth = model.get_num_layers()
    maximum_layer = depth + 1
    groups: dict[tuple[int, bool], dict[str, Any]] = {}
    no_decay_names = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        layer_id = layer_id_for_parameter(name, depth)
        no_decay = parameter.ndim <= 1 or name.endswith(".bias") or name in no_decay_names
        key = (layer_id, no_decay)
        if key not in groups:
            lr_scale = args.layer_decay ** (maximum_layer - layer_id)
            groups[key] = {
                "params": [],
                "weight_decay": 0.0 if no_decay else args.weight_decay,
                "lr_scale": lr_scale,
                "lr": args.lr * lr_scale,
            }
        groups[key]["params"].append(parameter)
    return torch.optim.AdamW(list(groups.values()), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)


def scheduled_learning_rate(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    warmup_lr: float,
    min_lr: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        fraction = step / max(1, warmup_steps)
        return warmup_lr + fraction * (base_lr - warmup_lr)
    fraction = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    fraction = min(max(fraction, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * fraction))


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * float(group.get("lr_scale", 1.0))


def make_loader(
    dataset: NMTWindowDataset,
    sampler: Any,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "sampler": sampler,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def reshape_input(samples: torch.Tensor, input_scale: float) -> torch.Tensor:
    if samples.ndim != 3 or tuple(samples.shape[1:]) != (21, 2000):
        raise ValueError(f"Expected batch [B, 21, 2000], got {tuple(samples.shape)}")
    return (samples / input_scale).reshape(samples.shape[0], 21, 10, 200)


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    return torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_dtype is not None,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    input_chans: list[int],
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    start = time.time()

    progress = tqdm(loader, desc=f"Train {epoch + 1}/{args.epochs}", dynamic_ncols=True)
    for batch_index, (samples, labels, _) in enumerate(progress):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        samples = reshape_input(samples, args.input_scale)

        with autocast_context(device, amp_dtype):
            logits = model(samples, input_chans=input_chans)
            loss = criterion(logits, labels)
            scaled_loss = loss / args.update_freq
        scaler.scale(scaled_loss).backward()

        should_step = (batch_index + 1) % args.update_freq == 0 or batch_index + 1 == len(loader)
        if should_step:
            learning_rate = scheduled_learning_rate(
                global_step,
                total_steps,
                warmup_steps,
                args.lr,
                args.warmup_lr,
                args.min_lr,
            )
            set_learning_rate(optimizer, learning_rate)
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        batch_size = samples.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=f"{total_loss / total_examples:.4f}")

    return {
        "loss": total_loss / max(1, total_examples),
        "seconds": time.time() - start,
        "learning_rate": max(group["lr"] for group in optimizer.param_groups),
    }, global_step


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    input_chans: list[int],
    input_scale: float,
    threshold: float | None,
    description: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    recording_ids: list[str] = []
    total_loss = 0.0
    total_examples = 0

    for samples, labels, batch_recording_ids in tqdm(loader, desc=description, dynamic_ncols=True):
        samples = samples.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True).unsqueeze(1)
        samples = reshape_input(samples, input_scale)
        with autocast_context(device, amp_dtype):
            logits = model(samples, input_chans=input_chans)
            loss = criterion(logits, labels_device)
        batch_probabilities = torch.sigmoid(logits).float().cpu().numpy().reshape(-1)
        batch_labels = labels.numpy().astype(np.int64).reshape(-1)
        probabilities.append(batch_probabilities)
        labels_all.append(batch_labels)
        recording_ids.extend(list(batch_recording_ids))
        total_loss += float(loss) * len(batch_labels)
        total_examples += len(batch_labels)

    probability_array = np.concatenate(probabilities)
    label_array = np.concatenate(labels_all)
    record_labels, record_probabilities, ordered_recording_ids = aggregate_recordings(
        label_array, probability_array, recording_ids
    )
    selected_threshold = (
        find_balanced_accuracy_threshold(record_labels, record_probabilities)
        if threshold is None
        else threshold
    )
    metrics = {
        "loss": total_loss / max(1, total_examples),
        "window": binary_metrics(label_array, probability_array, selected_threshold),
        "recording": binary_metrics(record_labels, record_probabilities, selected_threshold),
    }
    predictions = {
        "recording_ids": ordered_recording_ids,
        "labels": record_labels,
        "probabilities": record_probabilities,
        "threshold": selected_threshold,
    }
    return metrics, predictions


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    threshold: float,
    validation_metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "threshold": threshold,
            "validation_metrics": validation_metrics,
            "args": serializable_args(args),
        },
        path,
    )


def restore_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    allow_unsafe: bool,
) -> tuple[int, int, float]:
    checkpoint = safe_torch_load(path, allow_unsafe)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Not an NMT fine-tuning checkpoint: {path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("threshold", 0.5)),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def write_recording_predictions(path: Path, predictions: dict[str, Any]) -> None:
    threshold = float(predictions["threshold"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["recording_id", "label", "probability_abnormal", "prediction", "threshold"])
        for recording_id, label, probability in zip(
            predictions["recording_ids"],
            predictions["labels"],
            predictions["probabilities"],
            strict=True,
        ):
            writer.writerow(
                [recording_id, int(label), float(probability), int(probability >= threshold), threshold]
            )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.update_freq <= 0 or args.epochs <= 0:
        raise ValueError("Batch size, update frequency, and epochs must be positive")
    if args.input_scale <= 0:
        raise ValueError("--input-scale must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"AMP dtype: {amp_dtype}")
    print(f"Channel order ({len(NMT_CHANNELS)}): {NMT_CHANNELS}")

    all_entries = read_manifest(args.manifest)
    print("Dataset summary:")
    print(json.dumps(summarize_entries(all_entries), indent=2))
    train_entries = select_split(all_entries, "train")
    val_entries = select_split(all_entries, "val", args.max_val_windows_per_recording)
    test_entries = select_split(all_entries, "test", args.max_test_windows_per_recording)

    train_dataset = NMTWindowDataset(train_entries)
    val_dataset = NMTWindowDataset(val_entries)
    test_dataset = NMTWindowDataset(test_entries)
    if args.sampling_strategy == "recording-balanced":
        train_sampler = make_recording_balanced_sampler(train_entries, args.epoch_size, args.seed)
    else:
        train_sampler = RandomSampler(train_dataset)
    pin_memory = device.type == "cuda" and not args.no_pin_memory
    train_loader = make_loader(train_dataset, train_sampler, args.batch_size, args.num_workers, pin_memory)
    val_loader = make_loader(
        val_dataset, SequentialSampler(val_dataset), args.batch_size, args.num_workers, pin_memory
    )
    test_loader = make_loader(
        test_dataset, SequentialSampler(test_dataset), args.batch_size, args.num_workers, pin_memory
    )

    model = create_labram_model(args.drop_path)
    load_pretrained_weights(model, args.checkpoint, args.allow_unsafe_checkpoint)
    model.to(device)
    if args.compile:
        model = torch.compile(model)
    optimizer = build_optimizer(model, args)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16
    )
    input_chans = get_input_chans()

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.update_freq)
    total_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = optimizer_steps_per_epoch * args.warmup_epochs
    start_epoch = 0
    global_step = 0
    best_threshold = 0.5
    if args.resume is not None:
        start_epoch, global_step, best_threshold = restore_training_checkpoint(
            args.resume,
            model,
            None if args.eval_only else optimizer,
            None if args.eval_only else scaler,
            args.allow_unsafe_checkpoint,
        )
        print(f"Resumed from {args.resume} (next epoch {start_epoch + 1})")
    elif args.eval_only:
        raise ValueError("--eval-only requires --resume pointing to a fine-tuned checkpoint")

    if args.eval_only:
        test_metrics, test_predictions = evaluate(
            model,
            test_loader,
            criterion,
            device,
            amp_dtype,
            input_chans,
            args.input_scale,
            best_threshold,
            "Test",
        )
        write_json(args.output_dir / "test_metrics.json", test_metrics)
        write_recording_predictions(args.output_dir / "test_recording_predictions.csv", test_predictions)
        print(json.dumps(test_metrics, indent=2))
        return

    best_score = -float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs):
        train_metrics, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            amp_dtype,
            input_chans,
            args,
            epoch,
            global_step,
            total_steps,
            warmup_steps,
        )
        val_metrics, val_predictions = evaluate(
            model,
            val_loader,
            criterion,
            device,
            amp_dtype,
            input_chans,
            args.input_scale,
            None,
            "Validation",
        )
        threshold = float(val_predictions["threshold"])
        score = float(val_metrics["recording"][args.selection_metric])
        epoch_result = {"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics}
        history.append(epoch_result)
        write_json(args.output_dir / "history.json", {"epochs": history})
        print(json.dumps(epoch_result, indent=2))

        save_training_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scaler,
            epoch,
            global_step,
            threshold,
            val_metrics,
            args,
        )
        if score > best_score:
            best_score = score
            best_threshold = threshold
            epochs_without_improvement = 0
            save_training_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                global_step,
                threshold,
                val_metrics,
                args,
            )
            print(f"New best recording-level {args.selection_metric}: {best_score:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    _, _, best_threshold = restore_training_checkpoint(
        args.output_dir / "best.pt", model, None, None, args.allow_unsafe_checkpoint
    )
    test_metrics, test_predictions = evaluate(
        model,
        test_loader,
        criterion,
        device,
        amp_dtype,
        input_chans,
        args.input_scale,
        best_threshold,
        "Final test",
    )
    write_json(args.output_dir / "test_metrics.json", test_metrics)
    write_recording_predictions(args.output_dir / "test_recording_predictions.csv", test_predictions)
    print("Final test metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "CUDA ran out of memory. Reduce --batch-size (for example, 16 to 8) and "
                "increase --update-freq to keep the same effective batch size."
            ) from exc
        raise
