"""Fine-tune EEGPT with recording-level MIL and validation-only model selection."""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import atomic_torch_save, load_pretrained_eegpt
from .cohort import print_training_cohort_status, validate_manifest_for_training
from .config import load_config, require_sections, resolve_path, save_config_snapshot
from .data import RecordingBagDataset, recording_class_counts
from .metrics import select_threshold
from .model import EEGPTMILClassifier, parameter_summary, set_encoder_trainability


def set_reproducibility(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy, CPU PyTorch, and every visible CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def model_options(config: dict[str, Any]) -> dict[str, Any]:
    """Select only YAML keys accepted by EEGPTMILClassifier."""
    model_config = config["model"]
    return {
        "aggregation": model_config.get("aggregation", "attention"),
        "topk_fraction": float(model_config.get("topk_fraction", 0.25)),
        "adapter": model_config.get("adapter", "spatial_temporal"),
        "temporal_kernel": int(model_config.get("temporal_kernel", 15)),
        "adapter_dropout": float(model_config.get("adapter_dropout", 0.25)),
        "feature_dropout": float(model_config.get("feature_dropout", 0.25)),
        "max_norm_adapter": float(model_config.get("max_norm_adapter", 1.0)),
    }


def build_optimizer(model: EEGPTMILClassifier, options: dict[str, Any]) -> AdamW:
    """Create discriminative learning-rate groups with layer-wise decay."""
    head_lr = float(options.get("head_learning_rate", 5e-4))
    encoder_lr = float(options.get("encoder_learning_rate", 2e-5))
    weight_decay = float(options.get("weight_decay", 0.05))
    layer_decay = float(options.get("layer_decay", 0.75))
    number_of_blocks = len(model.target_encoder.blocks)
    grouped: dict[tuple[float, float], list[nn.Parameter]] = {}

    for name, parameter in model.named_parameters():
        is_encoder = "window_encoder.target_encoder." in name
        if is_encoder:
            block_marker = ".blocks."
            if block_marker in name:
                block_index = int(name.split(block_marker, 1)[1].split(".", 1)[0])
                depth_from_output = number_of_blocks - 1 - block_index
            elif ".norm." in name:
                depth_from_output = 0
            else:
                depth_from_output = number_of_blocks
            learning_rate = encoder_lr * (layer_decay ** depth_from_output)
        else:
            learning_rate = head_lr
        no_decay = parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower()
        group_decay = 0.0 if no_decay else weight_decay
        grouped.setdefault((learning_rate, group_decay), []).append(parameter)

    parameter_groups = [
        {"params": parameters, "lr": learning_rate, "weight_decay": group_decay}
        for (learning_rate, group_decay), parameters in grouped.items()
    ]
    return AdamW(parameter_groups, betas=(0.9, 0.999))


def build_scheduler(
    optimizer: AdamW,
    total_updates: int,
    warmup_updates: int,
    minimum_lr_ratio: float,
) -> LambdaLR:
    """Warm up linearly and then decay smoothly with a cosine curve."""
    def multiplier(update: int) -> float:
        if update < warmup_updates:
            return max(1e-8, (update + 1) / max(1, warmup_updates))
        progress = (update - warmup_updates) / max(1, total_updates - warmup_updates)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=multiplier)


@torch.no_grad()
def predict_recordings(
    model: EEGPTMILClassifier,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    encode_chunk_size: int,
) -> pd.DataFrame:
    """Produce exactly one probability and label per recording."""
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="Validation", leave=False):
        windows = batch["windows"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            output = model(windows, mask, encode_chunk_size=encode_chunk_size)
        probabilities = torch.sigmoid(output["recording_logits"]).float().cpu().numpy()
        labels = batch["label"].numpy()
        for recording_id, label, probability in zip(batch["recording_id"], labels, probabilities):
            rows.append(
                {"recording_id": recording_id, "label": int(label), "probability_abnormal": float(probability)}
            )
    return pd.DataFrame(rows)


def _checkpoint_payload(
    model: EEGPTMILClassifier,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_metric: float,
    validation_threshold: float,
    validation_metrics: dict[str, float],
    model_config: dict[str, Any],
    pretrained_report: dict[str, Any],
    config: dict[str, Any],
    patience_counter: int,
) -> dict[str, Any]:
    """Save everything required to reproduce, resume, and evaluate this run."""
    clean_config = {key: value for key, value in config.items() if not key.startswith("_")}
    return {
        "format_version": 2,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_selection_metric": float(best_metric),
        "validation_threshold": float(validation_threshold),
        "validation_metrics": validation_metrics,
        "patience_counter": int(patience_counter),
        "model_options": model_config,
        "pretrained_load_report": pretrained_report,
        "config": clean_config,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def train(config: dict[str, Any]) -> Path:
    paths = config["paths"]
    options = config["training"]
    seed = int(options.get("seed", 2026))
    set_reproducibility(seed, bool(options.get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(options.get("amp", True)) and device.type == "cuda"
    print(f"Device: {device}; mixed precision: {amp_enabled}")

    manifest_path = resolve_path(config, paths["processed_manifest_csv"])
    pretrained_path = resolve_path(config, paths["pretrained_checkpoint"])
    runs_directory = resolve_path(config, paths["runs_dir"])
    assert manifest_path is not None and pretrained_path is not None and runs_directory is not None
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run preprocessing first; missing manifest: {manifest_path}")
    manifest_audit = pd.read_csv(manifest_path)
    cohort_summary = validate_manifest_for_training(manifest_audit, config)
    print_training_cohort_status(cohort_summary)

    run_name = str(options.get("run_name", f"eegpt_nmt_v2_seed{seed}"))
    run_directory = runs_directory / run_name
    resume_path = resolve_path(config, options.get("resume_checkpoint"))
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Configured resume checkpoint does not exist: {resume_path}")
    if resume_path is not None and resume_path.parent.resolve() != run_directory.resolve():
        raise ValueError(
            "resume_checkpoint must be last.pt inside the directory selected by training.run_name."
        )
    if (
        run_directory.exists()
        and any(run_directory.iterdir())
        and resume_path is None
        and not bool(options.get("allow_existing_run", False))
    ):
        raise FileExistsError(
            f"Run directory already contains files: {run_directory}. Change training.run_name or set allow_existing_run=true."
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    snapshot_name = "config_used.yaml" if resume_path is None else "config_resume.yaml"
    save_config_snapshot(config, run_directory / snapshot_name)

    train_dataset = RecordingBagDataset(
        manifest_path,
        config["_project_root"],
        split="internal_train",
        windows_per_recording=int(options.get("train_windows_per_recording", 8)),
        training=True,
        seed=seed,
    )
    validation_dataset = RecordingBagDataset(
        manifest_path,
        config["_project_root"],
        split="validation",
        windows_per_recording=int(options.get("validation_windows_per_recording", 32)),
        training=False,
        seed=seed,
    )
    workers = int(options.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(options.get("recording_batch_size", 2)),
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(options.get("validation_recording_batch_size", 1)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    current_model_options = model_options(config)
    model = EEGPTMILClassifier(**current_model_options)
    freeze_epochs = int(options.get("freeze_encoder_epochs", 5))
    unfreeze_last_n = int(options.get("unfreeze_last_n_blocks", 2))
    start_epoch = 1
    resume_checkpoint: dict[str, Any] | None = None
    if resume_path is None:
        pretrained_report = load_pretrained_eegpt(
            model,
            pretrained_path,
            minimum_encoder_coverage=float(config["model"].get("minimum_encoder_coverage", 0.95)),
            report_path=run_directory / "pretrained_load_report.json",
        )
    else:
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_checkpoint.get("model_options") != current_model_options:
            raise ValueError("The resume checkpoint model options do not match the current YAML model section.")
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        pretrained_report = resume_checkpoint["pretrained_load_report"]
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        print(f"Resuming after epoch {start_epoch - 1} from {resume_path}")
    set_encoder_trainability(model, 0 if start_epoch <= freeze_epochs else unfreeze_last_n)
    model.to(device)
    print(f"Model parameters entering epoch {start_epoch}: {parameter_summary(model)}")

    negatives, positives = recording_class_counts(train_dataset)
    positive_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = build_optimizer(model, options)
    epochs = int(options.get("epochs", 40))
    accumulation = int(options.get("gradient_accumulation_steps", 1))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = max(1, updates_per_epoch * epochs)
    warmup_updates = int(options.get("warmup_epochs", 5)) * updates_per_epoch
    scheduler = build_scheduler(
        optimizer,
        total_updates=total_updates,
        warmup_updates=warmup_updates,
        minimum_lr_ratio=float(options.get("minimum_lr_ratio", 0.05)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_metric = -float("inf") if resume_checkpoint is None else float(
        resume_checkpoint.get("best_selection_metric", -float("inf"))
    )
    best_path = run_directory / "best.pt"
    history_path = run_directory / "history.csv"
    history_rows: list[dict[str, Any]] = (
        pd.read_csv(history_path).to_dict(orient="records") if resume_checkpoint is not None and history_path.exists() else []
    )
    patience_counter = 0 if resume_checkpoint is None else int(resume_checkpoint.get("patience_counter", 0))
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        if resume_checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        rng = resume_checkpoint.get("rng_state", {})
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
    selection_name = str(options.get("selection_metric", "auroc"))
    threshold_objective = str(options.get("threshold_objective", "balanced_accuracy"))
    encode_chunk_size = int(options.get("encode_chunk_size", 16))

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        train_dataset.set_epoch(epoch)
        if epoch == freeze_epochs + 1:
            set_encoder_trainability(model, unfreeze_last_n)
            print(f"Unfroze the final {unfreeze_last_n} EEGPT blocks: {parameter_summary(model)}")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        processed_recordings = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for step, batch in enumerate(progress, start=1):
            windows = batch["windows"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(windows, mask, encode_chunk_size=encode_chunk_size)
                full_loss = criterion(output["recording_logits"], labels)
                loss = full_loss / accumulation
            scaler.scale(loss).backward()

            should_update = step % accumulation == 0 or step == len(train_loader)
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(options.get("gradient_clip_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            batch_size = int(labels.numel())
            running_loss += float(full_loss.detach()) * batch_size
            processed_recordings += batch_size
            progress.set_postfix(loss=f"{running_loss / processed_recordings:.4f}")

        validation_predictions = predict_recordings(
            model,
            validation_loader,
            device,
            amp_enabled,
            encode_chunk_size,
        )
        threshold, validation_metrics = select_threshold(
            validation_predictions["label"].to_numpy(),
            validation_predictions["probability_abnormal"].to_numpy(),
            objective=threshold_objective,
        )
        validation_predictions.to_csv(run_directory / f"validation_predictions_epoch_{epoch:03d}.csv", index=False)
        if selection_name not in validation_metrics:
            raise KeyError(f"Unknown selection metric {selection_name!r}; available: {sorted(validation_metrics)}")
        selection_value = float(validation_metrics[selection_name])

        history_row = {
            "epoch": epoch,
            "train_recording_loss": running_loss / processed_recordings,
            "seconds": time.time() - epoch_start,
            "selection_metric": selection_name,
            "selection_value": selection_value,
            **{f"val_{name}": value for name, value in validation_metrics.items()},
            "head_lr": max(group["lr"] for group in optimizer.param_groups),
        }
        history_rows.append(history_row)
        pd.DataFrame(history_rows).to_csv(run_directory / "history.csv", index=False)
        print(
            f"Epoch {epoch}: val AUROC={validation_metrics['auroc']:.4f}, "
            f"balanced accuracy={validation_metrics['balanced_accuracy']:.4f}, "
            f"macro F1={validation_metrics['macro_f1']:.4f}, threshold={threshold:.3f}"
        )

        improved = selection_value > best_metric + float(options.get("minimum_improvement", 1e-4))
        if improved:
            best_metric = selection_value
            patience_counter = 0
        else:
            patience_counter += 1
        checkpoint_payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_metric,
            threshold,
            validation_metrics,
            current_model_options,
            pretrained_report,
            config,
            patience_counter,
        )
        atomic_torch_save(checkpoint_payload, run_directory / "last.pt")
        if improved:
            atomic_torch_save(checkpoint_payload, best_path)
            print(f"Saved new best checkpoint selected only on validation {selection_name}.")
        else:
            if patience_counter >= int(options.get("early_stopping_patience", 10)):
                print(f"Early stopping after {patience_counter} epochs without sufficient improvement.")
                break

    # Recalibrate the operating threshold once using all validation windows.
    # Per-epoch validation uses a smaller fixed subset for speed; this final
    # pass makes validation and final-test aggregation identical.
    if not best_path.exists():
        raise FileNotFoundError(
            f"No best checkpoint exists in the active run directory: {best_path}. "
            "Do not resume into a different run_name."
        )
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    model.to(device)
    calibration_windows = int(options.get("calibration_windows_per_recording", 0))
    calibration_dataset = RecordingBagDataset(
        manifest_path,
        config["_project_root"],
        split="validation",
        windows_per_recording=calibration_windows,
        training=False,
        seed=seed,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=1 if calibration_windows == 0 else int(options.get("validation_recording_batch_size", 1)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    full_validation_predictions = predict_recordings(
        model,
        calibration_loader,
        device,
        amp_enabled,
        encode_chunk_size,
    )
    final_threshold, full_validation_metrics = select_threshold(
        full_validation_predictions["label"].to_numpy(),
        full_validation_predictions["probability_abnormal"].to_numpy(),
        objective=threshold_objective,
    )
    full_validation_predictions.to_csv(run_directory / "validation_predictions_full_best.csv", index=False)
    best_checkpoint["selection_validation_metrics"] = best_checkpoint["validation_metrics"]
    best_checkpoint["validation_metrics"] = full_validation_metrics
    best_checkpoint["validation_threshold"] = float(final_threshold)
    best_checkpoint["threshold_calibration_windows_per_recording"] = calibration_windows
    atomic_torch_save(best_checkpoint, best_path)

    print(f"\nTraining complete. Best validation checkpoint: {best_path}")
    print(
        f"Final threshold {final_threshold:.3f} was calibrated using "
        f"{'all' if calibration_windows == 0 else calibration_windows} validation windows per recording."
    )
    print("The official evaluation partition has not been loaded by this script.")
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--allow-incomplete-evaluation-for-training",
        action="store_true",
        help="Train on a complete official training cohort while deferring missing evaluation EDFs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, ["paths", "split", "model", "training"])
    if args.allow_incomplete_evaluation_for_training:
        config["training"]["allow_incomplete_evaluation_for_training"] = True
    train(config)


if __name__ == "__main__":
    main()
