from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import ensure_output_directories, load_config, resolved_config
from .data import WindowedRecordingDataset, window_parameters
from .metrics import aggregate_recording_predictions, binary_metrics
from .model import build_model, count_trainable_parameters
from .runtime import amp_context, make_grad_scaler
from .utils import (
    atomic_write_json,
    human_duration,
    print_heading,
    set_global_seed,
    stable_hash,
)


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name, suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        torch.save(value, temporary_name)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_processed_manifest(path: Path, expected_channels: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed manifest not found: {path}. Run preprocessing first."
        )
    frame = pd.read_csv(path)
    required = {
        "recording_id",
        "split",
        "label",
        "cache_path",
        "n_channels",
        "n_samples",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Processed manifest is missing columns: {sorted(missing)}")
    if not (frame["n_channels"].astype(int) == expected_channels).all():
        raise RuntimeError(
            "Cached channel counts do not match dataset.channels in config.yaml."
        )
    missing_cache = [path for path in frame["cache_path"] if not Path(path).exists()]
    if missing_cache:
        raise FileNotFoundError(
            f"Missing {len(missing_cache)} cached arrays; rerun preprocessing."
        )
    return frame


def _create_or_load_internal_split(
    train_frame: pd.DataFrame,
    split_path: Path,
    validation_fraction: float,
    seed: int,
) -> pd.DataFrame:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    if split_path.exists():
        assignment = pd.read_csv(split_path)
        expected_ids = set(train_frame["recording_id"].astype(str))
        assigned_ids = set(assignment["recording_id"].astype(str))
        if expected_ids != assigned_ids:
            raise RuntimeError(
                "The frozen internal split does not match the current processed training manifest. "
                "Resolve dataset/cache changes, then delete the run's splits folder to create a new split."
            )
        return assignment

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=float(validation_fraction), random_state=int(seed)
    )
    _train_indices, validation_indices = next(
        splitter.split(train_frame["recording_id"], train_frame["label"])
    )
    assignment = train_frame[["recording_id", "label"]].copy()
    assignment["internal_split"] = "development_train"
    assignment.loc[validation_indices, "internal_split"] = "development_validation"
    assignment = assignment.sort_values("recording_id").reset_index(drop=True)
    assignment.to_csv(split_path, index=False)
    return assignment


def _frame_for_assignment(
    train_frame: pd.DataFrame, assignment: pd.DataFrame, value: str
) -> pd.DataFrame:
    ids = set(
        assignment.loc[assignment["internal_split"] == value, "recording_id"].astype(
            str
        )
    )
    result = train_frame[train_frame["recording_id"].astype(str).isin(ids)].copy()
    if result.empty:
        raise RuntimeError(f"Internal split {value} is empty.")
    return result


def _class_weights(
    dataset: WindowedRecordingDataset, device: torch.device
) -> torch.Tensor:
    counts = np.bincount(dataset.window_labels, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError(
            f"Training windows do not contain both classes: {counts.tolist()}"
        )
    weights = len(dataset.window_labels) / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _loader(
    dataset: WindowedRecordingDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int = 2,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    worker_count = int(num_workers)
    loader_options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": shuffle,
        "num_workers": worker_count,
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(worker_count > 0),
        "generator": generator,
        "drop_last": False,
    }
    if worker_count > 0:
        loader_options["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**loader_options)


def _runtime_integer(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return int(default)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, found {raw_value!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, found {value}")
    return value


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    accumulation_steps: int,
    progress_description: str = "Training",
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_batches = len(loader)

    with tqdm(
        loader,
        total=total_batches,
        desc=progress_description,
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        mininterval=0.5,
        smoothing=0.1,
    ) as progress:
        for batch_index, (inputs, targets, _) in enumerate(progress):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            group_position = batch_index % accumulation_steps
            batches_left = total_batches - (batch_index - group_position)
            current_group_size = min(accumulation_steps, batches_left)

            with amp_context(device, amp_enabled):
                logits = model(inputs)
                unscaled_loss = criterion(logits, targets)
                loss = unscaled_loss / current_group_size
            scaler.scale(loss).backward()

            group_complete = group_position + 1 == current_group_size
            if group_complete:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = int(targets.shape[0])
            total_loss += float(unscaled_loss.detach().cpu()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += batch_size

            if batch_index % 10 == 0 or batch_index + 1 == total_batches:
                postfix: dict[str, str] = {
                    "loss": f"{total_loss / total_examples:.4f}",
                    "acc": f"{total_correct / total_examples:.4f}",
                }
                if device.type == "cuda":
                    postfix["GPU"] = (
                        f"{torch.cuda.memory_allocated(device) / 2**30:.2f} GiB"
                    )
                progress.set_postfix(postfix, refresh=False)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.inference_mode()
def _evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    threshold: float = 0.5,
    progress_description: str = "Validation",
    best_score_so_far: float | None = None,
    best_epoch_so_far: int | None = None,
    previous_epoch_score: float | None = None,
    current_epoch: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    probabilities: list[float] = []
    labels: list[int] = []
    recording_ids: list[str] = []

    with tqdm(
        loader,
        total=len(loader),
        desc=progress_description,
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        mininterval=0.5,
        smoothing=0.1,
    ) as progress:
        for batch_index, (inputs, targets, batch_recording_ids) in enumerate(
            progress
        ):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with amp_context(device, amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)
            batch_probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
            batch_size = int(targets.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            probabilities.extend(batch_probabilities.detach().cpu().numpy().tolist())
            labels.extend(targets.detach().cpu().numpy().astype(int).tolist())
            recording_ids.extend(str(value) for value in batch_recording_ids)

            if batch_index % 10 == 0 or batch_index + 1 == len(loader):
                postfix = {"loss": f"{total_loss / total_examples:.4f}"}
                if device.type == "cuda":
                    postfix["GPU"] = (
                        f"{torch.cuda.memory_allocated(device) / 2**30:.2f} GiB"
                    )
                progress.set_postfix(postfix, refresh=False)

        window_metrics = binary_metrics(labels, probabilities, threshold)
        recordings = aggregate_recording_predictions(
            recording_ids, labels, probabilities
        )
        recording_metrics = binary_metrics(
            recordings["true_label"], recordings["abnormal_probability"], threshold
        )
        current_auroc = recording_metrics["auroc"]
        selection_score = (
            float(current_auroc)
            if current_auroc is not None
            else -(total_loss / total_examples)
        )
        improved = best_score_so_far is None or selection_score > best_score_so_far
        displayed_best_score = selection_score if improved else best_score_so_far
        displayed_best_epoch = current_epoch if improved else best_epoch_so_far
        delta_text = (
            "n/a"
            if previous_epoch_score is None
            else f"{selection_score - previous_epoch_score:+.4f}"
        )
        final_postfix = {
            "rec_acc": f"{recording_metrics['accuracy']:.4f}",
            "rec_AUROC": (
                "n/a" if current_auroc is None else f"{float(current_auroc):.4f}"
            ),
            "delta": delta_text,
            "status": "NEW BEST" if improved else "NO IMPROVEMENT",
            "best": (
                "n/a"
                if displayed_best_epoch is None or displayed_best_score is None
                else f"E{displayed_best_epoch:02d}/{displayed_best_score:.4f}"
            ),
        }
        progress.set_postfix(final_postfix, refresh=True)
    return (
        {
            "loss": total_loss / total_examples,
            "window": window_metrics,
            "recording": recording_metrics,
        },
        recordings,
    )


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    phase: str,
    seed: int,
    history: list[dict[str, Any]],
    best_score: float | None,
    best_epoch: int | None,
    config_hash: str,
    channels: list[str],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "phase": phase,
        "epoch": int(epoch),
        "seed": int(seed),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": history,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "config_hash": config_hash,
        "channels": channels,
    }


def _run_phase(
    *,
    config: dict[str, Any],
    phase: str,
    phase_dir: Path,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
    epochs: int,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        if summary.get("status") == "complete":
            print(f"Phase {phase} is already complete; reusing it.")
            return summary

    training_cfg = config["training"]
    configured_micro_batch = int(training_cfg["micro_batch_size"])
    configured_accumulation = int(training_cfg["gradient_accumulation_steps"])
    configured_effective_batch = configured_micro_batch * configured_accumulation
    runtime_micro_batch = _runtime_integer(
        "MBK_RUNTIME_MICRO_BATCH_SIZE", configured_micro_batch
    )
    runtime_accumulation = _runtime_integer(
        "MBK_RUNTIME_ACCUMULATION_STEPS", configured_accumulation
    )
    runtime_effective_batch = runtime_micro_batch * runtime_accumulation
    if runtime_effective_batch != configured_effective_batch:
        raise RuntimeError(
            "Runtime settings must preserve the configured effective batch size "
            f"of {configured_effective_batch}; received {runtime_micro_batch} x "
            f"{runtime_accumulation} = {runtime_effective_batch}."
        )
    runtime_validation_batch = _runtime_integer(
        "MBK_RUNTIME_VALIDATION_BATCH_SIZE",
        int(training_cfg["validation_batch_size"]),
    )
    runtime_num_workers = _runtime_integer(
        "MBK_RUNTIME_NUM_WORKERS", int(training_cfg["num_workers"]), minimum=0
    )
    runtime_prefetch_factor = _runtime_integer(
        "MBK_RUNTIME_PREFETCH_FACTOR", 2
    )
    channels = list(config["dataset"]["channels"])
    window_size, stride, include_final = window_parameters(config)
    train_dataset = WindowedRecordingDataset(
        train_frame, window_size, stride, include_final
    )
    validation_dataset = (
        WindowedRecordingDataset(validation_frame, window_size, stride, include_final)
        if validation_frame is not None
        else None
    )

    requested_device = str(training_cfg.get("device", "cuda")).lower()
    if requested_device not in {"cuda", "cpu"}:
        raise ValueError("training.device must be cuda or cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by config.yaml but is unavailable.")
    device = torch.device(requested_device)
    set_global_seed(seed, bool(training_cfg["deterministic"]))
    model = build_model(config, n_channels=len(channels)).to(device)
    weights = _class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        betas=(float(training_cfg["beta1"]), float(training_cfg["beta2"])),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs) - 1)
    )
    amp_enabled = bool(training_cfg["amp"])
    scaler = make_grad_scaler(device, amp_enabled)
    config_hash = stable_hash(config)

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_score: float | None = None
    best_epoch: int | None = None
    last_path = phase_dir / "last.pt"
    best_path = phase_dir / "best.pt"

    if last_path.exists():
        if not resume:
            raise RuntimeError(
                f"An incomplete checkpoint exists at {last_path}. Run 06_resume_training.ps1."
            )
        checkpoint = _torch_load(last_path, device)
        if checkpoint["phase"] != phase or checkpoint["config_hash"] != config_hash:
            raise RuntimeError(
                "Checkpoint phase/configuration does not match this run."
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        history = list(checkpoint.get("history", []))
        best_score = checkpoint.get("best_score")
        best_epoch = checkpoint.get("best_epoch")
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming {phase} at epoch {start_epoch}/{epochs}.")

    validation_loader = None
    if validation_dataset is not None:
        validation_loader = _loader(
            validation_dataset,
            runtime_validation_batch,
            False,
            seed,
            runtime_num_workers,
            bool(training_cfg["pin_memory"]),
            runtime_prefetch_factor,
        )

    phase_start = time.perf_counter()
    print_heading(f"Training phase: {phase}")
    print(f"recordings: {len(train_frame)}")
    print(f"windows: {len(train_dataset)}")
    print(f"class weights [normal, abnormal]: {weights.detach().cpu().tolist()}")
    print(f"trainable parameters: {count_trainable_parameters(model):,}")
    print(
        "runtime batching: "
        f"micro={runtime_micro_batch}, accumulation={runtime_accumulation}, "
        f"effective={runtime_effective_batch}"
    )
    print(
        "runtime loading: "
        f"workers={runtime_num_workers}, prefetch={runtime_prefetch_factor}, "
        f"validation_batch={runtime_validation_batch}"
    )

    for epoch in range(start_epoch, int(epochs) + 1):
        epoch_start = time.perf_counter()
        train_loader = _loader(
            train_dataset,
            runtime_micro_batch,
            True,
            seed + epoch,
            runtime_num_workers,
            bool(training_cfg["pin_memory"]),
            runtime_prefetch_factor,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            runtime_accumulation,
            progress_description=f"{phase} E{epoch:02d}/{epochs:02d} train",
        )

        row: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_window_accuracy": train_metrics["accuracy"],
            "runtime_micro_batch_size": runtime_micro_batch,
            "runtime_gradient_accumulation_steps": runtime_accumulation,
            "runtime_effective_batch_size": runtime_effective_batch,
            "runtime_num_workers": runtime_num_workers,
            "runtime_prefetch_factor": runtime_prefetch_factor,
        }
        score: float | None = None
        if validation_loader is not None:
            previous_epoch_score = None
            if history:
                previous_value = history[-1].get("validation_recording_auroc")
                if previous_value is not None:
                    previous_epoch_score = float(previous_value)
            validation_metrics, recording_predictions = _evaluate_dataset(
                model,
                validation_loader,
                criterion,
                device,
                amp_enabled,
                threshold=0.5,
                progress_description=(
                    f"{phase} E{epoch:02d}/{epochs:02d} validation"
                ),
                best_score_so_far=best_score,
                best_epoch_so_far=best_epoch,
                previous_epoch_score=previous_epoch_score,
                current_epoch=epoch,
            )
            recording_predictions.to_csv(
                phase_dir / "latest_validation_recording_predictions.csv", index=False
            )
            row.update(
                {
                    "validation_loss": validation_metrics["loss"],
                    "validation_window_accuracy": validation_metrics["window"][
                        "accuracy"
                    ],
                    "validation_recording_accuracy": validation_metrics["recording"][
                        "accuracy"
                    ],
                    "validation_recording_balanced_accuracy": validation_metrics[
                        "recording"
                    ]["balanced_accuracy"],
                    "validation_recording_f1": validation_metrics["recording"][
                        "f1_abnormal"
                    ],
                    "validation_recording_sensitivity": validation_metrics["recording"][
                        "sensitivity"
                    ],
                    "validation_recording_specificity": validation_metrics["recording"][
                        "specificity"
                    ],
                    "validation_recording_auroc": validation_metrics["recording"][
                        "auroc"
                    ],
                    "validation_recording_pr_auc": validation_metrics["recording"][
                        "pr_auc"
                    ],
                }
            )
            score = validation_metrics["recording"]["auroc"]
            if score is None:
                score = -float(validation_metrics["loss"])
            improved = best_score is None or float(score) > float(best_score)
            if improved:
                best_score = float(score)
                best_epoch = int(epoch)
            row.update(
                {
                    "validation_recording_auroc_delta_previous_epoch": (
                        None
                        if previous_epoch_score is None
                        else float(score) - previous_epoch_score
                    ),
                    "validation_recording_auroc_improved": bool(improved),
                    "best_validation_recording_auroc_so_far": best_score,
                    "best_epoch_so_far": best_epoch,
                }
            )

        scheduler.step()
        row["epoch_seconds"] = time.perf_counter() - epoch_start
        history.append(row)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            phase=phase,
            seed=seed,
            history=history,
            best_score=best_score,
            best_epoch=best_epoch,
            config_hash=config_hash,
            channels=channels,
        )
        _atomic_torch_save(payload, last_path)
        if validation_loader is not None and best_epoch == epoch:
            _atomic_torch_save(payload, best_path)
        pd.DataFrame(history).to_csv(phase_dir / "history.csv", index=False)

        validation_text = ""
        if validation_loader is not None:
            delta_value = row["validation_recording_auroc_delta_previous_epoch"]
            delta_text = "n/a" if delta_value is None else f"{delta_value:+.4f}"
            improvement_text = (
                "NEW BEST"
                if row["validation_recording_auroc_improved"]
                else "NO IMPROVEMENT"
            )
            validation_text = (
                f" | val rec Acc {row['validation_recording_accuracy']:.4f}"
                f" | val rec AUROC {row['validation_recording_auroc']:.4f}"
                f" | delta {delta_text} | {improvement_text}"
                f" | best E{best_epoch:02d} AUROC {best_score:.4f}"
            )
        memory_gib = (
            torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        )
        print(
            f"Epoch {epoch:02d}/{epochs} | train loss {row['train_loss']:.4f}"
            f" | train win Acc {row['train_window_accuracy']:.4f}{validation_text}"
            f" | {human_duration(row['epoch_seconds'])} | peak {memory_gib:.2f} GiB"
        )

    if validation_dataset is None:
        best_epoch = int(epochs)
    summary = {
        "status": "complete",
        "phase": phase,
        "seed": int(seed),
        "epochs_completed": int(epochs),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "training_recordings": len(train_frame),
        "training_windows": len(train_dataset),
        "validation_recordings": len(validation_frame)
        if validation_frame is not None
        else 0,
        "validation_windows": len(validation_dataset)
        if validation_dataset is not None
        else 0,
        "class_weights": weights.detach().cpu().tolist(),
        "runtime_micro_batch_size": runtime_micro_batch,
        "runtime_gradient_accumulation_steps": runtime_accumulation,
        "runtime_effective_batch_size": runtime_effective_batch,
        "runtime_validation_batch_size": runtime_validation_batch,
        "runtime_num_workers": runtime_num_workers,
        "runtime_prefetch_factor": runtime_prefetch_factor,
        "elapsed_seconds": time.perf_counter() - phase_start,
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated() / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path) if best_path.exists() else None,
    }
    atomic_write_json(summary_path, summary)
    return summary


def _export_final_model(
    source_checkpoint: Path, destination: Path, config: dict[str, Any]
) -> None:
    source = _torch_load(source_checkpoint, "cpu")
    export = {
        "format_version": 1,
        "model_state": source["model_state"],
        "epoch": source["epoch"],
        "phase": source["phase"],
        "seed": source["seed"],
        "channels": source["channels"],
        "config_hash": source["config_hash"],
        "model_config": config["model"],
        "preprocessing_config": config["preprocessing"],
    }
    _atomic_torch_save(export, destination)


def train_pipeline(config_path: str | Path, resume: bool = False) -> int:
    config, paths = load_config(config_path)
    ensure_output_directories(paths)
    if (
        str(config["training"].get("device", "cuda")).lower() == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA is unavailable. Run 01_verify_install.ps1 first.")

    seed = int(config["training"]["seed"])
    run_dir = paths.runs_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "resolved_config.json", resolved_config(config, paths))

    manifest = _load_processed_manifest(
        paths.audit_dir / "processed_manifest.csv", len(config["dataset"]["channels"])
    )
    train_frame = manifest[manifest["split"] == "train"].copy().reset_index(drop=True)
    evaluation_frame = manifest[manifest["split"] == "evaluation"]
    expected_eval = sum(config["dataset"]["expected_counts"]["evaluation"].values())
    if len(evaluation_frame) != expected_eval:
        raise RuntimeError(
            f"Expected {expected_eval} successfully preprocessed evaluation recordings, "
            f"found {len(evaluation_frame)}. Evaluation must remain complete."
        )

    configured_mode = str(config["training"]["mode"])
    maximum_epochs = int(config["training"]["maximum_epochs"])
    if configured_mode not in {"development_then_full", "development_best_only"}:
        raise ValueError(
            "training.mode must be development_best_only. "
            "development_then_full is accepted as a legacy alias."
        )

    effective_mode = "development_best_only"
    print_heading("Model-selection protocol")
    print("effective mode: development_best_only")
    print("selection metric: recording-level internal-validation AUROC")
    print("full-data retraining: disabled")
    print("official evaluation during training: disabled")

    assignment = _create_or_load_internal_split(
        train_frame,
        run_dir / "splits" / "internal_split.csv",
        float(config["training"]["validation_fraction"]),
        seed,
    )
    development_train = _frame_for_assignment(
        train_frame, assignment, "development_train"
    )
    development_validation = _frame_for_assignment(
        train_frame, assignment, "development_validation"
    )
    development_summary = _run_phase(
        config=config,
        phase="development",
        phase_dir=run_dir / "development",
        train_frame=development_train,
        validation_frame=development_validation,
        epochs=maximum_epochs,
        seed=seed,
        resume=resume,
    )
    selected_epoch = int(development_summary["best_epoch"])
    best_validation_auroc = float(development_summary["best_score"])
    best_checkpoint_value = development_summary.get("best_checkpoint")
    if not best_checkpoint_value:
        raise RuntimeError("Best validation checkpoint was not created.")
    best_checkpoint = Path(str(best_checkpoint_value))
    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"Best validation checkpoint is missing: {best_checkpoint}"
        )

    final_checkpoint = run_dir / "best_validation_model.pt"
    _export_final_model(best_checkpoint, final_checkpoint, config)
    atomic_write_json(
        run_dir / "epoch_selection.json",
        {
            "selection_source": "internal_training_validation_only",
            "metric": "recording_auroc",
            "selected_epoch": selected_epoch,
            "best_validation_auroc": best_validation_auroc,
            "maximum_epochs_tested": maximum_epochs,
            "best_checkpoint": str(best_checkpoint),
            "full_training_restarted": False,
            "evaluation_partition_used": False,
        },
    )

    completion = {
        "status": "complete",
        "configured_mode": configured_mode,
        "effective_mode": effective_mode,
        "final_model_strategy": (
            "best_internal_validation_auroc_checkpoint_no_retraining"
        ),
        "seed": seed,
        "selected_epoch": selected_epoch,
        "best_validation_auroc": best_validation_auroc,
        "development_training_recordings": len(development_train),
        "internal_validation_recordings": len(development_validation),
        "best_checkpoint": str(best_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "full_data_retraining_performed": False,
        "evaluation_partition_used_during_training": False,
    }
    atomic_write_json(run_dir / "training_complete.json", completion)
    print_heading("Training complete")
    print(json.dumps(completion, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Multi-BK-Net on NMT-4K.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        raise SystemExit(train_pipeline(args.config, args.resume))
    except torch.cuda.OutOfMemoryError as exc:
        print(f"CUDA out of memory: {exc}")
        print(
            "For an optimized resume, set MBK_RUNTIME_MICRO_BATCH_SIZE=32 and "
            "MBK_RUNTIME_ACCUMULATION_STEPS=2, then resume again."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
