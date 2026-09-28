from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from nmt_deep4.checkpoint import load_checkpoint, save_checkpoint
from nmt_deep4.common import device_from_config, ensure_dirs, get_paths, load_config, save_json, set_global_seed
from nmt_deep4.data import FixedWindowDataset, RandomWindowDataset, load_channel_stats, load_manifest
from nmt_deep4.eval_utils import predict_recordings
from nmt_deep4.metrics import compute_binary_metrics, find_best_f1_threshold
from nmt_deep4.model import Deep4Net


def build_loader(ds, batch_size, shuffle, num_workers, seed, epoch=0):
    g = torch.Generator()
    g.manual_seed(int(seed) + int(epoch))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        generator=g,
    )


def append_history(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    ensure_dirs(paths)
    set_global_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))
    device = device_from_config(cfg)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    df = load_manifest(paths["manifest"])
    if not paths["stats"].exists():
        raise FileNotFoundError("Training stats missing. Run 02_preprocess_cache.py first.")
    missing_cache = [p for p in df["cache_path"] if not Path(p).exists()]
    if missing_cache:
        raise FileNotFoundError(f"Missing {len(missing_cache)} cache files. Run 02_preprocess_cache.py.")

    train_df = df[df["role"] == "train"].reset_index(drop=True)
    val_df = df[df["role"] == "val"].reset_index(drop=True)
    mean, std = load_channel_stats(paths["stats"])

    # Safety checks for normalization statistics.
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Normalization statistics contain NaN/Inf. Re-run 02_preprocess_cache.py.")
    if np.any(std <= 0):
        raise ValueError("Normalization statistics contain non-positive standard deviations.")

    tcfg = cfg["training"]
    sfreq = float(cfg["preprocessing"].get("sfreq", 100.0))
    n_times = int(round(float(tcfg.get("window_seconds", 6.0)) * sfreq))
    if n_times < 441:
        raise ValueError("Default Deep4Net needs about 441 samples minimum. Increase window_seconds.")

    train_ds = RandomWindowDataset(
        train_df, n_times=n_times,
        windows_per_recording=int(tcfg.get("windows_per_recording", 4)),
        mean=mean, std=std, seed=int(cfg.get("seed", 42)),
    )
    val_ds = FixedWindowDataset(
        val_df, n_times=n_times, mean=mean, std=std,
        windows_per_recording=int(tcfg.get("val_windows_per_recording", 8)),
    )

    model = Deep4Net(n_chans=19, n_times=n_times, drop_prob=float(tcfg.get("dropout", 0.5))).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    n_normal = int((train_df["label"].str.lower() == "normal").sum())
    n_abnormal = int((train_df["label"].str.lower() == "abnormal").sum())
    pos_weight_value = (n_normal / max(1, n_abnormal)) if bool(tcfg.get("use_pos_weight", True)) else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Train recordings: normal={n_normal}, abnormal={n_abnormal}, pos_weight={pos_weight_value:.4f}")

    optimizer = Adam(model.parameters(), lr=float(tcfg.get("lr", 1e-3)), weight_decay=float(tcfg.get("weight_decay", 0.0)))
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=float(tcfg.get("lr_factor", 0.5)),
        patience=int(tcfg.get("lr_patience", 3)), min_lr=float(tcfg.get("min_lr", 1e-5))
    )

    use_amp = bool(tcfg.get("amp", True) and device.type == "cuda")

    # ------------------------------------------------------------------
    # AMP safety guard
    # ------------------------------------------------------------------
    # The preprocessing stage may preserve an extreme recording as float32
    # rather than clipping it. Because the training data are z-normalized,
    # such an extreme value can become much larger than float16's finite
    # range (65504). If the preprocessing QC report indicates this risk,
    # automatically disable fp16 AMP for the entire run. This preserves the
    # signal values rather than silently clipping them.
    extreme_report = paths["manifest_dir"] / "extreme_amplitude_recordings.csv"
    if use_amp and extreme_report.exists():
        try:
            extreme_df = pd.read_csv(extreme_report)
            if not extreme_df.empty and "max_abs_uv" in extreme_df.columns:
                max_abs_uv = float(pd.to_numeric(extreme_df["max_abs_uv"], errors="coerce").max())
                min_std = float(np.min(std))
                max_abs_mean = float(np.max(np.abs(mean)))
                worst_case_z = (max_abs_uv + max_abs_mean) / max(min_std, 1e-12)

                if np.isfinite(worst_case_z) and worst_case_z > 60000.0:
                    print(
                        f"WARNING: preprocessing QC contains an extreme amplitude "
                        f"(max |EEG|={max_abs_uv:.3f} uV). With the train-only "
                        f"normalization statistics this can reach ~{worst_case_z:,.0f} "
                        f"standardized units, which is unsafe for fp16 AMP."
                    )
                    print("Disabling AMP automatically for numerical safety.")
                    use_amp = False
        except Exception as e:
            print(f"WARNING: could not inspect {extreme_report}: {e}")

    print(f"Automatic mixed precision (AMP): {'ON' if use_amp else 'OFF'}")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    latest_path = paths["checkpoint_dir"] / "latest.pt"
    best_path = paths["checkpoint_dir"] / "best.pt"
    history_path = paths["artifacts_dir"] / "history.csv"

    start_epoch = 0
    resume_batch = 0
    best_score = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    partial_loss_sum = 0.0
    partial_seen = 0

    if latest_path.exists() and not args.no_resume:
        state = load_checkpoint(latest_path, model, optimizer, scheduler, scaler, map_location=device, restore_rng=True)
        start_epoch = int(state.get("epoch", 0))
        resume_batch = int(state.get("next_batch", 0))
        best_score = float(state.get("best_score", -math.inf))
        best_epoch = int(state.get("best_epoch", -1))
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        partial_loss_sum = float(state.get("partial_loss_sum", 0.0))
        partial_seen = int(state.get("partial_seen", 0))
        print(f"Resuming from {latest_path}: epoch={start_epoch}, next_batch={resume_batch}")

    epochs = int(tcfg.get("epochs", 60))
    batch_size = int(tcfg.get("batch_size", 64))
    val_batch_size = int(tcfg.get("eval_batch_size", 128))
    num_workers = int(tcfg.get("num_workers", 2))
    checkpoint_every = int(tcfg.get("checkpoint_every_steps", 100))
    patience = int(tcfg.get("early_stopping_patience", 10))
    grad_clip_norm = float(tcfg.get("grad_clip_norm", 5.0))
    print(f"Gradient clipping max norm: {grad_clip_norm if grad_clip_norm > 0 else 'disabled'}")

    val_loader = build_loader(val_ds, val_batch_size, False, num_workers, int(cfg.get("seed", 42)))

    for epoch in range(start_epoch, epochs):
        train_ds.set_epoch(epoch)
        train_loader = build_loader(train_ds, batch_size, True, num_workers, int(cfg.get("seed", 42)), epoch)
        model.train()
        loss_sum = partial_loss_sum if (epoch == start_epoch and resume_batch > 0) else 0.0
        seen = partial_seen if (epoch == start_epoch and resume_batch > 0) else 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        try:
            for batch_idx, (x, y) in pbar:
                if epoch == start_epoch and batch_idx < resume_batch:
                    continue
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if not torch.isfinite(x).all():
                    raise FloatingPointError(
                        f"Non-finite input detected at epoch={epoch+1}, batch={batch_idx}. "
                        "Re-run preprocessing/cache validation."
                    )
                if not torch.isfinite(y).all():
                    raise FloatingPointError(
                        f"Non-finite target detected at epoch={epoch+1}, batch={batch_idx}."
                    )

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch+1}, batch={batch_idx}. "
                        f"AMP={'ON' if use_amp else 'OFF'}."
                    )

                scaler.scale(loss).backward()

                # Unscale before clipping so the threshold applies to the true gradients.
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError(
                            f"Non-finite gradient norm at epoch={epoch+1}, batch={batch_idx}."
                        )

                scaler.step(optimizer)
                scaler.update()

                b = int(x.size(0))
                loss_sum += float(loss.item()) * b
                seen += b
                pbar.set_postfix(loss=f"{loss_sum/max(1, seen):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

                if checkpoint_every > 0 and (batch_idx + 1) % checkpoint_every == 0:
                    save_checkpoint(
                        latest_path, model, optimizer, scheduler, scaler,
                        epoch=epoch, next_batch=batch_idx + 1,
                        best_score=best_score, best_epoch=best_epoch,
                        epochs_without_improvement=epochs_without_improvement,
                        partial_loss_sum=loss_sum, partial_seen=seen,
                    )
        except KeyboardInterrupt:
            save_checkpoint(
                latest_path, model, optimizer, scheduler, scaler,
                epoch=epoch, next_batch=batch_idx + 1 if 'batch_idx' in locals() else 0,
                best_score=best_score, best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                partial_loss_sum=loss_sum, partial_seen=seen,
            )
            print("\nInterrupted. Checkpoint saved. Re-run the same command to resume.")
            return

        train_loss = loss_sum / max(1, seen)

        val_pred = predict_recordings(model, val_loader, val_df, device, amp=use_amp, desc="Validation")
        val_m05 = compute_binary_metrics(val_pred["target"], val_pred["prob_abnormal"], threshold=0.5)
        score = val_m05.auroc
        scheduler.step(score)

        improved = score > best_score + 1e-6
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path, model, optimizer, scheduler, scaler,
                epoch=epoch + 1, next_batch=0,
                best_score=best_score, best_epoch=best_epoch,
                epochs_without_improvement=0,
                extra={"val_metrics_at_0.5": val_m05.to_dict()},
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            latest_path, model, optimizer, scheduler, scaler,
            epoch=epoch + 1, next_batch=0,
            best_score=best_score, best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            partial_loss_sum=0.0, partial_seen=0,
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy_0.5": val_m05.accuracy,
            "val_f1_0.5": val_m05.f1,
            "val_sensitivity_0.5": val_m05.sensitivity,
            "val_specificity_0.5": val_m05.specificity,
            "val_auroc": val_m05.auroc,
            "lr": optimizer.param_groups[0]["lr"],
            "best_epoch": best_epoch + 1,
            "best_val_auroc": best_score,
        }
        append_history(history_path, row)
        print(
            f"Epoch {epoch+1}: loss={train_loss:.4f} val_acc={val_m05.accuracy:.4f} "
            f"val_f1={val_m05.f1:.4f} val_sens={val_m05.sensitivity:.4f} "
            f"val_spec={val_m05.specificity:.4f} val_auc={val_m05.auroc:.4f}"
        )

        resume_batch = 0
        partial_loss_sum = 0.0
        partial_seen = 0

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without AUROC improvement.")
            break

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was created.")

    load_checkpoint(best_path, model, map_location=device)
    val_pred = predict_recordings(model, val_loader, val_df, device, amp=use_amp, desc="Best model validation")
    threshold, val_best = find_best_f1_threshold(val_pred["target"], val_pred["prob_abnormal"])
    val_pred["pred_0.5"] = (val_pred["prob_abnormal"] >= 0.5).astype(int)
    val_pred["pred_selected"] = (val_pred["prob_abnormal"] >= threshold).astype(int)
    val_pred.to_csv(paths["artifacts_dir"] / "validation_predictions.csv", index=False)
    save_json({
        "selected_on": "validation",
        "criterion": "maximum F1",
        "threshold": threshold,
        "metrics_at_selected_threshold": val_best.to_dict(),
        "best_checkpoint_epoch": best_epoch + 1,
        "best_validation_auroc_at_0.5": best_score,
    }, paths["artifacts_dir"] / "threshold.json")

    print(f"\nTraining complete. Best epoch: {best_epoch+1}; best val AUROC: {best_score:.4f}")
    print(f"Validation-selected F1 threshold: {threshold:.6f}")
    print("Best checkpoint:", best_path)


if __name__ == "__main__":
    main()
