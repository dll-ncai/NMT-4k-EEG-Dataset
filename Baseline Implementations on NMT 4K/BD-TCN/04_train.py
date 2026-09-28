from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bdtcn_nmt_utils import (
    CachedWindowDataset,
    build_model,
    capture_rng_state,
    choose_threshold_f1,
    load_config,
    make_or_load_validation_split,
    output_root,
    predict_recordings,
    read_metadata,
    restore_rng_state,
    set_all_seeds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--resume",
        default="auto",
        help="'auto', 'none', or explicit checkpoint path.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one small forward/backward sanity check and exit.",
    )
    return p.parse_args()


def checkpoint_paths(out: Path):
    d = out / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d / "last.pt", d / "best.pt"


def classification_loss(logits, targets, criterion, l2_penalty, model):
    """Cross-entropy for either dense BxCxT logits or pooled BxC logits."""
    if logits.ndim == 3:
        # Apply the recording/window target at every valid dense temporal
        # prediction, matching the cropped/dense training logic of the original
        # BD-TCN implementation.
        b, c, t = logits.shape
        dense_logits = logits.transpose(1, 2).reshape(b * t, c)
        dense_targets = targets[:, None].expand(b, t).reshape(b * t)
        loss = criterion(dense_logits, dense_targets)
    elif logits.ndim == 2:
        loss = criterion(logits, targets)
    else:
        raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")

    if l2_penalty > 0:
        l2 = torch.zeros((), device=logits.device, dtype=torch.float32)
        for p in model.parameters():
            if p.requires_grad:
                l2 = l2 + p.float().pow(2).sum()
        loss = loss + l2_penalty * l2
    return loss


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["training"]["seed"])
    set_all_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke_test:
        raise RuntimeError(
            "CUDA is not available. Full BD-TCN training on CPU is not recommended. "
            "Run 01_check_setup.py and fix the GPU environment first."
        )

    df = read_metadata(cfg)
    fit_df, val_df = make_or_load_validation_split(df, cfg, args.config)

    print(f"Fit recordings       : {len(fit_df):,}")
    print(f"Validation recordings: {len(val_df):,}")
    print("Fit label counts:")
    print(fit_df["label"].value_counts().to_string())
    print("Validation label counts:")
    print(val_df["label"].value_counts().to_string())

    train_ds = CachedWindowDataset(fit_df, cfg, args.config)
    print(f"Training windows     : {len(train_ds):,}")

    batch_size = int(cfg["training"]["batch_size"])
    workers = int(cfg["training"]["num_workers"])
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
        drop_last=False,
    )

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters     : {n_params:,}")
    print(f"Device               : {device}")

    class_weight = None
    if bool(cfg["training"].get("class_weighting", False)):
        counts = fit_df["label"].value_counts()
        n_normal = float(counts.get("Normal", 1))
        n_abnormal = float(counts.get("Abnormal", 1))
        total = n_normal + n_abnormal
        weights = torch.tensor(
            [total / (2 * n_normal), total / (2 * n_abnormal)],
            dtype=torch.float32,
            device=device,
        )
        class_weight = weights
        print(f"Class weights        : {weights.detach().cpu().tolist()}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)

    if args.smoke_test:
        xb, yb = next(iter(loader))
        xb, yb = xb[:2].to(device), yb[:2].to(device)
        model.train()
        use_amp = bool(cfg["training"]["mixed_precision"]) and device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=use_amp,
        ):
            logits = model(xb)
            loss = classification_loss(
                logits,
                yb,
                criterion,
                float(cfg["training"].get("l2_penalty", 0.0)),
                model,
            )
        loss.backward()
        print(f"Smoke-test input     : {tuple(xb.shape)}")
        print(f"Smoke-test logits    : {tuple(logits.shape)}")
        print(f"Smoke-test loss      : {float(loss.detach().cpu()):.6f}")
        print("Smoke test PASSED.")
        return

    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"]["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    max_epochs = int(cfg["training"]["max_epochs"])
    accum = int(cfg["training"]["gradient_accumulation_steps"])
    if accum < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    optimizer_steps_per_epoch = math.ceil(len(loader) / accum)
    total_optimizer_steps = max_epochs * optimizer_steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_optimizer_steps),
        eta_min=0.0,
    )

    use_amp = bool(cfg["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = output_root(cfg, args.config)
    out.mkdir(parents=True, exist_ok=True)
    last_path, best_path = checkpoint_paths(out)
    history_path = out / "training_history.csv"

    start_epoch = 0
    best_score = -float("inf")
    epochs_without_improvement = 0
    history = []

    resume_arg = str(args.resume).lower()
    resume_path = None
    if resume_arg == "auto" and last_path.exists():
        resume_path = last_path
    elif resume_arg not in {"auto", "none"}:
        resume_path = Path(args.resume)

    if resume_path is not None:
        print(f"Resuming from: {resume_path}")
        # Load the checkpoint on CPU first.
        # The checkpoint contains PyTorch RNG states. If map_location=device
        # moves the saved CPU RNG ByteTensor to CUDA, torch.set_rng_state()
        # fails because it requires a CPU ByteTensor.
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_score = float(ckpt.get("best_score", -float("inf")))
        epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
        history = list(ckpt.get("history", []))
        rng_state = ckpt.get("rng_state")
        if rng_state is not None:
            try:
                restore_rng_state(rng_state)
            except (TypeError, RuntimeError) as e:
                print(
                    "WARNING: Exact RNG state could not be restored from the "
                    f"checkpoint ({e}). Training will still resume from epoch "
                    f"{start_epoch + 1}; the process seed remains {seed}."
                )
        print(f"Continuing at epoch {start_epoch + 1}/{max_epochs}")

    patience = int(cfg["training"].get("early_stopping_patience", 0))
    clip_norm = float(cfg["training"]["gradient_clip_norm"])

    for epoch in range(start_epoch, max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        seen = 0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1:02d}/{max_epochs:02d} training",
            unit="batch",
        )
        for step, (xb, yb) in enumerate(pbar):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                logits = model(xb)
                loss = classification_loss(
                    logits,
                    yb,
                    criterion,
                    float(cfg["training"].get("l2_penalty", 0.0)),
                    model,
                )
                scaled_loss = loss / accum

            scaler.scale(scaled_loss).backward()
            should_step = ((step + 1) % accum == 0) or (step + 1 == len(loader))

            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                # Legacy BD-TCN scheduled both learning rate and decoupled
                # weight decay with the cosine schedule.
                lr_ratio = (
                    optimizer.param_groups[0]["lr"] / lr
                    if lr > 0 else 0.0
                )
                for group in optimizer.param_groups:
                    group["weight_decay"] = wd * lr_ratio

            batch_n = yb.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_n
            seen += batch_n
            pbar.set_postfix(
                loss=f"{total_loss / max(1, seen):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_loss = total_loss / max(1, seen)

        val_pred = predict_recordings(
            model,
            val_df,
            cfg,
            args.config,
            device,
            desc=f"Epoch {epoch + 1:02d} validation",
        )
        y_true = val_pred["y_true"].to_numpy()
        probs = val_pred["prob_abnormal"].to_numpy()
        threshold, val_metrics = choose_threshold_f1(y_true, probs)

        score = float(val_metrics[str(cfg["training"]["selection_metric"])])
        improved = score > best_score
        if improved:
            best_score = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_threshold": threshold,
            **{f"val_{k}": v for k, v in val_metrics.items() if k not in {"tn", "fp", "fn", "tp"}},
            "val_tn": val_metrics["tn"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
            "val_tp": val_metrics["tp"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "validation_threshold": threshold,
            "validation_metrics": val_metrics,
            "rng_state": capture_rng_state(),
            "config": cfg,
        }

        # Save last checkpoint atomically.
        temp_last = last_path.with_suffix(".tmp.pt")
        torch.save(state, temp_last)
        temp_last.replace(last_path)

        if improved:
            temp_best = best_path.with_suffix(".tmp.pt")
            torch.save(state, temp_best)
            temp_best.replace(best_path)

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_sens={val_metrics['sensitivity']:.4f} | "
            f"val_spec={val_metrics['specificity']:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f} | "
            f"threshold={threshold:.4f}"
        )
        if improved:
            print(f"New best checkpoint -> {best_path}")

        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"Early stopping after {epochs_without_improvement} epochs "
                "without validation improvement."
            )
            break

    print("\nTraining finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
