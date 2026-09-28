from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config, project_path
from src.data import CachedEEGDataset, make_weighted_sampler
from src.model import build_model
from src.train_utils import (
    atomic_torch_save,
    compute_metrics,
    evaluate_loader,
    get_device,
    seed_everything,
)


def make_loader(dataset, batch_size, num_workers, pin_memory, shuffle=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--resume", default="auto", help="auto | none | path-to-checkpoint")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["data"]["seed"])
    seed_everything(seed)

    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        raise SystemExit("CUDA GPU not available. Fix PyTorch/CUDA installation before training.")

    manifest = pd.read_csv(project_path(cfg, cfg["data"]["manifest_file"]))
    cache_dir = Path(cfg["data"]["cache_dir"])
    train_ds = CachedEEGDataset(
        manifest, cache_dir, role="fit",
        include_reverse_augmentation=bool(cfg["data"].get("time_reverse_augmentation", True)),
    )
    val_ds = CachedEEGDataset(manifest, cache_dir, role="validation", include_reverse_augmentation=False)

    tr_cfg = cfg["training"]
    batch_size = int(tr_cfg["batch_size"])
    num_workers = int(tr_cfg["num_workers"])
    pin_memory = bool(tr_cfg["pin_memory"] and device.type == "cuda")

    sampler = None
    shuffle = True
    if bool(tr_cfg.get("balanced_sampling", True)):
        sampler = make_weighted_sampler(train_ds.labels, seed=seed)
        shuffle = False

    train_loader = make_loader(train_ds, batch_size, num_workers, pin_memory, shuffle=shuffle, sampler=sampler)
    val_loader = make_loader(val_ds, batch_size, num_workers, pin_memory, shuffle=False)

    print(f"Training samples (including available reverse augmentation): {len(train_ds)}")
    print(f"Validation recordings: {len(val_ds)}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(tr_cfg["learning_rate"]),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(tr_cfg["lr_reduce_factor"]),
        patience=int(tr_cfg["lr_patience"]),
        min_lr=float(tr_cfg["min_learning_rate"]),
    )

    amp_enabled = bool(tr_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    accum_steps = max(1, int(tr_cfg.get("gradient_accumulation_steps", 1)))
    max_grad_norm = float(tr_cfg.get("max_grad_norm", 0.0))

    run_dir = project_path(cfg, tr_cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = run_dir / "last.pt"
    best_ckpt = run_dir / "best.pt"
    history_path = run_dir / "history.csv"

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    resume_path = None
    if args.resume.lower() == "auto" and last_ckpt.exists():
        resume_path = last_ckpt
    elif args.resume.lower() not in ("auto", "none"):
        resume_path = Path(args.resume)

    if resume_path is not None:
        print(f"Resuming: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        patience_counter = int(ckpt.get("patience_counter", 0))
        print(f"Continuing from epoch {start_epoch + 1}")

    max_epochs = int(tr_cfg["max_epochs"])
    early_patience = int(tr_cfg["early_stopping_patience"])

    for epoch in range(start_epoch, max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} train")
        for step, (x, y, _, _) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y) / accum_steps

            scaler.scale(loss).backward()

            do_step = ((step + 1) % accum_steps == 0) or (step + 1 == len(train_loader))
            if do_step:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_loss = float(loss.item()) * accum_steps
            running_loss += batch_loss * x.size(0)
            seen += x.size(0)
            pbar.set_postfix(loss=f"{running_loss/max(1,seen):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss = running_loss / max(1, seen)
        val_out = evaluate_loader(model, val_loader, device, amp_enabled, desc="Validation")
        val_metrics = compute_metrics(val_out["y_true"], val_out["prob_abnormal"], threshold=0.5)
        val_loss = float(val_out["loss"])
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_auroc": val_metrics["auroc"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        write_header = not history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                w.writeheader()
            w.writerow(row)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if amp_enabled else None,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "config": cfg,
        }
        atomic_torch_save(state, last_ckpt)
        if improved:
            atomic_torch_save(state, best_ckpt)

        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"Acc={val_metrics['accuracy']*100:.2f}% F1={val_metrics['f1']*100:.2f}% "
            f"Sens={val_metrics['sensitivity']*100:.2f}% Spec={val_metrics['specificity']*100:.2f}% "
            f"AUROC={val_metrics['auroc']*100:.2f}% patience={patience_counter}/{early_patience}"
        )

        if patience_counter >= early_patience:
            print("Early stopping triggered.")
            break

    print(f"\nBest checkpoint: {best_ckpt}")
    print(f"Last checkpoint: {last_ckpt}")


if __name__ == "__main__":
    main()
