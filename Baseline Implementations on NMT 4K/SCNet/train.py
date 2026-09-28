from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (
    get_rng_state, load_config, resolve_path, save_json, seed_everything, set_rng_state,
)
from src.data import CachedEEGDataset, make_or_load_splits
from src.metrics import binary_metrics, find_best_threshold
from src.scnet import SCNet, count_trainable_parameters


def atomic_torch_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def make_loader(dataset, batch_size, shuffle, workers, pin_memory, generator=None):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


@torch.no_grad()
def run_validation(model, loader, criterion, device, amp_enabled):
    model.eval()
    total_loss = 0.0
    n = 0
    ys, probs = [], []
    bar = tqdm(loader, desc="Validation", leave=False, unit="batch")
    for x, y, _ in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)
        p = torch.softmax(logits.float(), dim=1)[:, 1]
        total_loss += float(loss.item()) * y.numel()
        n += y.numel()
        ys.append(y.cpu().numpy())
        probs.append(p.cpu().numpy())
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, n), np.concatenate(ys), np.concatenate(probs)


def build_checkpoint(
    model, optimizer, scaler, epoch, batch_in_epoch, global_step, best_auroc,
    best_threshold, best_epoch, epochs_without_improvement, history, rng_state,
    input_channels, cfg,
):
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "global_step": int(global_step),
        "best_auroc": float(best_auroc),
        "best_threshold": float(best_threshold),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "history": history,
        "rng_state": rng_state,
        "input_channels": int(input_channels),
        "config": cfg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SCNet on cached NMT-4K EEG data.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", choices=["auto", "never"], default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accum-steps", type=int, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    base = cfg_path.parent
    cache_dir = resolve_path(base, cfg["cache_dir"])
    output_dir = resolve_path(base, cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = dict(cfg["train"])
    seed = int(train_cfg.get("seed", 42))
    seed_everything(seed)

    manifest_path = cache_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run preprocess.py first. Missing {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype={"file_name": str})
    errors = manifest[manifest["status"] != "ok"]
    if len(errors):
        raise RuntimeError(
            f"Preprocessing manifest contains {len(errors)} errors. Fix them before training."
        )

    train_df, val_df, eval_df = make_or_load_splits(
        manifest,
        output_dir=output_dir,
        val_fraction=float(train_cfg.get("val_fraction", 0.15)),
        seed=seed,
    )
    print(f"Model fitting: {len(train_df)} recordings")
    print(f"Validation:    {len(val_df)} recordings")
    print(f"Held-out eval: {len(eval_df)} recordings (not used during training)")
    print("Train class counts:", train_df["label"].value_counts().to_dict())
    print("Val class counts:  ", val_df["label"].value_counts().to_dict())

    pp_mode = cfg["preprocess"]["mode"]
    input_channels = 19 if pp_mode == "nmt4k19" else 22
    model = SCNet(input_channels=input_channels, num_classes=2)
    print(f"SCNet trainable parameters: {count_trainable_parameters(model):,}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This 7-minute SCNet training setup is intended for the NVIDIA GPU. "
            "Run check_setup.py and reinstall the CUDA-enabled PyTorch wheel if needed."
        )
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    model.to(device)

    micro_batch = int(args.batch_size or train_cfg.get("batch_size", 8))
    accum_steps = int(args.accum_steps or train_cfg.get("grad_accum_steps", 4))
    workers = int(train_cfg.get("num_workers", 2))
    amp_enabled = bool(train_cfg.get("amp", True))
    epochs = int(train_cfg.get("epochs", 60))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    patience = int(train_cfg.get("early_stopping_patience", 10))
    save_every = int(train_cfg.get("save_every_optimizer_steps", 200))
    threshold_objective = str(train_cfg.get("threshold_tuning", "f1"))
    print(f"Micro-batch={micro_batch}, accumulation={accum_steps}, effective batch≈{micro_batch * accum_steps}")
    print(f"AMP={amp_enabled}, workers={workers}")

    train_ds = CachedEEGDataset(train_df, cache_dir)
    val_ds = CachedEEGDataset(val_df, cache_dir)
    val_loader = make_loader(
        val_ds, micro_batch, shuffle=False, workers=workers,
        pin_memory=True, generator=None,
    )

    class_weight_mode = str(train_cfg.get("class_weights", "none")).lower()
    weights = None
    if class_weight_mode == "balanced":
        counts = train_df["target"].value_counts().to_dict()
        n_total = len(train_df)
        w0 = n_total / (2.0 * counts.get(0, 1))
        w1 = n_total / (2.0 * counts.get(1, 1))
        weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        print(f"Using balanced class weights: normal={w0:.4f}, abnormal={w1:.4f}")
    elif class_weight_mode != "none":
        raise ValueError("train.class_weights must be 'none' or 'balanced'")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    resume_batch = 0
    global_step = 0
    best_auroc = -math.inf
    best_threshold = 0.5
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    last_path = ckpt_dir / "last.pt"
    if args.resume == "auto" and last_path.exists():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", 0))
        resume_batch = int(ckpt.get("batch_in_epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        best_auroc = float(ckpt.get("best_auroc", -math.inf))
        best_threshold = float(ckpt.get("best_threshold", 0.5))
        best_epoch = int(ckpt.get("best_epoch", -1))
        epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
        history = ckpt.get("history", [])
        set_rng_state(ckpt.get("rng_state"))
        print(f"Resuming from epoch {start_epoch + 1}, batch {resume_batch}, global step {global_step}")

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            g = torch.Generator()
            g.manual_seed(seed + epoch)
            train_loader = make_loader(
                train_ds, micro_batch, shuffle=True, workers=workers,
                pin_memory=True, generator=g,
            )
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            seen = 0
            optimizer_steps_this_epoch = 0

            bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            for batch_idx, (x, y, _) in bar:
                if epoch == start_epoch and batch_idx < resume_batch:
                    continue

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                try:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                        logits = model(x)
                        raw_loss = criterion(logits, y)
                        loss = raw_loss / accum_steps
                    scaler.scale(loss).backward()
                except torch.cuda.OutOfMemoryError as e:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"CUDA OOM with micro-batch {micro_batch}. Re-run with e.g. "
                        f"--batch-size {max(1, micro_batch // 2)} --accum-steps {accum_steps * 2}."
                    ) from e

                running_loss += float(raw_loss.item()) * y.numel()
                seen += y.numel()
                do_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(train_loader))
                if do_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    optimizer_steps_this_epoch += 1

                    if save_every > 0 and global_step % save_every == 0:
                        state = build_checkpoint(
                            model, optimizer, scaler, epoch, batch_idx + 1, global_step,
                            best_auroc, best_threshold, best_epoch, epochs_without_improvement,
                            history, get_rng_state(), input_channels, cfg,
                        )
                        atomic_torch_save(state, last_path)

                bar.set_postfix(
                    loss=f"{raw_loss.item():.4f}",
                    avg=f"{running_loss / max(1, seen):.4f}",
                    vram=f"{torch.cuda.memory_allocated()/1024**3:.2f}G",
                )

            resume_batch = 0
            train_loss = running_loss / max(1, seen)
            val_loss, y_val, p_val = run_validation(model, val_loader, criterion, device, amp_enabled)
            tuned_threshold, tuned_val_metrics = find_best_threshold(
                y_val, p_val, objective=threshold_objective
            )
            val_metrics_05 = binary_metrics(y_val, p_val, threshold=0.5)
            val_auroc = val_metrics_05["auroc"]

            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auroc": val_auroc,
                "val_accuracy_05": val_metrics_05["accuracy"],
                "val_f1_05": val_metrics_05["f1"],
                "tuned_threshold": tuned_threshold,
                "tuned_val_f1": tuned_val_metrics["f1"],
                "tuned_val_sensitivity": tuned_val_metrics["sensitivity"],
                "tuned_val_specificity": tuned_val_metrics["specificity"],
            }
            history.append(row)
            pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

            improved = val_auroc > best_auroc + 1e-8
            if improved:
                best_auroc = val_auroc
                best_threshold = tuned_threshold
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                best_state = build_checkpoint(
                    model, optimizer, scaler, epoch + 1, 0, global_step,
                    best_auroc, best_threshold, best_epoch, epochs_without_improvement,
                    history, get_rng_state(), input_channels, cfg,
                )
                atomic_torch_save(best_state, ckpt_dir / "best.pt")
            else:
                epochs_without_improvement += 1

            last_state = build_checkpoint(
                model, optimizer, scaler, epoch + 1, 0, global_step,
                best_auroc, best_threshold, best_epoch, epochs_without_improvement,
                history, get_rng_state(), input_channels, cfg,
            )
            atomic_torch_save(last_state, last_path)
            save_json(
                {
                    "best_epoch": best_epoch,
                    "best_val_auroc": best_auroc,
                    "best_validation_threshold": best_threshold,
                    "threshold_objective": threshold_objective,
                },
                output_dir / "best_model_summary.json",
            )

            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_AUROC={val_auroc*100:.2f}% tuned_thr={tuned_threshold:.3f} "
                f"val_F1(tuned)={tuned_val_metrics['f1']*100:.2f}%"
            )
            if improved:
                print(f"  -> New best checkpoint (val AUROC {best_auroc*100:.2f}%).")

            if patience > 0 and epochs_without_improvement >= patience:
                print(f"Early stopping after {epochs_without_improvement} epochs without AUROC improvement.")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. The latest completed optimizer-step/epoch checkpoint is in checkpoints/last.pt.")
        return

    print(f"\nTraining complete. Best epoch: {best_epoch}, val AUROC: {best_auroc*100:.2f}%")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")
    print("Next: python evaluate.py --config config.yaml")


if __name__ == "__main__":
    main()
