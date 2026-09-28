from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, dataset_path, project_path


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    metadata_path = dataset_path(cfg, cfg["data"]["metadata_file"])
    df = norm_cols(pd.read_csv(metadata_path, sep="\t"))
    required = {"file_name", "split", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}. Found: {list(df.columns)}")

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    bad_labels = set(df["label"].unique()) - {"normal", "abnormal"}
    if bad_labels:
        raise ValueError(f"Unexpected labels: {bad_labels}")

    train_df = df.loc[df["split"].eq("train")].copy()
    eval_df = df.loc[df["split"].eq("evaluation")].copy()
    if train_df.empty or eval_df.empty:
        raise ValueError(f"Expected both train and evaluation splits. Counts: {df['split'].value_counts().to_dict()}")

    fit_idx, val_idx = train_test_split(
        train_df.index,
        test_size=float(cfg["data"]["validation_fraction"]),
        random_state=int(cfg["data"]["seed"]),
        stratify=train_df["label"],
    )

    role = pd.Series(index=df.index, dtype="object")
    role.loc[fit_idx] = "fit"
    role.loc[val_idx] = "validation"
    role.loc[eval_df.index] = "evaluation"

    root = Path(cfg["data"]["dataset_root"])
    rows = []
    missing_paths = []
    for idx, row in df.iterrows():
        rid = str(row["file_name"]).strip()
        fname = rid if rid.lower().endswith(".edf") else rid + ".edf"
        edf_path = root / row["split"] / row["label"] / "edf" / fname
        if not edf_path.exists():
            missing_paths.append(str(edf_path))
        rows.append({
            "record_id": rid[:-4] if rid.lower().endswith(".edf") else rid,
            "dataset_split": row["split"],
            "role": role.loc[idx],
            "label": row["label"],
            "y": 1 if row["label"] == "abnormal" else 0,
            "edf_path": str(edf_path),
        })

    manifest = pd.DataFrame(rows)
    out_path = project_path(cfg, cfg["data"]["manifest_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)

    print(f"Saved manifest: {out_path}")
    print("\nRole x label counts:")
    print(pd.crosstab(manifest["role"], manifest["label"], margins=True))
    print(f"\nMissing EDF paths: {len(missing_paths)}")
    if missing_paths:
        for p in missing_paths[:20]:
            print("  ", p)
        raise SystemExit("Some EDF files were not found. Fix paths before preprocessing.")


if __name__ == "__main__":
    main()
