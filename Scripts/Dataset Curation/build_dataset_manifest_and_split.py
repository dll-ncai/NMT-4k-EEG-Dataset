from pathlib import Path, PureWindowsPath
import pandas as pd
import numpy as np
import re
import math
import hashlib

# ============================================================
# PATHS
# ============================================================

DATASET_FOLDER = Path(r"E:\EEG Dataset\EEG\Dataset")

RECORDINGS_TSV = DATASET_FOLDER / "recordings.tsv"

CSV_FOLDER = DATASET_FOLDER / "csv"

EDF_NORMAL_FOLDER = DATASET_FOLDER / "edf" / "Normal"
EDF_ABNORMAL_FOLDER = DATASET_FOLDER / "edf" / "Abnormal"

REPORT_NORMAL_FOLDER = DATASET_FOLDER / "Reports" / "Normal"
REPORT_ABNORMAL_FOLDER = DATASET_FOLDER / "Reports" / "Abnormal"

OUTPUT_TSV = DATASET_FOLDER / "recordings_updated_with_splits.tsv"

MISSING_FILES_TSV = DATASET_FOLDER / "missing_required_files.tsv"
UNUSED_RECORDS_TSV = DATASET_FOLDER / "unused_records_not_selected.tsv"
SPLIT_SUMMARY_TSV = DATASET_FOLDER / "split_summary.tsv"
YEAR_COVERAGE_TSV = DATASET_FOLDER / "split_year_coverage.tsv"
DUPLICATES_TSV = DATASET_FOLDER / "duplicate_file_names_in_recordings.tsv"

# ============================================================
# TARGET SPLIT COUNTS
# ============================================================

TARGETS = {
    "Normal": {
        "train": 2787,
        "evaluation": 540
    },
    "Abnormal": {
        "train": 686,
        "evaluation": 460
    }
}

RANDOM_SEED = 42

# Normal needs EDF and report.
# Abnormal needs EDF, CSV annotation, and report.
REQUIRE_CSV_FOR_NORMAL = False
REQUIRE_CSV_FOR_ABNORMAL = True

# ============================================================
# HELPERS
# ============================================================

SOURCE_YEAR_RE = re.compile(r"^(ffh|mh)_(\d{4})_", re.IGNORECASE)


def clean_stem(value):
    value = str(value).strip().strip('"').strip("'")
    name = PureWindowsPath(value).name
    return Path(name).stem.lower()


def normalize_label(value):
    value = str(value).strip().lower()

    if "abnormal" in value:
        return "Abnormal"

    if "normal" in value:
        return "Normal"

    return ""


def parse_source_year(file_stem):
    match = SOURCE_YEAR_RE.match(file_stem)

    if not match:
        return "unknown", "unknown"

    source = match.group(1).lower()
    year = match.group(2)

    return source, year


def safe_number(value):
    value = str(value).strip()

    if value == "":
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


def make_age(years_value, months_value):
    years = safe_number(years_value)
    months = safe_number(months_value)

    if np.isnan(years) and np.isnan(months):
        return ""

    if np.isnan(years):
        years = 0

    if np.isnan(months):
        months = 0

    age = years + months / 12
    return round(age, 2)


def stable_seed(text):
    text = str(text)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return RANDOM_SEED + int(digest[:8], 16)


def index_files(folder, extensions):
    indexed = {}

    if not folder.exists():
        print(f"WARNING: Folder not found: {folder}")
        return indexed

    extensions = {ext.lower() for ext in extensions}

    for file_path in folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            stem = clean_stem(file_path.name)

            if stem not in indexed:
                indexed[stem] = file_path

    return indexed


def bounded_allocate(weights, total, lower_bounds, upper_bounds):
    """
    Allocate an exact total across groups using weights,
    while respecting lower and upper bounds.
    """

    keys = list(weights.keys())

    allocation = {
        key: int(lower_bounds.get(key, 0))
        for key in keys
    }

    upper_bounds = {
        key: int(upper_bounds.get(key, 0))
        for key in keys
    }

    lower_sum = sum(allocation.values())
    upper_sum = sum(upper_bounds.values())

    if lower_sum > total:
        raise ValueError(
            f"Cannot allocate {total}. Minimum required is {lower_sum}."
        )

    if upper_sum < total:
        raise ValueError(
            f"Cannot allocate {total}. Maximum possible is {upper_sum}."
        )

    remaining = total - lower_sum

    while remaining > 0:
        active_keys = [
            key for key in keys
            if allocation[key] < upper_bounds[key]
        ]

        if not active_keys:
            raise ValueError("No active groups left for allocation.")

        active_weight_sum = sum(max(float(weights[key]), 0.0) for key in active_keys)

        if active_weight_sum == 0:
            active_weight_sum = len(active_keys)
            active_weights = {key: 1.0 for key in active_keys}
        else:
            active_weights = {key: max(float(weights[key]), 0.0) for key in active_keys}

        ideal_extra = {
            key: remaining * active_weights[key] / active_weight_sum
            for key in active_keys
        }

        floor_extra = {}
        added_now = 0

        for key in active_keys:
            capacity = upper_bounds[key] - allocation[key]
            amount = min(math.floor(ideal_extra[key]), capacity)
            floor_extra[key] = amount
            added_now += amount

        if added_now > 0:
            for key, amount in floor_extra.items():
                allocation[key] += amount

            remaining -= added_now
        else:
            ranked = sorted(
                active_keys,
                key=lambda key: (
                    ideal_extra[key] - math.floor(ideal_extra[key]),
                    active_weights[key],
                    str(key)
                ),
                reverse=True
            )

            for key in ranked:
                if remaining == 0:
                    break

                if allocation[key] < upper_bounds[key]:
                    allocation[key] += 1
                    remaining -= 1

    return allocation


def assign_one_label(label_df, label, train_target, eval_target):
    total_target = train_target + eval_target

    if len(label_df) < total_target:
        raise ValueError(
            f"Not enough valid {label} recordings. "
            f"Required {total_target}, found {len(label_df)}."
        )

    label_df = label_df.copy()

    group_counts = label_df.groupby("__stratum").size().to_dict()
    strata = list(group_counts.keys())

    if total_target >= len(strata):
        selected_lower = {key: 1 for key in strata}
    else:
        selected_lower = {key: 0 for key in strata}

    selected_upper = group_counts.copy()

    selected_counts = bounded_allocate(
        weights=group_counts,
        total=total_target,
        lower_bounds=selected_lower,
        upper_bounds=selected_upper
    )

    selected_parts = []

    for stratum, count in selected_counts.items():
        part = label_df[label_df["__stratum"] == stratum]

        sampled = part.sample(
            n=count,
            random_state=stable_seed(f"select_{label}_{stratum}")
        )

        selected_parts.append(sampled)

    selected = pd.concat(selected_parts, ignore_index=False).copy()

    selected_group_counts = selected.groupby("__stratum").size().to_dict()
    selected_strata = list(selected_group_counts.keys())

    # Try to place each source and year in both train and evaluation.
    # This is only possible when a stratum has at least 2 selected files.
    eval_lower = {}
    eval_upper = {}

    for stratum, count in selected_group_counts.items():
        if count >= 2:
            eval_lower[stratum] = 1
            eval_upper[stratum] = count - 1
        else:
            eval_lower[stratum] = 0
            eval_upper[stratum] = 0

    try:
        eval_counts = bounded_allocate(
            weights=selected_group_counts,
            total=eval_target,
            lower_bounds=eval_lower,
            upper_bounds=eval_upper
        )
    except ValueError:
        print(
            f"WARNING: Full year coverage in both splits was not possible for {label}. "
            f"Using relaxed evaluation allocation."
        )

        eval_lower = {key: 0 for key in selected_strata}
        eval_upper = selected_group_counts.copy()

        eval_counts = bounded_allocate(
            weights=selected_group_counts,
            total=eval_target,
            lower_bounds=eval_lower,
            upper_bounds=eval_upper
        )

    selected["split"] = "train"

    for stratum, count in eval_counts.items():
        if count == 0:
            continue

        part = selected[selected["__stratum"] == stratum]

        eval_indices = part.sample(
            n=count,
            random_state=stable_seed(f"eval_{label}_{stratum}")
        ).index

        selected.loc[eval_indices, "split"] = "evaluation"

    actual_train = (selected["split"] == "train").sum()
    actual_eval = (selected["split"] == "evaluation").sum()

    if actual_train != train_target or actual_eval != eval_target:
        raise ValueError(
            f"Split count mismatch for {label}. "
            f"Train expected {train_target}, got {actual_train}. "
            f"Evaluation expected {eval_target}, got {actual_eval}."
        )

    return selected


# ============================================================
# LOAD FILE INDEXES
# ============================================================

print("Indexing files...")

edf_normal_index = index_files(EDF_NORMAL_FOLDER, {".edf"})
edf_abnormal_index = index_files(EDF_ABNORMAL_FOLDER, {".edf"})

csv_index = index_files(CSV_FOLDER, {".csv"})

report_normal_index = index_files(REPORT_NORMAL_FOLDER, {".txt", ".pdf", ".doc", ".docx"})
report_abnormal_index = index_files(REPORT_ABNORMAL_FOLDER, {".txt", ".pdf", ".doc", ".docx"})

print(f"Normal EDF files found: {len(edf_normal_index)}")
print(f"Abnormal EDF files found: {len(edf_abnormal_index)}")
print(f"CSV annotation files found: {len(csv_index)}")
print(f"Normal report files found: {len(report_normal_index)}")
print(f"Abnormal report files found: {len(report_abnormal_index)}")

# ============================================================
# READ ORIGINAL RECORDINGS TSV
# ============================================================

df = pd.read_csv(RECORDINGS_TSV, sep="\t", dtype=str, keep_default_na=False)
df.columns = [col.strip() for col in df.columns]

required_columns = [
    "File Name",
    "Date",
    "Gender",
    "Age (Years)",
    "Age (Months)",
    "Label (Normal/Abnormal)"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns in recordings.tsv: {missing_columns}")

df["__stem"] = df["File Name"].apply(clean_stem)
df["Label (Normal/Abnormal)"] = df["Label (Normal/Abnormal)"].apply(normalize_label)

df[["source", "year"]] = df["__stem"].apply(
    lambda x: pd.Series(parse_source_year(x))
)

df["age"] = df.apply(
    lambda row: make_age(row["Age (Years)"], row["Age (Months)"]),
    axis=1
)

df["gender"] = df["Gender"]
df["date"] = df["Date"]

# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicates = df[df.duplicated("__stem", keep=False)].copy()

if not duplicates.empty:
    duplicates.to_csv(DUPLICATES_TSV, sep="\t", index=False)
    print(f"Duplicate file names saved to: {DUPLICATES_TSV}")

# Keep one row per recording
df = df.drop_duplicates("__stem", keep="first").copy()

# ============================================================
# CHECK REQUIRED FILES
# ============================================================

edf_paths = []
csv_paths = []
report_paths = []
missing_reasons = []

for _, row in df.iterrows():
    stem = row["__stem"]
    label = row["Label (Normal/Abnormal)"]

    missing = []

    edf_path = ""
    csv_path = ""
    report_path = ""

    if label == "Normal":
        edf_path = edf_normal_index.get(stem, "")
        report_path = report_normal_index.get(stem, "")
        csv_path = csv_index.get(stem, "")

        if not edf_path:
            missing.append("missing normal edf")

        if not report_path:
            missing.append("missing normal report")

        if REQUIRE_CSV_FOR_NORMAL and not csv_path:
            missing.append("missing normal csv annotation")

    elif label == "Abnormal":
        edf_path = edf_abnormal_index.get(stem, "")
        report_path = report_abnormal_index.get(stem, "")
        csv_path = csv_index.get(stem, "")

        if not edf_path:
            missing.append("missing abnormal edf")

        if not report_path:
            missing.append("missing abnormal report")

        if REQUIRE_CSV_FOR_ABNORMAL and not csv_path:
            missing.append("missing abnormal csv annotation")

    else:
        missing.append("invalid or missing label")

    edf_paths.append(str(edf_path) if edf_path else "")
    csv_paths.append(str(csv_path) if csv_path else "")
    report_paths.append(str(report_path) if report_path else "")
    missing_reasons.append(", ".join(missing))

df["edf_path"] = edf_paths
df["annotation_csv_path"] = csv_paths
df["report_path"] = report_paths
df["missing_reason"] = missing_reasons

missing_df = df[df["missing_reason"] != ""].copy()
valid_df = df[df["missing_reason"] == ""].copy()

if not missing_df.empty:
    missing_df.to_csv(MISSING_FILES_TSV, sep="\t", index=False)
    print(f"Missing required files saved to: {MISSING_FILES_TSV}")

print(f"Valid recordings after file check: {len(valid_df)}")

# ============================================================
# CREATE STRATUM FOR BALANCED SOURCE YEAR SPLIT
# ============================================================

valid_df["__stratum"] = (
    valid_df["Label (Normal/Abnormal)"].astype(str)
    + "_"
    + valid_df["source"].astype(str)
    + "_"
    + valid_df["year"].astype(str)
)

# ============================================================
# ASSIGN SPLITS
# ============================================================

selected_parts = []

for label, counts in TARGETS.items():
    label_df = valid_df[valid_df["Label (Normal/Abnormal)"] == label].copy()

    selected_label_df = assign_one_label(
        label_df=label_df,
        label=label,
        train_target=counts["train"],
        eval_target=counts["evaluation"]
    )

    selected_parts.append(selected_label_df)

selected_df = pd.concat(selected_parts, ignore_index=True).copy()

selected_stems = set(selected_df["__stem"])
unused_df = valid_df[~valid_df["__stem"].isin(selected_stems)].copy()

if not unused_df.empty:
    unused_df.to_csv(UNUSED_RECORDS_TSV, sep="\t", index=False)
    print(f"Unused valid records saved to: {UNUSED_RECORDS_TSV}")

# ============================================================
# FINAL OUTPUT COLUMNS
# ============================================================

output_columns = [
    "File Name",
    "split",
    "age",
    "gender",
    "date",
    "Age (Years)",
    "Age (Months)",
    "Label (Normal/Abnormal)",
    "source",
    "year",
    "edf_path",
    "annotation_csv_path",
    "report_path"
]

selected_df = selected_df[output_columns].copy()

selected_df = selected_df.sort_values(
    by=["split", "Label (Normal/Abnormal)", "source", "year", "File Name"]
).reset_index(drop=True)

selected_df.to_csv(OUTPUT_TSV, sep="\t", index=False)

# ============================================================
# SUMMARY FILES
# ============================================================

summary = (
    selected_df
    .groupby(["split", "Label (Normal/Abnormal)"])
    .size()
    .reset_index(name="count")
)

summary.to_csv(SPLIT_SUMMARY_TSV, sep="\t", index=False)

coverage = (
    selected_df
    .groupby(["Label (Normal/Abnormal)", "source", "year", "split"])
    .size()
    .reset_index(name="count")
)

coverage_pivot = coverage.pivot_table(
    index=["Label (Normal/Abnormal)", "source", "year"],
    columns="split",
    values="count",
    fill_value=0
).reset_index()

coverage_pivot.to_csv(YEAR_COVERAGE_TSV, sep="\t", index=False)

# ============================================================
# PRINT FINAL RESULT
# ============================================================

print("\nDone.")
print(f"Updated recordings TSV saved to: {OUTPUT_TSV}")
print(f"Split summary saved to: {SPLIT_SUMMARY_TSV}")
print(f"Year coverage saved to: {YEAR_COVERAGE_TSV}")

print("\nFinal split counts:")
print(summary)

print("\nYear coverage preview:")
print(coverage_pivot.head(30))