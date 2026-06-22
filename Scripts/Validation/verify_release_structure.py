from pathlib import Path, PureWindowsPath
import pandas as pd
from collections import defaultdict

# ============================================================
# MAIN PATHS
# ============================================================

NMT_ROOT = Path(r"E:\EEG Dataset\EEG\Dataset\NMT-4K-EEG")
METADATA_DIR = NMT_ROOT / "metadata"
RECORDINGS_TSV = METADATA_DIR / "recordings.tsv"

# ============================================================
# DESTINATION FOLDERS TO VERIFY
# ============================================================

DESTINATIONS = {
    ("train", "Abnormal", "annotation"): NMT_ROOT / "train" / "abnormal" / "annotations",
    ("train", "Abnormal", "edf"): NMT_ROOT / "train" / "abnormal" / "edf",
    ("train", "Abnormal", "report"): NMT_ROOT / "train" / "abnormal" / "reports",

    ("train", "Normal", "edf"): NMT_ROOT / "train" / "normal" / "edf",
    ("train", "Normal", "report"): NMT_ROOT / "train" / "normal" / "reports",

    ("evaluation", "Abnormal", "annotation"): NMT_ROOT / "evaluation" / "abnormal" / "annotations",
    ("evaluation", "Abnormal", "edf"): NMT_ROOT / "evaluation" / "abnormal" / "edf",
    ("evaluation", "Abnormal", "report"): NMT_ROOT / "evaluation" / "abnormal" / "reports",

    ("evaluation", "Normal", "edf"): NMT_ROOT / "evaluation" / "normal" / "edf",
    ("evaluation", "Normal", "report"): NMT_ROOT / "evaluation" / "normal" / "reports",
}

# ============================================================
# EXPECTED COUNTS
# ============================================================

EXPECTED_RECORDING_COUNTS = {
    ("train", "Normal"): 2787,
    ("train", "Abnormal"): 686,
    ("evaluation", "Normal"): 540,
    ("evaluation", "Abnormal"): 460,
}

EXPECTED_FILE_COUNTS = {
    ("train", "Abnormal", "annotation"): 686,
    ("train", "Abnormal", "edf"): 686,
    ("train", "Abnormal", "report"): 686,

    ("train", "Normal", "edf"): 2787,
    ("train", "Normal", "report"): 2787,

    ("evaluation", "Abnormal", "annotation"): 460,
    ("evaluation", "Abnormal", "edf"): 460,
    ("evaluation", "Abnormal", "report"): 460,

    ("evaluation", "Normal", "edf"): 540,
    ("evaluation", "Normal", "report"): 540,
}

# ============================================================
# EXTENSIONS
# ============================================================

EXTENSIONS = {
    "edf": {".edf"},
    "annotation": {".csv"},
    "report": {".txt", ".pdf", ".doc", ".docx", ".rtf"},
}

# ============================================================
# OUTPUT REPORTS
# ============================================================

MATCHED_TSV = METADATA_DIR / "verification_matched_files.tsv"
MISSING_TSV = METADATA_DIR / "verification_missing_files.tsv"
EXTRA_TSV = METADATA_DIR / "verification_extra_files.tsv"
DUPLICATES_TSV = METADATA_DIR / "verification_duplicate_files.tsv"
SUMMARY_TSV = METADATA_DIR / "verification_summary.tsv"
RECORDING_COUNT_TSV = METADATA_DIR / "verification_recording_counts.tsv"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_stem(value):
    value = str(value).strip().strip('"').strip("'")
    name = PureWindowsPath(value).name
    return Path(name).stem.lower()


def normalize_split(value):
    value = str(value).strip().lower()

    if value in {"train", "training"}:
        return "train"

    if value in {"evaluation", "eval", "test", "testing"}:
        return "evaluation"

    return value


def normalize_label(value):
    value = str(value).strip().lower()

    if "abnormal" in value:
        return "Abnormal"

    if "normal" in value:
        return "Normal"

    return ""


def scan_folder(folder, allowed_extensions):
    """
    Scans a destination folder and returns:
    files_by_stem = {stem: [paths]}
    """
    files_by_stem = defaultdict(list)

    if not folder.exists():
        return files_by_stem

    allowed_extensions = {ext.lower() for ext in allowed_extensions}

    for file_path in folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
            stem = clean_stem(file_path.name)
            files_by_stem[stem].append(file_path)

    return files_by_stem


# ============================================================
# LOAD RECORDINGS TSV
# ============================================================

if not RECORDINGS_TSV.exists():
    raise FileNotFoundError(f"recordings.tsv not found: {RECORDINGS_TSV}")

df = pd.read_csv(RECORDINGS_TSV, sep="\t", dtype=str, keep_default_na=False)
df.columns = [col.strip() for col in df.columns]

required_columns = [
    "File Name",
    "split",
    "Label (Normal/Abnormal)"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns in recordings.tsv: {missing_columns}")

df["__stem"] = df["File Name"].apply(clean_stem)
df["split"] = df["split"].apply(normalize_split)
df["Label (Normal/Abnormal)"] = df["Label (Normal/Abnormal)"].apply(normalize_label)

df = df[df["split"].isin(["train", "evaluation"])].copy()
df = df[df["Label (Normal/Abnormal)"].isin(["Normal", "Abnormal"])].copy()

# Remove duplicate File Name rows from TSV for verification
duplicate_tsv_rows = df[df.duplicated("__stem", keep=False)].copy()

if not duplicate_tsv_rows.empty:
    print("WARNING: Duplicate File Name rows found in recordings.tsv.")
    duplicate_tsv_rows.to_csv(METADATA_DIR / "verification_duplicate_rows_in_recordings.tsv", sep="\t", index=False)

df = df.drop_duplicates("__stem", keep="first").copy()

# ============================================================
# VERIFY RECORDING COUNTS FROM TSV
# ============================================================

recording_count_rows = []

for (split, label), expected_count in EXPECTED_RECORDING_COUNTS.items():
    actual_count = len(
        df[
            (df["split"] == split)
            & (df["Label (Normal/Abnormal)"] == label)
        ]
    )

    status = "OK" if actual_count == expected_count else "CHECK"

    recording_count_rows.append({
        "split": split,
        "label": label,
        "expected_recordings": expected_count,
        "actual_recordings_in_tsv": actual_count,
        "status": status
    })

recording_count_df = pd.DataFrame(recording_count_rows)
recording_count_df.to_csv(RECORDING_COUNT_TSV, sep="\t", index=False)

# ============================================================
# BUILD EXPECTED FILE LIST FROM TSV
# ============================================================

expected_items = []

for _, row in df.iterrows():
    file_name = row["File Name"]
    stem = row["__stem"]
    split = row["split"]
    label = row["Label (Normal/Abnormal)"]

    if label == "Normal":
        expected_items.append({
            "File Name": file_name,
            "stem": stem,
            "split": split,
            "label": label,
            "file_type": "edf"
        })
        expected_items.append({
            "File Name": file_name,
            "stem": stem,
            "split": split,
            "label": label,
            "file_type": "report"
        })

    elif label == "Abnormal":
        expected_items.append({
            "File Name": file_name,
            "stem": stem,
            "split": split,
            "label": label,
            "file_type": "edf"
        })
        expected_items.append({
            "File Name": file_name,
            "stem": stem,
            "split": split,
            "label": label,
            "file_type": "annotation"
        })
        expected_items.append({
            "File Name": file_name,
            "stem": stem,
            "split": split,
            "label": label,
            "file_type": "report"
        })

expected_df = pd.DataFrame(expected_items)

# ============================================================
# SCAN ALL DESTINATION FOLDERS
# ============================================================

actual_folder_files = {}

for key, folder in DESTINATIONS.items():
    split, label, file_type = key
    allowed_extensions = EXTENSIONS[file_type]
    actual_folder_files[key] = scan_folder(folder, allowed_extensions)

# ============================================================
# CHECK MISSING AND MATCHED FILES
# ============================================================

matched_rows = []
missing_rows = []

for _, row in expected_df.iterrows():
    split = row["split"]
    label = row["label"]
    file_type = row["file_type"]
    stem = row["stem"]

    key = (split, label, file_type)
    folder = DESTINATIONS[key]
    files_by_stem = actual_folder_files[key]

    matched_paths = files_by_stem.get(stem, [])

    if len(matched_paths) == 0:
        missing_rows.append({
            "File Name": row["File Name"],
            "stem": stem,
            "split": split,
            "label": label,
            "file_type": file_type,
            "expected_folder": str(folder),
            "status": "missing"
        })
    else:
        for path in matched_paths:
            matched_rows.append({
                "File Name": row["File Name"],
                "stem": stem,
                "split": split,
                "label": label,
                "file_type": file_type,
                "matched_path": str(path),
                "status": "matched"
            })

# ============================================================
# CHECK EXTRA FILES IN DESTINATION FOLDERS
# ============================================================

expected_sets = {}

for key in DESTINATIONS.keys():
    split, label, file_type = key

    stems = set(
        expected_df[
            (expected_df["split"] == split)
            & (expected_df["label"] == label)
            & (expected_df["file_type"] == file_type)
        ]["stem"]
    )

    expected_sets[key] = stems

extra_rows = []

for key, files_by_stem in actual_folder_files.items():
    split, label, file_type = key
    folder = DESTINATIONS[key]
    expected_stems = expected_sets[key]

    for stem, paths in files_by_stem.items():
        if stem not in expected_stems:
            for path in paths:
                extra_rows.append({
                    "stem": stem,
                    "split_folder": split,
                    "label_folder": label,
                    "file_type_folder": file_type,
                    "folder": str(folder),
                    "extra_file_path": str(path),
                    "status": "extra_or_wrong_folder"
                })

# ============================================================
# CHECK DUPLICATE FILE STEMS INSIDE DESTINATION FOLDERS
# ============================================================

duplicate_rows = []

for key, files_by_stem in actual_folder_files.items():
    split, label, file_type = key
    folder = DESTINATIONS[key]

    for stem, paths in files_by_stem.items():
        if len(paths) > 1:
            duplicate_rows.append({
                "stem": stem,
                "split": split,
                "label": label,
                "file_type": file_type,
                "folder": str(folder),
                "duplicate_count": len(paths),
                "paths": " | ".join(str(p) for p in paths)
            })

# ============================================================
# SUMMARY BY FOLDER
# ============================================================

summary_rows = []

for key, folder in DESTINATIONS.items():
    split, label, file_type = key

    expected_count = EXPECTED_FILE_COUNTS[key]

    files_by_stem = actual_folder_files[key]
    actual_count = sum(len(paths) for paths in files_by_stem.values())

    expected_stems = expected_sets[key]
    actual_stems = set(files_by_stem.keys())

    matched_stems = expected_stems & actual_stems
    missing_stems = expected_stems - actual_stems
    extra_stems = actual_stems - expected_stems

    status = "OK"

    if (
        actual_count != expected_count
        or len(missing_stems) > 0
        or len(extra_stems) > 0
    ):
        status = "CHECK"

    summary_rows.append({
        "split": split,
        "label": label,
        "file_type": file_type,
        "folder": str(folder),
        "expected_count": expected_count,
        "actual_file_count": actual_count,
        "matched_unique_stems": len(matched_stems),
        "missing_unique_stems": len(missing_stems),
        "extra_unique_stems": len(extra_stems),
        "status": status
    })

summary_df = pd.DataFrame(summary_rows)

# ============================================================
# SAVE REPORTS
# ============================================================

matched_df = pd.DataFrame(matched_rows)
missing_df = pd.DataFrame(missing_rows)
extra_df = pd.DataFrame(extra_rows)
duplicates_df = pd.DataFrame(duplicate_rows)

matched_df.to_csv(MATCHED_TSV, sep="\t", index=False)
missing_df.to_csv(MISSING_TSV, sep="\t", index=False)
extra_df.to_csv(EXTRA_TSV, sep="\t", index=False)
duplicates_df.to_csv(DUPLICATES_TSV, sep="\t", index=False)
summary_df.to_csv(SUMMARY_TSV, sep="\t", index=False)

# ============================================================
# PRINT RESULT
# ============================================================

print("\nVerification completed.\n")

print("Recording count check:")
print(recording_count_df)

print("\nFolder verification summary:")
print(summary_df)

print("\nReports saved here:")
print(f"Matched files: {MATCHED_TSV}")
print(f"Missing files: {MISSING_TSV}")
print(f"Extra or wrong folder files: {EXTRA_TSV}")
print(f"Duplicate files: {DUPLICATES_TSV}")
print(f"Folder summary: {SUMMARY_TSV}")
print(f"Recording counts: {RECORDING_COUNT_TSV}")

total_missing = len(missing_df)
total_extra = len(extra_df)
total_duplicates = len(duplicates_df)

print("\nFinal result:")
print(f"Missing files: {total_missing}")
print(f"Extra or wrong folder files: {total_extra}")
print(f"Duplicate files: {total_duplicates}")

if (
    total_missing == 0
    and total_extra == 0
    and total_duplicates == 0
    and all(summary_df["status"] == "OK")
    and all(recording_count_df["status"] == "OK")
):
    print("\nSUCCESS: All copied files match recordings.tsv, split, and label.")
else:
    print("\nCHECK REQUIRED: Some files are missing, extra, duplicated, or counts do not match.")