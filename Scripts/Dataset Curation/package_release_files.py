from pathlib import Path, PureWindowsPath
import pandas as pd
import shutil

# ============================================================
# MAIN PATHS
# ============================================================

DATASET_ROOT = Path(r"E:\EEG Dataset\EEG\Dataset")

NMT_ROOT = DATASET_ROOT / "NMT-4K-EEG"
METADATA_DIR = NMT_ROOT / "metadata"

RECORDINGS_TSV = METADATA_DIR / "recordings.tsv"

# ============================================================
# SOURCE FOLDERS
# Change these if your real source folders are different
# ============================================================

SOURCE_NORMAL_EDF_DIR = DATASET_ROOT / "edf" / "Normal"
SOURCE_ABNORMAL_EDF_DIR = DATASET_ROOT / "edf" / "Abnormal"

SOURCE_NORMAL_REPORT_DIR = DATASET_ROOT / "Reports" / "Normal"
SOURCE_ABNORMAL_REPORT_DIR = DATASET_ROOT / "Reports" / "Abnormal"

SOURCE_ANNOTATION_DIR = DATASET_ROOT / "csv"

# ============================================================
# DESTINATION FOLDERS
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

# ============================================================
# OUTPUT LOG FILES
# ============================================================

COPY_LOG_TSV = METADATA_DIR / "copy_log.tsv"
MISSING_FILES_TSV = METADATA_DIR / "missing_files_during_copy.tsv"
SUMMARY_TSV = METADATA_DIR / "copy_summary.tsv"
DUPLICATE_SOURCE_FILES_TSV = METADATA_DIR / "duplicate_source_files.tsv"

# ============================================================
# SETTINGS
# ============================================================

OVERWRITE_EXISTING = False

EDF_EXTENSIONS = {".edf"}
ANNOTATION_EXTENSIONS = {".csv"}
REPORT_EXTENSIONS = {".txt", ".pdf", ".doc", ".docx", ".rtf"}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_stem(value):
    """
    Converts file name to lowercase stem.
    Example:
    ffh_2023_0000005.edf -> ffh_2023_0000005
    ffh_2023_0000005 -> ffh_2023_0000005
    """
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


def index_files(folder, extensions):
    """
    Recursively index files by stem.
    Returns:
    indexed = {stem: file_path}
    duplicates = list of duplicate records
    """
    indexed = {}
    duplicates = []

    if not folder.exists():
        print(f"WARNING: Source folder not found: {folder}")
        return indexed, duplicates

    extensions = {ext.lower() for ext in extensions}

    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in extensions:
            continue

        stem = clean_stem(file_path.name)

        if stem in indexed:
            duplicates.append({
                "stem": stem,
                "first_path": str(indexed[stem]),
                "duplicate_path": str(file_path)
            })
        else:
            indexed[stem] = file_path

    return indexed, duplicates


def copy_file(src, dst_folder):
    dst_folder.mkdir(parents=True, exist_ok=True)

    dst = dst_folder / src.name

    if dst.exists() and not OVERWRITE_EXISTING:
        return "already_exists", dst

    shutil.copy2(src, dst)

    return "copied", dst


# ============================================================
# CHECK RECORDINGS FILE
# ============================================================

if not RECORDINGS_TSV.exists():
    raise FileNotFoundError(f"recordings.tsv not found here: {RECORDINGS_TSV}")

# ============================================================
# CREATE DESTINATION FOLDERS
# ============================================================

for folder in DESTINATIONS.values():
    folder.mkdir(parents=True, exist_ok=True)

METADATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD RECORDINGS TSV
# ============================================================

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

# Remove duplicate rows by File Name
df = df.drop_duplicates(subset=["__stem"], keep="first").copy()

# Keep only valid train and evaluation records
df = df[df["split"].isin(["train", "evaluation"])].copy()

# Keep only valid labels
df = df[df["Label (Normal/Abnormal)"].isin(["Normal", "Abnormal"])].copy()

# ============================================================
# CHECK EXPECTED RECORDING COUNTS
# ============================================================

print("\nChecking expected recording counts from recordings.tsv...\n")

actual_counts = (
    df.groupby(["split", "Label (Normal/Abnormal)"])
    .size()
    .reset_index(name="recording_count")
)

print(actual_counts)

for key, expected_count in EXPECTED_RECORDING_COUNTS.items():
    split, label = key

    actual_count = len(
        df[
            (df["split"] == split)
            & (df["Label (Normal/Abnormal)"] == label)
        ]
    )

    if actual_count != expected_count:
        print(
            f"WARNING: Expected {expected_count} records for {split} {label}, "
            f"but found {actual_count} in recordings.tsv."
        )

# ============================================================
# INDEX SOURCE FILES
# ============================================================

print("\nIndexing source files...\n")

normal_edf_index, dup_normal_edf = index_files(SOURCE_NORMAL_EDF_DIR, EDF_EXTENSIONS)
abnormal_edf_index, dup_abnormal_edf = index_files(SOURCE_ABNORMAL_EDF_DIR, EDF_EXTENSIONS)

normal_report_index, dup_normal_report = index_files(SOURCE_NORMAL_REPORT_DIR, REPORT_EXTENSIONS)
abnormal_report_index, dup_abnormal_report = index_files(SOURCE_ABNORMAL_REPORT_DIR, REPORT_EXTENSIONS)

annotation_index, dup_annotations = index_files(SOURCE_ANNOTATION_DIR, ANNOTATION_EXTENSIONS)

all_duplicates = (
    dup_normal_edf
    + dup_abnormal_edf
    + dup_normal_report
    + dup_abnormal_report
    + dup_annotations
)

if all_duplicates:
    pd.DataFrame(all_duplicates).to_csv(DUPLICATE_SOURCE_FILES_TSV, sep="\t", index=False)
    print(f"Duplicate source file report saved to: {DUPLICATE_SOURCE_FILES_TSV}")

print(f"Normal EDF files indexed: {len(normal_edf_index)}")
print(f"Abnormal EDF files indexed: {len(abnormal_edf_index)}")
print(f"Normal reports indexed: {len(normal_report_index)}")
print(f"Abnormal reports indexed: {len(abnormal_report_index)}")
print(f"CSV annotations indexed: {len(annotation_index)}")

# ============================================================
# COPY FILES
# ============================================================

copy_logs = []
missing_logs = []

print("\nStarting file copy...\n")

for _, row in df.iterrows():
    file_name = row["File Name"]
    stem = row["__stem"]
    split = row["split"]
    label = row["Label (Normal/Abnormal)"]

    required_items = []

    if label == "Normal":
        required_items.append(("edf", normal_edf_index.get(stem)))
        required_items.append(("report", normal_report_index.get(stem)))

    elif label == "Abnormal":
        required_items.append(("edf", abnormal_edf_index.get(stem)))
        required_items.append(("annotation", annotation_index.get(stem)))
        required_items.append(("report", abnormal_report_index.get(stem)))

    for file_type, src_path in required_items:
        destination_key = (split, label, file_type)
        destination_folder = DESTINATIONS[destination_key]

        if src_path is None:
            missing_logs.append({
                "File Name": file_name,
                "stem": stem,
                "split": split,
                "label": label,
                "missing_file_type": file_type,
                "expected_destination": str(destination_folder)
            })

            copy_logs.append({
                "File Name": file_name,
                "stem": stem,
                "split": split,
                "label": label,
                "file_type": file_type,
                "source_path": "",
                "destination_path": "",
                "status": "missing"
            })

            continue

        status, dst_path = copy_file(src_path, destination_folder)

        copy_logs.append({
            "File Name": file_name,
            "stem": stem,
            "split": split,
            "label": label,
            "file_type": file_type,
            "source_path": str(src_path),
            "destination_path": str(dst_path),
            "status": status
        })

# ============================================================
# SAVE LOGS
# ============================================================

copy_log_df = pd.DataFrame(copy_logs)
missing_df = pd.DataFrame(missing_logs)

copy_log_df.to_csv(COPY_LOG_TSV, sep="\t", index=False)

if not missing_df.empty:
    missing_df.to_csv(MISSING_FILES_TSV, sep="\t", index=False)

# ============================================================
# CREATE SUMMARY
# ============================================================

summary = (
    copy_log_df
    .groupby(["split", "label", "file_type", "status"])
    .size()
    .reset_index(name="count")
)

summary.to_csv(SUMMARY_TSV, sep="\t", index=False)

print("\nCopy process finished.")
print(f"Copy log saved to: {COPY_LOG_TSV}")
print(f"Summary saved to: {SUMMARY_TSV}")

if not missing_df.empty:
    print(f"Missing file report saved to: {MISSING_FILES_TSV}")
else:
    print("No missing files found.")

print("\nCopy summary:")
print(summary)

# ============================================================
# FINAL STRICT VALIDATION BY DESTINATION FOLDER
# ============================================================

print("\nFinal destination folder counts:\n")

folder_checks = [
    ("train abnormal annotations", NMT_ROOT / "train" / "abnormal" / "annotations", ANNOTATION_EXTENSIONS, 686),
    ("train abnormal edf", NMT_ROOT / "train" / "abnormal" / "edf", EDF_EXTENSIONS, 686),
    ("train abnormal reports", NMT_ROOT / "train" / "abnormal" / "reports", REPORT_EXTENSIONS, 686),

    ("train normal edf", NMT_ROOT / "train" / "normal" / "edf", EDF_EXTENSIONS, 2787),
    ("train normal reports", NMT_ROOT / "train" / "normal" / "reports", REPORT_EXTENSIONS, 2787),

    ("evaluation abnormal annotations", NMT_ROOT / "evaluation" / "abnormal" / "annotations", ANNOTATION_EXTENSIONS, 460),
    ("evaluation abnormal edf", NMT_ROOT / "evaluation" / "abnormal" / "edf", EDF_EXTENSIONS, 460),
    ("evaluation abnormal reports", NMT_ROOT / "evaluation" / "abnormal" / "reports", REPORT_EXTENSIONS, 460),

    ("evaluation normal edf", NMT_ROOT / "evaluation" / "normal" / "edf", EDF_EXTENSIONS, 540),
    ("evaluation normal reports", NMT_ROOT / "evaluation" / "normal" / "reports", REPORT_EXTENSIONS, 540),
]

validation_rows = []

for name, folder, extensions, expected in folder_checks:
    count = 0

    if folder.exists():
        count = sum(
            1
            for file_path in folder.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in extensions
        )

    status = "OK" if count == expected else "CHECK"

    validation_rows.append({
        "folder_check": name,
        "folder": str(folder),
        "expected_count": expected,
        "actual_count": count,
        "status": status
    })

    print(f"{name}: expected={expected}, actual={count}, status={status}")

validation_df = pd.DataFrame(validation_rows)
validation_df.to_csv(METADATA_DIR / "destination_folder_validation.tsv", sep="\t", index=False)

print("\nValidation file saved to:")
print(METADATA_DIR / "destination_folder_validation.tsv")