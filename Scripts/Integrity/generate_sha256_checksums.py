from pathlib import Path
import hashlib

# ============================================================
# DATASET ROOT
# ============================================================

NMT_ROOT = Path(r"E:\EEG Dataset\EEG\Dataset\NMT-4K-EEG")

# Save checksum file here
SHA256_FILE = NMT_ROOT / "metadata" / "sha256.txt"

# ============================================================
# SETTINGS
# ============================================================

# Files or folders that should not be included
EXCLUDE_NAMES = {
    "sha256.txt",
    ".git",
    "__pycache__",
    ".DS_Store",
    "Thumbs.db",
}

# Optional: exclude temporary verification/log files
EXCLUDE_PREFIXES = {
    "verification_",
    "copy_log",
    "missing_",
    "duplicate_",
    "destination_folder_validation",
}

BUFFER_SIZE = 1024 * 1024  # 1 MB


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def should_exclude(path: Path) -> bool:
    name = path.name

    if name in EXCLUDE_NAMES:
        return True

    for part in path.parts:
        if part in EXCLUDE_NAMES:
            return True

    for prefix in EXCLUDE_PREFIXES:
        if name.startswith(prefix):
            return True

    return False


def sha256sum(file_path: Path) -> str:
    h = hashlib.sha256()

    with file_path.open("rb") as f:
        while True:
            chunk = f.read(BUFFER_SIZE)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# CREATE SHA256 CHECKSUM FILE
# ============================================================

if not NMT_ROOT.exists():
    raise FileNotFoundError(f"Dataset folder not found: {NMT_ROOT}")

SHA256_FILE.parent.mkdir(parents=True, exist_ok=True)

files = []

for file_path in NMT_ROOT.rglob("*"):
    if not file_path.is_file():
        continue

    if should_exclude(file_path):
        continue

    files.append(file_path)

files = sorted(files, key=lambda p: p.relative_to(NMT_ROOT).as_posix())

with SHA256_FILE.open("w", encoding="utf-8", newline="\n") as out:
    for file_path in files:
        relative_path = file_path.relative_to(NMT_ROOT).as_posix()
        checksum = sha256sum(file_path)
        out.write(f"{checksum}  {relative_path}\n")

print("SHA256 checksum file created successfully.")
print(f"Saved to: {SHA256_FILE}")
print(f"Total files included: {len(files)}")