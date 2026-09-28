# Windows troubleshooting

## Version 1.0.0 reports `Found Python 3.13`

The initial package rejected Python 3.13 too conservatively. Version 1.0.1 and
later accept 64-bit Python 3.13 and use the official Windows CUDA 12.8 PyTorch
wheel. Replace the package with the latest version, then run:

```powershell
python --version
Set-ExecutionPolicy -Scope Process Bypass
.\00_create_environment.ps1
```

No second Python installation is required.

## Cropping reports six unusually short recordings

Package version 1.0.2 adds an explicit, conservative short-recording policy.
For a recording with at least 60 seconds of real EEG but less than 60 seconds
remaining after the requested lead-in crop, it uses the recording's last
complete 60 seconds. It never zero-pads, repeats, or synthesizes EEG samples.
Recordings shorter than 60 seconds remain excluded.

After updating the package in place, rerun:

```powershell
.\03_preprocess.ps1
```

Do not use `--overwrite`. The 4,494 valid cache entries from version 1.0.1 are
reused, and only the previously failed EDFs are retried. Confirm that
`evaluation_failures` is zero, then review:

```text
G:\Dataset\NMT-4K-EEG\multibknet_nmt_outputs\audit\short_recording_adaptations.csv
G:\Dataset\NMT-4K-EEG\multibknet_nmt_outputs\audit\preprocessing_failures.csv
```

If an evaluation recording is genuinely shorter than 60 seconds, the package
still blocks evaluation rather than fabricating a model input.

## `torch.cuda.is_available()` is False

1. Run `nvidia-smi` in PowerShell and confirm the RTX 5050 is listed.
2. Rerun `00_create_environment.ps1`; it installs from the CUDA 12.8 PyTorch
   wheel index.
3. Confirm the environment's Python is used:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Do not install the CPU-only Torch wheel from the default pip index afterward.

## `sm_120 is not compatible` or `no kernel image is available`

The installed Torch build predates RTX 50-series/Blackwell support. Remove and
reinstall Torch from the CUDA 12.8 index:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch
.\.venv\Scripts\python.exe -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

Then rerun `01_verify_install.ps1`.

## PowerShell blocks scripts

Run this once in the current window:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## GPU out of memory

In `config.yaml`, change:

```yaml
micro_batch_size: 8
gradient_accumulation_steps: 8
validation_batch_size: 16
```

The effective training batch remains 64. For evaluation, reduce
`evaluation.batch_size` to 16 or 8. Resume with `06_resume_training.ps1`.

## Windows DataLoader worker crash or RAM pressure

Keep `training.num_workers: 0` and `evaluation.num_workers: 0`. This package is
designed to stream memory-mapped arrays, and GPU computation is likely to be the
main bottleneck. Raising workers is optional, not required.

## Missing channels

Review `audit/dataset_inventory.csv` and `audit/channel_counts.csv`. The package
recognizes common `EEG ...-REF`, `EEG ...-LE`, T7/T8/P7/P8, and T3/T4/T5/T6
naming variants. It intentionally refuses bipolar channel pairs such as
`Fp1-F7`, because they are not equivalent to referential scalp channels.

Do not silently synthesize absent channels or fill them with zeros. If your
specific release truly has 21 referential channels including A1/A2, confirm the
release documentation, add A1 and A2 to `dataset.channels`, delete the old
cache/output run, and preprocess/train from the beginning.

## Preprocessing was interrupted

Run `03_preprocess.ps1` again. A cache entry is reused only if its EDF size and
modification time, preprocessing signature, channel count, and NumPy shape all
match. Partial temporary files are not accepted.

## Training was interrupted

Run `06_resume_training.ps1`. Checkpoints are written after every completed
epoch. Resume verifies the phase and configuration hash before loading optimizer,
scheduler, AMP scaler, and model states.

## You changed preprocessing, channels, model, or optimizer settings

Do not resume an incompatible run. Use a new output directory in `config.yaml`
or move the old run directory out of the way. Preprocessing-signature changes
will automatically force affected cache entries to be rebuilt.

## Evaluation recording count is below 1,000

Do not report the result as the complete predefined evaluation benchmark. Fix
or restore the missing/unreadable EDFs and rerun preprocessing. The default
package blocks final evaluation if any official evaluation recording failed.
