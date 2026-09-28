# Multi-BK-Net for NMT-4K-EEG on Windows

This package adapts the official Multi-BK-Net implementation to the current
NMT-4K-EEG release and to a Windows laptop with an NVIDIA RTX 5050 (8 GB VRAM)
and 16 GB system RAM.

## Short answer

The original paper reports 1.59 GB CUDA memory for training, but its first 
temporal layers create large activation maps. This package therefore defaults
to automatic mixed precision and a micro-batch of 16 with four-step gradient 
accumulation. The effective batch size remains 64, while the peak memory 
requirement is substantially reduced.

Do not use the original repository's `environment.yml` or `requirements.txt`
on Windows. They contain Linux-only packages, direct Linux wheel paths, an
absolute Linux environment prefix, mixed CUDA runtimes, legacy Braindecode 0.7
APIs, and hard-coded `/home/...` paths. This package implements the same network
in current, native PyTorch and does not require Braindecode, skorch, Optuna, or
Weights & Biases for the final published configuration.

## What this package does

1. Checks CUDA, the RTX GPU, dependencies, and an exact model forward/backward
   pass.
2. Audits the NMT-4K directory, class counts, EDF headers, channel names,
   sampling rates, durations, duplicate identifiers, and `recordings.tsv`.
3. Preprocesses every EDF into a resumable, memory-mapped cache.
4. Creates a stratified internal validation split using only the official
   training partition.
5. Trains for the paper's maximum 42 epochs, selects the epoch using internal
   validation AUROC, then optionally retrains from scratch on the complete
   official training partition for the selected number of epochs.
6. Evaluates once on the untouched official evaluation partition.
7. Saves window-level and recording-level predictions, metrics, confusion
   matrices, ROC/PR curves, histories, checkpoints, exclusions, and resolved
   configuration files.

The primary reported results are recording-level results. Window probabilities
are averaged within each EDF before applying the 0.5 decision threshold, as in
the paper.

## Expected NMT-4K layout

The scanner is recursive, so both the compact layout described by the user and
the nested release layout are supported:

```text
G:\Dataset\NMT-4K-EEG
|-- train
|   |-- normal
|   |   `-- [optional edf folder] *.edf
|   `-- abnormal
|       `-- [optional edf folder] *.edf
|-- evaluation
|   |-- normal
|   |   `-- [optional edf folder] *.edf
|   `-- abnormal
|       `-- [optional edf folder] *.edf
`-- metadata
    `-- recordings.tsv
```

For the current v1.2 release, the expected counts are:

| Partition | Normal | Abnormal | Total |
| --- | ---: | ---: | ---: |
| Train | 2,796 | 704 | 3,500 |
| Evaluation | 540 | 460 | 1,000 |
| Total | 3,336 | 1,164 | 4,500 |

## Step-by-step Windows instructions

Open PowerShell in this package directory. Do not run the commands from inside
the dataset folder.

### 1. Allow the supplied scripts in this PowerShell window

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

This is temporary and applies only to the current PowerShell process.

### 2. Create the environment

Use 64-bit Python 3.10, 3.11, 3.12, or 3.13. Check the active interpreter:

```powershell
python --version
```

Version 1.0.1 and later support Python 3.13. Then run:

```powershell
.\00_create_environment.ps1
```

The script selects a supported 64-bit interpreter, creates `.venv`, installs a
CUDA 12.8 PyTorch wheel suitable for RTX 50-series GPUs, installs the remaining
packages, and verifies CUDA. A separate CUDA Toolkit installation is not
required by PyTorch wheels; a sufficiently recent NVIDIA driver is required.

### 3. Verify GPU and model execution

```powershell
.\01_verify_install.ps1
```

This performs an actual mixed-precision forward/backward optimizer step using
the configured micro-batch. If it runs out of memory, edit `config.yaml` and set
`micro_batch_size: 8` and `gradient_accumulation_steps: 8`. The effective batch
will still be 64.

### 4. Check the configuration

The supplied `config.yaml` already points to:

```text
G:\Dataset\NMT-4K-EEG
```

Change only that path if the dataset is elsewhere. Output and cache directories
default to `G:\Dataset\NMT-4K-EEG\multibknet_nmt_outputs`.

### 5. Inspect the full dataset

```powershell
.\02_inspect_dataset.ps1
```

This reads EDF headers only, not full signals. Review:

```text
multibknet_nmt_outputs\audit\dataset_inventory.csv
multibknet_nmt_outputs\audit\dataset_summary.json
multibknet_nmt_outputs\audit\channel_counts.csv
multibknet_nmt_outputs\audit\inspection_failures.csv
```

Do not continue if official evaluation files are missing, duplicated, unreadable,
or lack required channels.

### 6. Preprocess the EDFs

```powershell
.\03_preprocess.ps1
```

Preprocessing is resumable. Re-running the same command skips valid cached
recordings. It processes one EDF at a time and writes NumPy arrays that are
memory-mapped during training, so 16 GB RAM is sufficient.

The exact order is:

1. Select the canonical 19 NMT scalp channels.
2. Remove the first 60 seconds for every recording that still has at least one
   complete 60-second window afterward.
3. Retain at most the next 20 minutes.
4. Convert MNE data to microvolts.
5. Clip amplitudes to ±800 µV.
6. Apply common-average reference across the 19 selected scalp channels.
7. Resample from 200 Hz (or the EDF's actual rate) to 100 Hz.
8. Save `float32` recording arrays without loading the dataset into RAM.
9. Define 60-second (6,000-sample) windows with a 60-second stride. If a
   recording has a non-multiple remainder, the final complete window is aligned
   to the recording end, matching `drop_last_window=False` behavior.

NMT-4K contains a small number of unusually short EDFs. When an EDF contains at
least 60 seconds of real EEG but cannot satisfy both the 60-second lead-in crop
and a 60-second input window, the package uses its last complete 60 seconds.
This maximizes the discarded lead-in without padding, repetition, or synthetic
samples. An EDF shorter than 60 seconds is excluded from training and remains a
blocking error if it occurs in evaluation. Review both audit files:

```text
multibknet_nmt_outputs\audit\short_recording_adaptations.csv
multibknet_nmt_outputs\audit\preprocessing_failures.csv
```

No band-pass/notch filter and no z-score normalization are added because the
paper did not use them in this pipeline.

### 7. Train Multi-BK-Net

```powershell
.\04_train.ps1
```

The default `development_then_full` mode is scientifically conservative:

1. Make a fixed 80/20 stratified split from the official training recordings.
2. Train the development model for 42 epochs.
3. Select the epoch with the best recording-level validation AUROC.
4. Reinitialize the model and train on all successfully preprocessed official
   training recordings for the selected number of epochs.
5. Save a final full-training checkpoint.

The official evaluation set is never loaded during training or checkpoint
selection. If training is interrupted, run:

```powershell
.\06_resume_training.ps1
```

### 8. Evaluate the untouched evaluation set

```powershell
.\05_evaluate.ps1
```

The main output is:

```text
multibknet_nmt_outputs\runs\seed_2026\evaluation\recording_metrics.json
```

It contains:

- Accuracy
- F1 score for the abnormal/pathological class
- Sensitivity
- Specificity
- AUROC
- Balanced accuracy
- F2 score
- PR-AUC
- TN, FP, FN, and TP

`recording_predictions.csv` contains one probability per EDF. AUROC and PR-AUC
are computed from continuous abnormal probabilities, not thresholded labels.

### Run everything after environment creation

After Step 2, Steps 3-8 can also be run with:

```powershell
.\07_run_all.ps1
```

## Paper settings and laptop settings

| Item | Paper | This package |
| --- | --- | --- |
| Input duration | 60 s | 60 s |
| Sampling rate | 100 Hz | 100 Hz |
| Input samples | 6,000 | 6,000 |
| Channels | 21 TUH channels incl. A1/A2 | 19 NMT scalp channels |
| Temporal kernels | 200, 25, 13, 7, 3 | Same |
| First-block filters | 35 total, 7/branch | Same |
| Normalization | GroupNorm | Same |
| Activation | GELU | Same |
| Dropout | 0.502959339666169 | Same |
| Optimizer | AdamW | Same |
| Learning rate | 0.0031414364096615 | Same |
| Betas | (0.5, 0.999) | Same |
| Weight decay | 1.8397405899531204e-05 | Same |
| Loss | weighted NLL | Equivalent weighted cross-entropy |
| Epoch maximum | 42 | 42 |
| Batch | 64 | 16 x 4 accumulation = 64 effective |
| Precision | FP32 | AMP by default |
| Final prediction | mean window probability | Same |
| Evaluation threshold | 0.5 | 0.5 |
| Independent runs | 10 | 1 by default; repeat seeds if required |

Because GroupNorm, rather than BatchNorm, is used, reducing the physical batch
and accumulating gradients does not alter running batch statistics.

## Important scientific distinction

The paper's NMT experiment was cross-institutional testing: models were trained
on TUH data and applied without NMT training to an older NMT cohort. This package
instead trains and evaluates in-domain using the current NMT-4K predefined split.
Your results therefore answer a different question and must not be directly
presented as a replication of the paper's NMT Table 7.

## Runtime expectations

Exact time depends on laptop power limits, disk speed, thermals, and recording
durations. A realistic planning range is several hours for preprocessing and
roughly 6-12 hours for the two-stage default training on an RTX 5050 laptop.
Evaluation should be much shorter. These are planning estimates, not guarantees.

The paper's full 100-trial, 5-fold hyperparameter search is not appropriate for
this laptop. It would require vastly more computation than final training. This
package uses the published best architecture and optimizer settings instead.

## References

- Official Multi-BK-Net repository: https://github.com/nrgrp/Multi-BK-Net-general-EEG-pathology-classification
- Multi-BK-Net paper: https://openreview.net/forum?id=IsG10xZAaA
- NMT-4K-EEG repository: https://github.com/dll-ncai/NMT-4k-EEG-Dataset
- PyTorch Windows/CUDA installer: https://pytorch.org/get-started/locally/
- MNE installation: https://mne.tools/stable/install/index.html

See `docs/ORIGINAL_REPOSITORY_AUDIT.md`, `docs/PAPER_PROTOCOL_AND_ADAPTATIONS.md`,
`docs/RESULTS_GUIDE.md`, and `docs/VALIDATION_REPORT.md` for the detailed analysis.

To rerun the included unit tests after environment creation:

```powershell
.\08_run_unit_tests.ps1
```
