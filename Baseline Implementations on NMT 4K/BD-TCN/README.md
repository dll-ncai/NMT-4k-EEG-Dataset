# BD-TCN on NMT-4K-EEG — Windows / NVIDIA GPU

This package implements the Braindecode TCN (BD-TCN) for binary normal-vs-abnormal
classification on the NMT-4K-EEG dataset.

It is designed for Windows, an NVIDIA GPU with limited VRAM (for example RTX 3050
6 GB), and the released NMT-4K-EEG directory structure.

## What this implementation uses

- Modern Braindecode `1.8.1`
- the modern Braindecode TCN implementation underlying `BDTCN`
- dense temporal prediction mode by default, matching the 2020 cropped-training behavior more closely
- Original Gemein et al. BD-TCN architecture settings:
  - 5 temporal blocks
  - 55 filters
  - kernel size 16
  - dropout 0.05270154233150525
- AdamW
- cosine learning-rate schedule
- 60 s model windows after resampling to 100 Hz
- 0.5–40 Hz filtering
- recording-level probability = mean of window-level abnormal probabilities
- abnormal = positive class
- 15% stratified validation split created only from the released training split
- held-out `evaluation` split used only for final metrics

The code supports:
- preprocessing cache with skip/resume behavior
- epoch checkpoints
- automatic training resume
- progress bars
- mixed precision on CUDA
- low-VRAM training with gradient accumulation
- final Accuracy, F1, Sensitivity, Specificity, and AUROC

## Important difference from the 2020 reproduction repository

The old `auto-eeg-diagnosis-comparison` repository depended on the old Braindecode
0.4.x API plus separate `braindecode_lazy` and `NeuralArchitectureSearch`
repositories. For a fresh Windows installation, reproducing that legacy software
stack is unnecessarily fragile.

The current Braindecode release contains the Gemein et al. BD-TCN code directly.
This package therefore uses the modern Braindecode implementation while retaining
the important published BD-TCN architecture hyperparameters. Its default
`paper_dense` mode uses the dense `TCN` output (6000 input samples -> 5070 dense
predictions for the published architecture), which is closer to the original
cropped-training repository. A `pooled_modern` option is also available in
`config.yaml` if you want the current convenience wrapper behavior.

`brainfeatures` is not required for BD-TCN. It belongs to the handcrafted
feature-based pipeline.

## 1. Put this project anywhere convenient

For example:

```powershell
cd C:\Users\YOUR_NAME\Documents
```

Extract this package and enter it:

```powershell
cd BDTCN_NMT4K_Windows
```

## 2. Create the Windows environment

Install Miniconda or Miniforge first if you do not already have Conda.

Open **Anaconda PowerShell Prompt** or PowerShell with Conda initialized.

If PowerShell blocks scripts for this session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Run:

```powershell
.\install_windows.ps1
```

Then:

```powershell
conda activate bdtcn-nmt
```

The installer uses Python 3.11, a CUDA-enabled PyTorch wheel, and Braindecode
1.8.1. You do not need to install the full CUDA Toolkit separately for the
PyTorch wheel; you do need a sufficiently recent NVIDIA display driver.

## 3. Confirm that the GPU is detected

```powershell
nvidia-smi
python 01_check_setup.py
```

The setup checker should report:
- CUDA available: True
- your NVIDIA RTX 3050
- about 6 GB VRAM
- Braindecode 1.8.1

If CUDA is False, fix the PyTorch / NVIDIA driver installation before starting
training.

## 4. Dataset path

`config.yaml` is already configured for:

```text
E:\EEG Dataset\EEG\Dataset\NMT\NMT-4K-EEG
```

Expected layout:

```text
NMT-4K-EEG/
├── train/
│   ├── normal/
│   │   └── edf/
│   └── abnormal/
│       └── edf/
├── evaluation/
│   ├── normal/
│   │   └── edf/
│   └── abnormal/
│       └── edf/
└── metadata/
    └── recordings.tsv
```

Reports and abnormal-event annotation CSVs are not used for this recording-level
BD-TCN experiment.

## 5. Validate metadata and EDF locations

```powershell
python 02_validate_dataset.py --config config.yaml
```

For a slower header check of every EDF:

```powershell
python 02_validate_dataset.py --config config.yaml --check-all-edf
```

The script verifies metadata, expected split/label combinations, EDF existence,
and channel/sampling information.

## 6. Build a resumable preprocessing cache

```powershell
python 03_build_cache.py --config config.yaml
```

What happens to each EDF:
1. select the 19 NMT scalp channels in the fixed order in `config.yaml`
2. retain at most 20 minutes
3. band-pass filter 0.5–40 Hz
4. resample 200 Hz -> 100 Hz
5. convert volts -> microvolts
6. clip to ±800 microvolts
7. save a NumPy cache file

The cache lives under the directory specified in `config.yaml`; the supplied configuration places it on your `E:` drive beside (not inside) the released dataset.
With float32 and this dataset, budget roughly several tens of GB of disk space (approximately 35–40 GB is a reasonable order-of-magnitude estimate). Set `cache_dtype: 'float16'` only if disk space is tight; training still converts windows to float32.
If preprocessing is interrupted, simply run the same command again. Existing
valid cache files are skipped.

A quick test on only a few records:

```powershell
python 03_build_cache.py --config config.yaml --limit 10
```

Do not train the full experiment after a limited cache build until the remaining
records have also been cached.

## 7. Run a model smoke test

After at least a few cache files exist:

```powershell
python 04_train.py --config config.yaml --smoke-test
```

This checks:
- dataset loading
- tensor shape
- BD-TCN forward pass
- CUDA mixed precision path

It does not run the full experiment.

## 8. Train BD-TCN

```powershell
python 04_train.py --config config.yaml --resume auto
```

The training code:
- recreates a deterministic 15% stratified validation split from `train`
- never uses `evaluation` for training or threshold selection
- saves `last.pt` after every epoch
- saves `best.pt` when validation AUROC improves
- stores optimizer, scheduler, AMP scaler, epoch, random states, history,
  and the validation threshold
- shows a tqdm progress bar

If Windows restarts, the process crashes, or you stop training, run exactly:

```powershell
python 04_train.py --config config.yaml --resume auto
```

Training continues from `last.pt`.

### RTX 3050 6 GB defaults

The defaults are intentionally conservative:

```text
batch_size = 8
gradient_accumulation_steps = 8
effective batch size ≈ 64
mixed_precision = true
num_workers = 0
```

Start with the provided `batch_size: 8`; dense BD-TCN keeps many temporal
activations and 6 GB VRAM is the limiting resource. If GPU memory is comfortable,
try `12` or `16`. If you get an out-of-memory error, reduce it to `4` while
increasing `gradient_accumulation_steps` to `16`.

On Windows, `num_workers: 0` is the safest setting. Once everything works, you
may try `2`.

## 9. Final held-out evaluation

After training:

```powershell
python 05_evaluate.py --config config.yaml
```

This loads `best.pt`, uses the threshold selected from the validation subset,
and evaluates the released 1,000-recording `evaluation` partition.

Outputs include:

```text
runs/bdtcn_nmt4k/
├── checkpoints/
│   ├── last.pt
│   └── best.pt
├── validation_split.tsv
├── training_history.csv
├── evaluation_predictions.tsv
└── evaluation_metrics.json
```

Metrics:
- Accuracy
- F1
- Sensitivity (recall for abnormal)
- Specificity (recall for normal)
- AUROC from recording-level abnormal probabilities

## 10. Reproducibility / experiment choices

The uploaded NMT-4K-EEG paper specifies:
- 3,500 released training recordings
- 1,000 held-out evaluation recordings
- 15% of training (525 recordings) used for validation
- 0.5–40 Hz filtering
- abnormal as the positive class

The original 2020 TUH BD-TCN pipeline additionally:
- used 21 channels from TUH
- downsampled to 100 Hz
- clipped at 800 microvolts
- used at most 20 min per recording
- discarded the first 60 s
- used AdamW and cosine annealing
- used a 5-level, 55-filter TCN

NMT has 19 scalp channels and a small number of unusually short recordings.
The package defaults to the original `skip_first_seconds: 60.0` convention when
doing so still leaves at least one complete 60-second analysis window. For a
short recording where this is impossible, it automatically starts at 0 seconds
rather than discarding almost the entire EEG. This is an NMT-specific safety
adaptation.

Do not tune this choice using the final evaluation set.

## 11. Why the annotations and reports are ignored

BD-TCN here is a recording-level binary classifier. Its target comes from
`recordings.tsv`:

```text
Normal   -> 0
Abnormal -> 1
```

The event annotation CSVs and clinical report TXT files are useful for other
tasks but are not inputs to this binary baseline.

## 12. Expected reference result

The NMT-4K-EEG paper reports the following single-run BD-TCN reference on its
held-out evaluation partition:

```text
Accuracy    79.20%
F1          75.30%
Sensitivity 68.91%
Specificity 87.96%
AUROC       78.44%
```

Do not expect an independently modernized implementation to reproduce every
decimal exactly. Differences can arise from implementation version, random
initialization, exact crop aggregation, filtering behavior, threshold selection,
and preprocessing details.

The correct goal for the first run is a clean, leakage-free, reproducible
pipeline. Once that works, exact-reproduction ablations can be performed one
choice at a time.
