# BD-Deep4 for NMT-4K-EEG (Windows, PyTorch)

This is an independent PyTorch implementation of the Deep ConvNet / BD-Deep4 architecture from Schirrmeister et al. for recording-level Normal vs Abnormal classification on NMT-4K-EEG.

The package is designed for a Windows laptop with an NVIDIA RTX 3050 6 GB GPU and 32 GB RAM. It uses short EEG windows, mixed-precision training, memory-mapped preprocessed caches, progress bars, validation-only threshold selection, recording-level probability aggregation, and resumable checkpoints.

## What this pipeline does

1. Reads `metadata/recordings.tsv` from the NMT-4K-EEG root.
2. Keeps the released `train` and `evaluation` split intact.
3. Splits 15% of the released training partition into validation using stratification and seed 42.
4. Resolves EDF files from `train|evaluation/normal|abnormal/edf/`.
5. Selects the 19 scalp EEG channels in the released channel order.
6. Filters every recording to 0.5-40 Hz and resamples to 100 Hz.
7. Saves each recording once as a float16 NumPy cache. Re-running preprocessing skips valid cached files.
8. Computes channel mean/std using only the model-fitting training subset.
9. Trains Deep4 using random 6-second windows and mixed precision.
10. Saves `latest.pt` during training and `best.pt` by validation AUROC. Re-running training resumes automatically from `latest.pt`.
11. Selects the binary decision threshold on validation only by maximum F1.
12. Evaluates the untouched released evaluation set, averaging window probabilities to one probability per recording.
13. Reports Accuracy, F1, Sensitivity, Specificity and AUROC, with Abnormal as the positive class.

## Expected NMT-4K-EEG layout

The default `config.yaml` already points to:

```text
E:\EEG Dataset\EEG\Dataset\NMT\NMT-4K-EEG
```

Expected important paths are:

```text
NMT-4K-EEG\
  metadata\
    recordings.tsv
  train\
    normal\edf\*.edf
    abnormal\edf\*.edf
  evaluation\
    normal\edf\*.edf
    abnormal\edf\*.edf
```

Reports and event-annotation CSVs are not used for this binary recording-level baseline.

## 1. Create the Windows environment

Install a current NVIDIA driver first. Then install Miniconda or Anaconda if needed.

Open **Anaconda Prompt** in this project directory and run:

```bat
conda create -n nmt-deep4 python=3.11 -y
conda activate nmt-deep4
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

PyTorch changes the CUDA builds offered by its installer over time. If the `cu128` command is no longer offered or your driver is too old, use the official PyTorch selector for Windows + Pip + Python + CUDA, then install `requirements.txt`.

You can alternatively run:

```bat
setup_windows.bat
```

## 2. Check that PyTorch can use the RTX 3050

```bat
python 00_check_gpu.py
```

You want to see:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 3050 ...
CUDA smoke test: PASS
```

Then verify the model itself:

```bat
python 05_smoke_test_model.py
```

## 3. Check `config.yaml`

The two paths that matter most are:

```yaml
dataset_root: 'E:\EEG Dataset\EEG\Dataset\NMT\NMT-4K-EEG'
work_dir: 'E:\EEG Dataset\EEG\Dataset\NMT\NMT-4K-EEG_BDDeep4_work'
```

Do not point `work_dir` inside an EDF directory. It holds generated manifests, preprocessed `.npy` files, checkpoints and result files.

The provided laptop-friendly defaults are:

```yaml
preprocessing:
  sfreq: 100
  l_freq: 0.5
  h_freq: 40.0
  cache_dtype: float16

training:
  window_seconds: 6.0
  windows_per_recording: 4
  val_windows_per_recording: 16
  epochs: 60
  batch_size: 64
  eval_batch_size: 128
  amp: true
```

If you get a CUDA out-of-memory error, first change `batch_size` from 64 to 32. If necessary use 16. The model architecture itself is small; 6 GB VRAM is sufficient for these window sizes.

## 4. Build the manifest and validation split

```bat
python 01_prepare_manifest.py --config config.yaml
```

This checks that every EDF referenced by metadata exists and creates:

```text
<work_dir>\manifests\records.csv
```

The released evaluation set is never mixed into training or validation.

## 5. Preprocess and cache the EEG recordings

```bat
python 02_preprocess_cache.py --config config.yaml
```

This is the longest CPU / disk-I/O stage. It displays a progress bar. It writes one 19-channel float16 `.npy` file for each EDF under the generated cache directory.

### Preprocessing resumability

If this command is stopped, run the exact same command again. Valid existing cache files are checked and skipped. Only missing/invalid files are processed.

To deliberately regenerate every cache file:

```bat
python 02_preprocess_cache.py --config config.yaml --overwrite
```

The script then computes global per-channel normalization statistics using only the model-fitting training subset. Validation and evaluation data are not used to estimate these statistics.

## 6. Train BD-Deep4

```bat
python 03_train.py --config config.yaml
```

Training uses random windows from every training recording, BCE-with-logits loss, class weighting by default, Adam, automatic mixed precision on CUDA, ReduceLROnPlateau, early stopping by validation AUROC, and a tqdm progress bar.

Important generated files are:

```text
<work_dir>\runs\bddeep4_100hz_6s_seed42\checkpoints\latest.pt
<work_dir>\runs\bddeep4_100hz_6s_seed42\checkpoints\best.pt
<work_dir>\runs\bddeep4_100hz_6s_seed42\artifacts\history.csv
<work_dir>\runs\bddeep4_100hz_6s_seed42\artifacts\threshold.json
```

### Training resumability

`latest.pt` is saved during an epoch and again at every epoch boundary. If Windows restarts, the laptop sleeps, or you press Ctrl+C, run:

```bat
python 03_train.py --config config.yaml
```

and it will resume from `latest.pt` automatically. The checkpoint contains the model, optimizer, scheduler, AMP scaler, epoch/batch position and random-number-generator states.

To intentionally start a fresh training run while keeping old checkpoints on disk:

```bat
python 03_train.py --config config.yaml --no-resume
```

For a genuinely fresh experiment, changing `run_name` in `config.yaml` is safer because it creates a new output directory.

## 7. Evaluate the held-out NMT evaluation set

Run this only after training is finished:

```bat
python 04_evaluate.py --config config.yaml --split evaluation
```

By default it scans each complete recording with non-overlapping 6-second windows, includes the end of the recording, predicts every window, then takes the mean abnormal probability across all windows belonging to the same recording.

The threshold used for the main metrics is selected on validation only. The script also saves results at a conventional threshold of 0.5 for transparency.

It prints:

```text
Accuracy
F1
Sensitivity
Specificity
AUROC
TN / FP / FN / TP
```

and saves:

```text
<run_dir>\artifacts\evaluation_predictions.csv
<run_dir>\artifacts\evaluation_metrics.json
```

Here **Abnormal = positive (1)**, so:

```text
Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)
F1          = F1 of the Abnormal class
AUROC       = ROC AUC from recording-level abnormal probabilities
```

## 8. Optional exhaustive validation check

To calculate validation results using the same sliding-window evaluation procedure as the final set:

```bat
python 04_evaluate.py --config config.yaml --split val
```

This does not alter the trained model or the held-out evaluation set.

## Deep4 architecture implemented here

For input `[batch, 19, time]`, the implementation is:

```text
Temporal Conv: 25 filters, kernel 10
Spatial Conv:  25 filters, kernel spans all 19 channels
BatchNorm -> ELU -> MaxPool(3, stride 3)

Dropout -> Conv 50,  kernel 10 -> BatchNorm -> ELU -> MaxPool(3,3)
Dropout -> Conv 100, kernel 10 -> BatchNorm -> ELU -> MaxPool(3,3)
Dropout -> Conv 200, kernel 10 -> BatchNorm -> ELU -> MaxPool(3,3)

Final convolution across the remaining temporal dimension -> 1 binary logit
```

The implementation uses Xavier/Glorot initialization and a dropout probability of 0.5 by default.

## Why 100 Hz and 6-second windows?

NMT-4K-EEG is recorded at 200 Hz, while the paper's common baseline protocol filters 0.5-40 Hz. Resampling the already band-limited signal to 100 Hz leaves a 50-Hz Nyquist frequency and greatly reduces CPU cache size, GPU memory use, and training time. Six seconds gives 600 samples, comfortably above the minimum input length needed by the default four Deep4 convolution/pooling blocks.

This is a resource-aware implementation for an RTX 3050 laptop, not a claim that 100 Hz / 6 s exactly matches the unpublished architecture-specific settings used for the paper's BD-Deep4 result. Those settings can be changed in `config.yaml` and the model code if an exact reference pipeline becomes available.

## One-command execution

After the environment is installed and `config.yaml` is correct, you may run:

```bat
run_pipeline.bat
```

For a real experiment, running the numbered scripts one by one is recommended because preprocessing and training can each take substantial time.

## Troubleshooting

### `CUDA available: False`

Do not start a long training run. Confirm `nvidia-smi` works, update the NVIDIA driver if needed, activate the correct Conda environment, and reinstall a CUDA-enabled PyTorch wheel from the official PyTorch Windows installer selector.

### CUDA out of memory

Change in `config.yaml`:

```yaml
batch_size: 32
eval_batch_size: 64
```

If needed reduce them to 16 and 32 respectively. Keep `amp: true`.

### DataLoader worker errors on Windows

Set:

```yaml
num_workers: 0
```

This is slower but is the most conservative Windows setting.

### Disk space

Because all 19 channels are cached at 100 Hz as float16, expect the generated cache to require many GB. Exact size depends on recording duration. Keeping roughly 25-30 GB free for cache, checkpoints and outputs is a sensible working margin for this dataset.

### Want a faster test before the full run?

Temporarily set `max_duration_minutes: 2` and/or `max_windows_per_recording: 20`. Do not use those shortened settings for the final benchmark. Change `run_name` before switching back to a full experiment so outputs are not mixed.

## Files in this project

```text
00_check_gpu.py             GPU/PyTorch verification
01_prepare_manifest.py      metadata, EDF paths, train/validation/evaluation roles
02_preprocess_cache.py      filter/resample/cache + training-only normalization stats
03_train.py                 resumable Deep4 training
04_evaluate.py              recording-level evaluation and requested metrics
05_smoke_test_model.py      forward/backward model test
config.yaml                 all important experiment settings
requirements.txt            non-PyTorch Python dependencies
setup_windows.bat           optional environment setup helper
run_pipeline.bat            optional sequential runner
nmt_deep4/model.py          Deep4 architecture
nmt_deep4/data.py           random/fixed EEG window datasets
nmt_deep4/metrics.py        Accuracy/F1/Sensitivity/Specificity/AUROC
nmt_deep4/checkpoint.py     atomic resumable checkpoints
nmt_deep4/eval_utils.py     window-to-recording probability aggregation
```
