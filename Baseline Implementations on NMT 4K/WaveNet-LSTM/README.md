# WaveNet-LSTM on NMT-4K-EEG (Windows, RTX 3050 6 GB)

This is a from-scratch PyTorch implementation of the dual-path WaveNet-LSTM architecture described by Albaqami et al. (Sensors 2023), adapted to the NMT-4K-EEG benchmark split.

## Important implementation choices

The source paper specifies the major architecture but does not provide enough detail to reproduce every implementation line exactly. Figure 5 also has an internal shape inconsistency: it labels lower-path intermediate tensors as `(N,30,20)` while the blocks are labelled `LSTM(64)`. This implementation preserves the specified 64-unit LSTMs and the unambiguous final `(N,30,2) -> 60` flattening, so the fusion vector is `64 + 60 = 124` as shown in the paper.

NMT-4K-specific defaults:

- Uses the released subject-wise train/evaluation split.
- Creates a stratified 15% validation subset from the 3,500 training recordings using seed 42.
- Treats abnormal as the positive class (`1`).
- Filters 0.5-40 Hz, matching the NMT-4K benchmark protocol.
- Creates the 20-channel TCP bipolar montage used by the WaveNet-LSTM paper from the 19 NMT scalp channels.
- Resamples from native 200 Hz to 250 Hz so 60 seconds equals 15,000 samples, matching Figure 5 of the WaveNet-LSTM paper.
- Uses the first 60 seconds as the main sample.
- For fitting only, a full second 60-second segment (60-120 s), when available, is time-reversed and used as an extra training sample.
- Short recordings are zero-padded after per-channel z-score normalization. This padding rule is an explicit adaptation for NMT-4K because NMT contains some recordings shorter than 60 seconds.
- Validation and evaluation recordings are never augmented.

## 1. Create the Windows environment

Open **Anaconda Prompt** or **PowerShell** in this project directory.

```powershell
conda create -n nmt-wavenet python=3.11 -y
conda activate nmt-wavenet
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build. For a recent NVIDIA driver, start with:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining packages:

```powershell
pip install -r requirements.txt
```

You do not need to separately install the full CUDA Toolkit for normal PyTorch wheel usage; the NVIDIA display/compute driver must be recent enough for the chosen PyTorch CUDA runtime.

## 2. Verify the dataset path

`config.yaml` is already set to:

```text
E:\EEG Dataset\EEG\Dataset\NMT\NMT-4K-EEG
```

Expected structure:

```text
NMT-4K-EEG\
  train\
    normal\edf\*.edf
    abnormal\edf\*.edf
  evaluation\
    normal\edf\*.edf
    abnormal\edf\*.edf
  metadata\recordings.tsv
```

## 3. Check GPU, metadata, one EDF, and model

```powershell
python 00_check_setup.py --config config.yaml
```

Optional full synthetic GPU forward test:

```powershell
python 00_check_setup.py --config config.yaml --model-forward
```

You should see `CUDA available: True` and your RTX 3050.

## 4. Create the manifest and 15% validation split

```powershell
python 01_make_splits.py --config config.yaml
```

Expected NMT-4K totals are 3,500 training records and 1,000 held-out evaluation records. The script creates 2,975 fit and 525 validation recordings from the training split.

## 5. Preprocess and cache EDFs

```powershell
python 02_preprocess.py --config config.yaml
```

This step has a tqdm progress bar and is resumable: already valid `.npy` cache files are skipped. If the process stops or Windows restarts, run the same command again.

Cache layout:

```text
cache_wavenet_lstm\
  base\       # first 60 s for fit/validation/evaluation
  reverse\    # reversed second 60 s, fit records only when a full segment exists
  cache_index.csv
  preprocess_failures.csv   # only if a file failed
```

The cache can require several GB of disk space.

## 6. Train

```powershell
python 03_train.py --config config.yaml --resume auto
```

Defaults are conservative for a 6 GB RTX 3050:

- physical batch size: 4
- gradient accumulation: 4 (effective optimizer batch around 16)
- mixed precision (AMP): on
- Adam, learning rate 0.001
- ReduceLROnPlateau down to 0.0001
- maximum 50 epochs
- early stopping after 10 validation-loss epochs without improvement
- weighted sampling to approximately balance normal/abnormal training samples

Training saves after every epoch:

```text
runs\wavenet_lstm_nmt4k\last.pt
runs\wavenet_lstm_nmt4k\best.pt
runs\wavenet_lstm_nmt4k\history.csv
```

If training is interrupted, rerun the same training command. `--resume auto` loads `last.pt` and continues from the next epoch.

If you get CUDA out-of-memory, edit `config.yaml` and change:

```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8
```

That keeps a similar effective optimizer batch while reducing VRAM usage.

## 7. Evaluate the held-out evaluation set

```powershell
python 04_evaluate.py --config config.yaml --split evaluation
```

The script reports:

- Accuracy
- F1 score (abnormal positive)
- Sensitivity / recall of abnormal
- Specificity / true-negative rate of normal
- AUROC
- TN, FP, FN, TP

It also saves:

```text
runs\wavenet_lstm_nmt4k\evaluation_results\metrics.json
runs\wavenet_lstm_nmt4k\evaluation_results\predictions.csv
runs\wavenet_lstm_nmt4k\evaluation_results\confusion_matrix.png
runs\wavenet_lstm_nmt4k\evaluation_results\roc_curve.png
```

The default classification threshold is 0.5. To select a threshold using validation only, change `evaluation.threshold_mode` to `f1` or `youden`. The held-out evaluation set is never used to select the threshold.

## One-command run after the environment is ready

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
```

For a first run, I recommend executing the scripts one at a time instead, so any EDF/channel/path issue is easy to identify before a long training run.
