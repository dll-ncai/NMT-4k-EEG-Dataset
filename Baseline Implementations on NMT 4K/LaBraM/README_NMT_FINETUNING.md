# LaBraM fine-tuning on NMT-4K EEG

This folder now contains a complete binary abnormality fine-tuning path for the preprocessed NMT windows. It leaves the upstream LaBraM scripts intact and uses a separate NMT runner with current PyTorch AMP APIs.

## What the scripts assume

- Each pickle is a dictionary with `X` shaped `(21, 2000)` and `y` equal to `0` (normal) or `1` (abnormal).
- Samples are 10 seconds at 200 Hz and are stored in microvolts.
- Channel order is exactly: `FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, A1, A2, FZ, CZ, PZ`.
- The processed root contains `train` and `eval` folders. Your report currently points to `F:\NMT_processed`.

## 1. Repair the environment

Activate a Python 3.10 or 3.11 environment. Your PyTorch trio is already the correct CUDA 12.8 combination, so normally run:

```powershell
cd path\to\LaBraM-NMT
powershell -ExecutionPolicy Bypass -File .\scripts\setup_nmt_cuda128.ps1
```

If the PyTorch installation itself fails the GPU check, reinstall that trio too:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_nmt_cuda128.ps1 -ReinstallTorch
```

The official PyTorch command used by the setup script is:

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

Do not install the old LaBraM `requirements.txt` afterward because it would downgrade `timm` to `0.4.12`.

## 2. Run fine-tuning

For the paths shown in your audit CSV:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_nmt_cuda128.ps1 -DataRoot "F:\NMT_processed"
```

The default batch size is 16 with four-step gradient accumulation, giving an effective batch size of 64. If the RTX 5050 runs out of memory, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_nmt_cuda128.ps1 `
  -DataRoot "F:\NMT_processed" -BatchSize 8 -UpdateFreq 8
```

Outputs are written under `outputs\nmt_labram_base`:

- `best.pt`: best validation checkpoint, selected by recording-level balanced accuracy
- `last.pt`: most recent checkpoint for resuming
- `history.json`: epoch-by-epoch training and validation metrics
- `test_metrics.json`: final window-level and recording-level metrics
- `test_recording_predictions.csv`: one probability and prediction per EEG recording

## 3. Resume or evaluate

Resume training by adding:

```powershell
python run_nmt_finetuning.py --resume outputs\nmt_labram_base\last.pt [the same training arguments]
```

Evaluate the saved best model:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\evaluate_nmt_cuda128.ps1
```

## Why this data split and sampler are important

The original data has train and evaluation sets but no validation set. `prepare_nmt_manifest.py` creates validation data only from original training recordings, stratified by label. Every window from one recording remains in one split, so no recording can leak across train, validation, or test.

Recordings contain from 3 to 1,082 windows. The training sampler first balances normal versus abnormal recordings and then gives each recording equal total probability, preventing long EEGs from dominating an epoch. Validation threshold selection and final reporting are performed at the recording level by averaging the window probabilities from each EEG.

The runner keeps the official LaBraM input scaling (`microvolts / 100`) and maps the exact 21 NMT channels to LaBraM's pretrained positional embeddings.

## Useful direct commands

Create only the manifest:

```powershell
python prepare_nmt_manifest.py --data-root "F:\NMT_processed" --verify-files all
```

Check only the environment and checkpoint:

```powershell
python check_labram_environment.py
```

Use every training window per epoch instead of drawing 50,000 balanced samples:

```powershell
python run_nmt_finetuning.py --epoch-size 0 [other arguments]
```

The bundled official checkpoint may contain an older Python training-argument object. The launch scripts use `--allow-unsafe-checkpoint` only because this checkpoint is the repository's known LaBraM artifact. Do not use that flag with an untrusted `.pth` file.
