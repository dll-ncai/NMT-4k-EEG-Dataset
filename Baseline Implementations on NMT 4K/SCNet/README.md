# SCNet for NMT-4K-EEG — Windows / RTX 3050 6 GB

This is a clean PyTorch reconstruction of **SCNet: A spatial feature fused convolutional network for multi-channel EEG pathology detection** (Wu et al., 2023), adapted to the current NMT-4K-EEG directory structure.

## Can an RTX 3050 6 GB run it?

Yes. SCNet itself is very small (roughly 40–50k trainable parameters depending on the number of input channels), but a 7-minute 100 Hz EEG input is large. The default configuration therefore uses:

- mixed precision (AMP)
- micro-batch size 8
- gradient accumulation 4 (effective batch ≈ 32)
- lazy loading from a float16 preprocessing cache
- resumable preprocessing and training checkpoints

If batch size 8 causes CUDA OOM on your specific laptop, run training with:

```bat
python train.py --config config.yaml --batch-size 4 --accum-steps 8 --resume auto
```

## What is reproduced from the SCNet paper?

The model follows Fig. 2 and Sections 3.1–3.3:

- SILM: channel-wise global average, max, and population-standard-deviation pooling, each with dropout 0.05, concatenated with the raw input
- DFRL with average/max pooling fusion, MFFM blocks, residual element-wise additions, spatial dropout 0.5, and 1D convolutions
- MFFM: Conv1D(8, k=5) then Conv1D(16, k=5), with BN + ReLU and dense concatenation
- main DFRL Conv1D layers: 32 filters, k=3, stride 1
- classifier: temporal global average pooling -> Dense(2)
- Xavier-uniform convolution/dense initialization and zero biases
- Adam, learning rate 1e-3, maximum 60 epochs
- original SCNet input length of 7 minutes

The paper does not explicitly state convolution padding, and MFFM skip concatenations require equal temporal dimensions. This implementation therefore uses `same` padding (`k//2`).

## Two preprocessing modes

The default `nmt4k19` mode is recommended for the current NMT-4K-EEG benchmark:

- use the 19 scalp channels only
- 0.5–40 Hz bandpass
- resample to 100 Hz
- clip to ±100 µV
- use one deterministic 7-minute window per recording
- discard the first minute when enough signal is available
- shorter recordings are zero-padded so all 1,000 evaluation recordings remain evaluable

The optional `paper22` mode is closer to the original 2023 SCNet preprocessing:

- select the 19 scalp channels plus A1/A2
- resample to 100 Hz
- discard the first minute when possible
- clip to ±100 µV
- form the 22 bipolar derivations listed in the SCNet paper

To use it, edit `config.yaml`:

```yaml
preprocess:
  mode: 'paper22'
  bandpass: null
```

and change `cache_dir` / `output_dir` so you do not mix incompatible caches.

## 1. Install Miniconda

Install Miniconda or Anaconda for Windows if `conda` is not already available. Open **Anaconda Prompt** in this project directory.

## 2. Create the environment

Automatic:

```bat
install_windows.bat
```

Manual equivalent:

```bat
conda create -n scnet_nmt4k python=3.11 -y
conda activate scnet_nmt4k
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

You do **not** need to install the full CUDA Toolkit separately for normal PyTorch wheel use. A sufficiently recent NVIDIA driver is required.

## 3. Check `config.yaml`

The supplied file already contains your dataset path:

```text
E:\EEG Dataset\EEG\Dataset\NMT\NMT-4K-EEG
```

Your expected layout is:

```text
NMT-4K-EEG/
  metadata/recordings.tsv
  train/normal/edf/*.edf
  train/abnormal/edf/*.edf
  evaluation/normal/edf/*.edf
  evaluation/abnormal/edf/*.edf
```

Annotations and reports are not needed for recording-level SCNet classification.

## 4. Validate GPU, metadata, and EDF channels

```bat
conda activate scnet_nmt4k
python check_setup.py --config config.yaml
```

Expected NMT-4K split counts are approximately:

- train: 3,500
- evaluation: 1,000
- train normal / abnormal: 2,796 / 704
- evaluation normal / abnormal: 540 / 460

`CUDA available` must be `True` before training.

## 5. Preprocess and cache the EEGs

```bat
python preprocess.py --config config.yaml
```

A tqdm progress bar is shown. The cache is resumable: re-running skips valid `.npy` files already produced. A `manifest.csv` records success/error status.

For 19 channels × 42,000 samples × float16, 4,500 recordings require about 7.2 GB before filesystem overhead. Keep roughly 10+ GB free for cache, checkpoints, and outputs.

## 6. Train SCNet

```bat
python train.py --config config.yaml --resume auto
```

Important behavior:

- fixed random seed 42
- stratified 15% validation subset created only from the 3,500 training recordings
- held-out evaluation recordings are never used for training/model selection
- best model selected by validation AUROC
- validation-only threshold tuning (F1 by default)
- progress bars for training and validation
- current VRAM allocation displayed during training
- `checkpoints/last.pt` supports automatic resume
- periodic mid-epoch checkpointing is enabled
- `checkpoints/best.pt` stores the best validation-AUROC model
- `training_history.csv` is updated each epoch

If the laptop loses power or training is interrupted, run the same command again. `--resume auto` continues from `last.pt`.

## 7. Evaluate the untouched evaluation set

```bat
python evaluate.py --config config.yaml --threshold auto
```

The script reports:

- Accuracy
- F1 score (abnormal is the positive class)
- Sensitivity
- Specificity
- AUROC
- TN / FP / FN / TP

It saves:

```text
runs/scnet_nmt4k19/evaluation_metrics.json
runs/scnet_nmt4k19/evaluation_predictions.csv
runs/scnet_nmt4k19/evaluation_confusion_matrix.png
```

For transparency, `evaluation_metrics.json` contains both metrics using the validation-tuned threshold and metrics using a fixed threshold of 0.5.

## Expected reference point

The NMT-4K-EEG paper reports the following SCNet single-run held-out baseline:

- Accuracy: 77.60%
- F1: 71.93%
- Sensitivity: 62.39%
- Specificity: 90.56%
- AUROC: 76.51%

Do not expect byte-for-byte or metric-identical reproduction solely from the two papers. The current NMT-4K paper states that model-specific cropping/segmentation, optimization, checkpoint selection, threshold selection, and recording-level aggregation are contained in its accompanying code repository; not all those details are specified in the manuscript text. This package therefore makes every adaptation/assumption explicit.

## Notes about the old 2023 NMT experiment

The original SCNet paper evaluated an older NMT subset of 1,805 recordings, not the current 4,500-recording NMT-4K release. It used 21 selected electrodes, re-referenced them into 22 bipolar channels, used 100 Hz sampling, discarded the first minute, clipped amplitude, used a 7-minute input, 90/10 train/validation split, Adam at 1e-3, batch size 32, and 60 epochs. Its reported result on that older NMT experiment was 80.74% accuracy and 77.58% F1.

Your current benchmark is different, so the NMT-4K evaluation numbers are the appropriate comparison.
