# Compatibility and dataset audit

## Environment findings

| Component | Supplied version | Finding | Action used |
|---|---:|---|---|
| PyTorch | 2.10.0+cu128 | Correct CUDA 12.8 wheel for the supplied torchvision and torchaudio versions | Keep |
| torchvision | 0.25.0+cu128 | Correct match for PyTorch 2.10.0 | Keep |
| torchaudio | 2.10.0+cu128 | Correct match for PyTorch 2.10.0 | Keep |
| NumPy | 2.2.6 | ABI-risky with the older pandas, h5py, MNE, scikit-learn and RDKit generation in this environment | Pin 1.26.4 |
| pandas | 1.5.3 | Predates general NumPy 2 wheel compatibility | Upgrade to 2.2.3 |
| timm | 0.4.12 | The original LaBraM pin is too old for a modern PyTorch stack; the project compatibility file already intended 0.9.16 | Upgrade to 0.9.16 |
| scikit-learn | 1.3.0 | Old alongside NumPy 2.x; metrics used by the new runner are stable in 1.5.2 | Upgrade to 1.5.2 |
| DeepSpeed | not installed | LaBraM's commented `deepspeed==0.4.0` is obsolete for PyTorch 2.x | Do not install; single-GPU native AMP is used |

PyTorch officially lists `torch 2.10.0`, `torchvision 0.25.0`, and `torchaudio 2.10.0` with the `cu128` index for Windows and Linux. NVIDIA introduced Blackwell support in CUDA 12.8. The environment checker also verifies that the installed PyTorch binary recognizes the GPU and, for compute capability 12.0, includes `sm_120`.

NumPy 2 introduced an ABI break, so extensions built against NumPy 1.x may fail to import. pandas identifies 2.2.2 as its first release generally compatible with NumPy 2. The conservative project environment therefore uses NumPy 1.26.4 while upgrading pandas and scikit-learn.

Primary references:

- https://pytorch.org/get-started/previous-versions/
- https://docs.pytorch.org/docs/stable/amp.html
- https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/
- https://numpy.org/devdocs/release/2.0.0-notes.html
- https://pandas.pydata.org/docs/whatsnew/v2.2.2.html

## Checkpoint finding

The working copy of `LaBraM/checkpoints/labram-base.pth` in the uploaded archive was 84,410,368 bytes and failed ZIP integrity testing because its central directory was missing. The embedded LaBraM Git object contains the intact 96,612,769-byte checkpoint. The delivered project restores that intact version and validates it before training.

## Dataset findings

| Split in supplied data | Normal recordings | Abnormal recordings | Windows |
|---|---:|---:|---:|
| Train | 2,795 | 704 | 383,464 |
| Evaluation/test | 537 | 460 | 109,912 |
| Total | 3,332 | 1,164 | 493,376 |

- All 493,376 audited pickles have shape `(21, 2000)` and status `Valid`.
- They represent 4,496 EEG recordings. Four metadata recordings have no valid pickle windows: one original training normal and three evaluation normals.
- Windows per recording range from 3 to 1,082, making ordinary window-uniform sampling strongly biased toward longer EEGs.
- The original preprocessing code lists 23 possible channels, but the saved and audited data contains 21. The fine-tuning runner therefore uses the exact 21-channel order, without nonexistent `T1` and `T2` inputs.
- Validation is created from original training recordings only. The supplied evaluation set remains untouched until final testing.
- Both window-level and recording-level results are saved, but recording-level metrics are the primary results for EEG abnormality classification.

## Training compatibility changes

- Uses `torch.amp.autocast` and `torch.amp.GradScaler`, replacing deprecated `torch.cuda.amp` calls.
- Uses bfloat16 automatically on the RTX 5050 when PyTorch reports support; otherwise uses float16.
- Avoids the old PyHealth metrics dependency in the training path.
- Loads only the pretrained LaBraM student backbone and discards the pretraining language-model head before creating a one-logit binary head.
- Uses the official base configuration: 200-sample patches, absolute positional embeddings, no relative positional bias, no QKV bias, layer decay 0.65, and input scaling by 100.
- Provides recording-balanced sampling, gradient accumulation, early stopping, safe resume checkpoints, and final recording-level prediction export.

## Validation performed on the delivered code

- Python compilation and Ruff static analysis pass for every new Python module.
- A full dry run over all 493,376 inspection rows produced 2,974 train, 525 validation, and 997 test recordings with zero recording overlap between splits.
- The restored base checkpoint passes ZIP member and CRC testing: 903 members, 96,612,769 bytes, SHA-256 `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c`.
- The `timm 0.9.16` wheel contains all legacy modules imported by LaBraM and has no `torch._six` references.
- Channel positional indices are unique and valid for all 21 NMT inputs.

An actual CUDA model forward/backward pass could not be executed in the packaging workspace because it does not contain PyTorch or an NVIDIA GPU. `check_labram_environment.py` performs the missing CUDA architecture and matrix-kernel smoke tests on the RTX 5050 before the PowerShell training script starts. The training entry point then validates checkpoint loading and model-state coverage before reading the first batch.
