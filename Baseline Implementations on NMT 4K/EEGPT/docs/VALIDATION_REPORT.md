# Package validation report

Checks completed while building this ZIP:

- All Python sources compile successfully with Python 3.12 syntax checking.
- Three automated tests pass:
  - official evaluation records never become internal validation;
  - internal patient groups are disjoint;
  - recording-level confusion metrics and threshold selection are deterministic.
- The attached 4,500-row metadata passes count and duplicate checks.
- The generated split contains 2,799 internal-training, 701 validation and 1,000
  official-evaluation recordings.
- Class counts are 2,236/563, 560/141 and 540/460 respectively.
- Patient overlap is zero for all three pairwise partition comparisons.
- Source inspection confirms `train.py` never constructs an evaluation dataset,
  and `evaluate.py` never calls the threshold-selection function.
- Source inspection confirms the active v2 model accepts only `[B,21,1024]` and
  does not call the legacy interpolation function retained in the upstream file.
- The package excludes checkpoints and processed EEG tensors.

The build workspace does not contain PyTorch, MNE, the NMT EDF corpus, or the
omitted EEGPT checkpoint, so an end-to-end GPU forward/training run could not be
executed here. `scripts/00_verify_setup.bat` closes this gap on the target
machine by checking CUDA, loading the real checkpoint with a 95% encoder
coverage requirement, and running a two-window model forward pass before any
long preprocessing or training begins.

