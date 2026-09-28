# Multi-BK-Net v1.0.3 training-progress hotfix

This hotfix adds live `tqdm` progress bars and enforces best-validation-
checkpoint evaluation without full-data retraining.

Each development epoch now displays two bars:

- training batches, including percentage, elapsed time, ETA, running loss,
  running accuracy, and allocated GPU memory;
- validation batches, including percentage, elapsed time, ETA, running loss,
  and allocated GPU memory. When validation finishes, the retained bar shows
  recording accuracy, recording AUROC, change from the preceding epoch,
  `NEW BEST` or `NO IMPROVEMENT`, and the best epoch/AUROC so far.

The epoch summary line records the same values in `development\history.csv`.

## Enforced model-selection protocol

1. Create the fixed 80/20 stratified split from the 3,496 usable training EDFs.
2. Train on 2,796 development-training recordings for all 42 candidate epochs.
3. Validate after each epoch on approximately 700 internal-validation
   recordings.
4. Replace `development\best.pt` only when recording-level validation AUROC
   improves. A tie retains the earlier epoch.
5. Export the saved best epoch as `best_validation_model.pt` without retraining.
6. Permit `05_evaluate.ps1` to evaluate the untouched 1,000-EDF evaluation set
   only after training finishes and only with that exact checkpoint.

The evaluator rejects a last-epoch checkpoint, a full-data-retrained
checkpoint, an incomplete training run, or any checkpoint whose epoch differs
from the selected best epoch. It also displays an evaluation progress bar.

The change does not alter model architecture, preprocessing, sample ordering,
optimization, class weights, or metric definitions. Existing `config.yaml`
files may retain `mode: development_then_full`; the updated code treats that
value as a legacy alias for `development_best_only`. The evaluator obtains the
approved checkpoint from `training_complete.json`, so the old
`evaluation.checkpoint` value is not used.

## Installation

1. Stop the active training process with `Ctrl+C` and wait for the PowerShell
   prompt to return.
2. Extract this ZIP directly into:
   `E:\Multi-BK-Net-NMT4K-Windows`
3. Allow Windows to replace the three existing Python files.
4. Run `08_run_unit_tests.ps1`.
5. If no epoch completed before training was stopped, run `04_train.ps1`.
   If one or more epochs completed and `last.pt` exists, run
   `06_resume_training.ps1` instead.

The dataset cache, internal split, and completed checkpoints are not changed.
