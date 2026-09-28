# Audit of the supplied Multi-BK-Net repository

## Conclusion

The supplied repository is valuable as a research reference, but it is not a
turnkey package for raw NMT-4K EDFs or for Windows. Running its instructions
unchanged would fail before training. Editing only the three dataset paths would
not be sufficient.

## Repository execution order as originally intended

The original materials imply this order:

1. Run `TUH_data_preprocessing_and_saving.ipynb` on TUAB/TUABEXB.
2. Serialize preprocessed Braindecode `BaseConcatDataset` objects into Linux
   folders such as `/home/TUABCOMB/final_train/`.
3. Run `HPO_MultiBKNet_TUABCOMB_5fold.py` for Optuna/W&B hyperparameter search.
4. Place the chosen parameter CSV in a specific `/home/...` location.
5. Run `RUN_final_training_MultiBKNet.py`, which imports
   `final_training_and_eval_MultiBKNet.py`.
6. Load the pre-serialized TUH datasets, create 6,000-sample windows, train, and
   evaluate TUAB and TUABEXB.

That flow assumes data products and directory conventions not included in the
repository and not present in NMT-4K.

## Blocking Windows/environment problems

- `environment.yml` was exported from Linux and ends with the absolute prefix
  `/home/kiessnek/miniconda3/envs/braindecode`.
- It contains Linux system libraries and Linux-only direct wheel paths.
- It mixes CUDA 11 and CUDA 12 packages and includes many unrelated packages.
- The plain `requirements.txt` also contains machine-specific Linux wheel paths
  and operating-system packages that pip cannot install on Windows.
- The code targets Python 3.8, Braindecode 0.7, skorch 0.11, and legacy imports
  such as `braindecode.datautil`, while a current Windows/PyTorch environment
  uses newer APIs.
- RTX 50-series GPUs require a PyTorch build with Blackwell support, normally a
  CUDA 12.8 wheel. The repository's fixed Torch/CUDA mixture is inappropriate.

## Blocking path and data-loader problems

- `/home/code/`, `/home/TUABCOMB/`, `/home/TUAB/`, `/home/TUABEXB/`,
  `/home/results/`, `/home/HPO/`, and `/home/wandb/` are hard-coded.
- The loader expects Braindecode's previously serialized concatenated datasets,
  not the user's `train/normal`, `train/abnormal`, `evaluation/normal`, and
  `evaluation/abnormal` raw EDF tree.
- TUH-specific fields such as `pathological`, TUH patient IDs, and a supplied
  TUABCOMB five-fold CSV are assumed.
- `RUN_final_training_MultiBKNet.py` tries to read a parameter file under a
  generated `Final_train_and_eval_MultiBKNet` directory, but that exact run file
  is not part of the supplied root layout.
- The published parameter CSV contains a column named `pool_block_conv ` with a
  trailing space, which is easy to break during manual editing.

## NMT incompatibilities

- The original final model hard-codes 21 channels and a spatial kernel spanning
  all 21. The current NMT-4K release documents 19 scalp channels. A model built
  for 21 channels cannot accept a `(batch, 19, 6000)` tensor.
- The preprocessing notebook has TUH channel-renaming dictionaries specifically
  for `EEG ...-REF` and `EEG ...-LE`. NMT requires its own robust channel audit
  and selection.
- The original code has no support for NMT-4K's fixed 3,500/1,000 directory split
  or its `recordings.tsv` audit.

## Evaluation defects corrected in this package

### AUROC and PR-AUC use hard labels

The original `compute_and_save_recording_results` converts probabilities into
class predictions and then calls:

```python
roc_auc_score(class_label, class_preds)
average_precision_score(class_label, class_preds)
```

That discards ranking information. AUROC and PR-AUC must use continuous
abnormal probabilities. The adapted evaluator does so at both window and
recording levels.

### Evaluation data is passed as a skorch validation split

The original final training function sets:

```python
train_split = predefined_split(window_eval_set)
```

Although the run has a fixed epoch count and no explicit early-stopping callback,
this still evaluates the designated test data every epoch and makes accidental
test-informed selection possible. NMT-4K explicitly states that its evaluation
partition must not be used for model selection. The adapted package creates an
internal validation subset only from the official training partition.

### Premature metric rounding

Several original precision/recall values are rounded to two decimals before
manual F1 computation. This introduces avoidable numerical error. The adapted
package computes all metrics at full precision and rounds only for display.

### Confusing duplicate metrics

The original result dictionary stores duplicated variants such as `accuracy`
and `accuracy2`, plus two F1 variants. The adapted output has one explicit,
machine-readable definition for each metric.

## What was retained

- The five-branch, multi-kernel architecture.
- Temporal kernels 200, 25, 13, 7, and 3.
- Seven first-block filters per branch (35 total).
- Temporal then spatial convolution in each branch.
- Mean pooling, GroupNorm, GELU, and the broader fourth block.
- Dropout 0.5029593396661691.
- 60-second inputs at 100 Hz (6,000 samples).
- AdamW, beta1 0.5, beta2 0.999, the published learning rate and weight decay.
- Balanced class-weighted negative-log-likelihood, implemented equivalently as
  weighted cross-entropy on logits.
- Cosine annealing without restarts.
- Mean window probability for the final recording probability.

The original source is MIT licensed. Its copyright and license text are retained
in this package's `LICENSE`.
