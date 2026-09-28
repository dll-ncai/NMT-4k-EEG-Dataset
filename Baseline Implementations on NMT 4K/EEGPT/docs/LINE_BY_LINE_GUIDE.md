# Line-by-line reading companion

This guide follows the execution order rather than alphabetical file order.
Keep the relevant source file open beside it. Comments in the source explain
local syntax; this document explains why each operation exists and what would
break if it were removed or moved.

## 1. `configs/default.yaml`

The YAML file is the experiment's scientific record. A number buried directly
inside Python is easy to forget; a number here is copied into every checkpoint.

### `paths`

- `dataset_root` is the only machine-specific dataset setting. Forward slashes
  work on Windows and avoid backslash escaping.
- `metadata_csv` identifies the original 4,500 rows. This file defines labels
  and official partitions; it is not rewritten.
- `recording_splits_csv` is created by `prepare_splits.py`. It adds only the
  internal train/validation decision.
- `processed_tensor_dir` contains one `.pt` file per recording. This replaces
  the old logical design of one training item per window.
- `processed_manifest_csv` is a light index of those tensors. The training code
  reads this index rather than the original metadata.
- `pretrained_checkpoint` is the released EEGPT base checkpoint, not a previous
  fine-tuned checkpoint.
- `runs_dir` and `evaluation_output_dir` keep validation artifacts separate from
  the final test report.

### `split`

- The expected counts are assertions. If a path error removes three files, the
  program must fail rather than quietly describe a different cohort.
- Five folds make one validation fold close to 20% of official training.
- `validation_fold: 0` fixes which fold is used; changing it creates a different
  experiment.
- The seed controls group-fold shuffling.
- Hospital is combined with the label for stratification when every composite
  stratum is large enough. This keeps the hospital mixture similar without
  giving hospital information to the model.

### `preprocessing`

- 256 Hz × 4 seconds equals 1,024. All three entries are stated explicitly so a
  future edit cannot accidentally change only one.
- A 0.5–40 Hz zero-phase FIR filter matches the broad NMT benchmark passband and
  removes drift/high-frequency contamination without time shifting events.
- `stride_seconds: 4` means no overlap. Overlap can be tested later because the
  recording-balanced loader prevents overlapped long recordings dominating.
- No 21-minute truncation is applied. The model sees uniformly sampled evidence
  across the whole EEG.
- Artifact thresholds are null because automatic amplitude rejection can remove
  true pathology. When enabled, they are documented experimental factors.
- `strict_complete_cohort` turns missing files into a stop condition.
- `overwrite` protects expensive compatible tensors. The saved preprocessing
  signature prevents accidentally reusing incompatible tensors.

### `model`

- `spatial_temporal` enables the small TUAB-inspired adapter. It still keeps the
  EEGPT input at four seconds.
- Kernel 15 is odd, so symmetric padding retains 1,024 time points.
- Dropout applies only to newly trained components; pretrained EEGPT starts
  unchanged.
- Attention is the recording aggregator. It receives no window labels.
- `minimum_encoder_coverage` makes checkpoint compatibility measurable.

### `training`

- `run_name` isolates files for this exact experiment.
- `resume_checkpoint` is null for a new run and points to `last.pt` after an
  interruption.
- Deterministic mode trades a little speed for repeatability.
- Forty epochs is a ceiling; patience ten usually stops earlier.
- Two recordings × eight windows produces sixteen EEGPT inputs per forward
  pass. Gradient accumulation four gives an effective eight-recording update.
- Per-epoch validation uses 32 deterministic windows for speed. Once the best
  epoch is known, `calibration_windows_per_recording: 0` recalibrates the
  threshold on every validation window exactly once.
- The adapter/head rate is 25 times the top encoder rate because those layers
  start new while EEGPT already contains useful representations.
- Layer decay reduces learning rates further toward lower encoder blocks.
- The first five epochs train only the new components. Two high EEGPT blocks
  then open; the lower six remain stable.
- AUROC chooses the checkpoint without needing a classification threshold.
- Balanced accuracy chooses the operating threshold using validation only.

### `evaluation`

Zero windows means all windows. Batch size one is required because different
recordings have different lengths. Encoding still happens in chunks, so the GPU
never holds a complete long recording's raw windows at once.

## 2. `constants.py`

`EEGPT_SCALP_CHANNELS` gives the exact order used both by the spatial adapter
output and EEGPT's channel-embedding lookup. If these orders differ, the signal
from F3 could receive the embedding for another electrode.

`NMT_INPUT_CHANNELS` appends A1 and A2 to the 19 scalp channels. The model starts
by passing scalp channels through and assigning zero weight to A1/A2, but can
learn to use the auricular channels.

`CHANNEL_ALIASES` maps historical names T3/T4/T5/T6 and M1/M2. The mapping is
made before checking missing channels.

The three EEGPT temporal constants make the pretraining contract executable.
The model checks them at runtime rather than merely documenting them.

## 3. `config.py`

- `Path(__file__).resolve().parents[1]` finds the project directory from the
  installed package. Commands therefore behave identically regardless of the
  shell's starting folder.
- `load_config` resolves the supplied YAML path, rejects a missing or non-mapping
  file, deep-copies it, and attaches two private absolute paths. The private
  keys begin with `_` so they are omitted from experiment snapshots.
- `resolve_path` leaves absolute paths absolute and anchors relative paths to
  the project. This replaces old `os.path.join` behavior that could depend on
  the current working directory or store Windows separators in CSV files.
- `save_config_snapshot` strips helper keys and writes human-readable YAML next
  to the checkpoint.
- `require_sections` turns an accidentally deleted YAML section into an early,
  specific exception.

## 4. `protocol.py` and `prepare_splits.py`

### Metadata canonicalization

`_first_existing_column` supports both your current headings and common future
variants. It does not guess from values.

`_parse_binary_label` accepts 0/1 or Normal/Abnormal and rejects everything
else. Treating an unknown label as normal would be a silent scientific error.

`_parse_official_split` normalizes spelling while retaining only `train` and
`evaluation`. There is intentionally no pre-existing `validation` accepted from
the NMT metadata because validation must be derived audibly.

`canonicalize_metadata` creates the exact columns used downstream:

1. Copy and strip recording IDs.
2. Use patient ID when available; otherwise a recording is conservatively its
   own group.
3. Parse labels and official partitions.
4. Preserve the old file path only as one path candidate.
5. Copy hospital, gender, age and year when present.
6. Reject empty and duplicate recording IDs.

### Partition checks

`validate_official_partition` counts rows before splitting. It also intersects
patient sets across official train and evaluation. This catches leakage before
any model sees data.

`create_experiment_splits` initially labels every row `evaluation`. It then
selects only rows whose official partition is train. This default is a safety
device: an evaluation record is never temporarily treated as train.

The composite stratification key is `label__hospital`. If a stratum is too
small for five folds, code falls back to label rather than producing an invalid
split. `StratifiedGroupKFold` distributes patient groups, not isolated rows.

Only the chosen fold's official-training IDs become `validation`; the rest
become `internal_train`. Two final assertions recheck patient separation and
that all official evaluation values remained untouched.

`prepare_splits.main` loads settings, calls this protocol, saves the full rows,
saves an easy count table, and writes a machine-readable report stating that
evaluation was not used.

## 5. `preprocess.py`

### Channel cleanup and path resolution

`clean_channel_name` removes an optional EEG prefix and common reference suffix,
uppercases the result, and then applies the alias map. Regex anchors ensure that
letters inside a legitimate channel name are not removed.

`resolve_edf_path` first accepts a still-valid metadata path. If the project
moved to another machine, it constructs paths from dataset root, official
partition, label, and recording ID. A recursive fallback accepts exactly one
match. Multiple matches fail because silently choosing one could attach the
wrong signal to a label.

### Non-finite repair

MNE filtering cannot safely propagate NaN or infinity. `_replace_nonfinite`
counts them and replaces them with that channel's finite median. Repaired counts
are saved. This is superior to changing all problems to zero without an audit,
but a high count should still prompt EDF inspection.

### Native EEGPT preprocessing

`preprocess_recording` computes samples from seconds and immediately asserts
256 Hz/1,024. The old interpolation bug cannot reappear downstream.

The EDF is preloaded because filtering and resampling require sample access.
Original frequency is saved before resampling. Normalized duplicate required
channels are rejected. Only the 21 retained channels are renamed and picked in
the constants' exact order.

Crop logic removes nothing by default. If a future experiment defines a start
or maximum duration, it is recorded in the signature.

The processing order is:

1. Repair non-finite samples.
2. Apply a zero-phase 0.5–40 Hz FIR filter.
3. Subtract the mean of the 19 scalp electrodes from all retained EEG channels.
4. Resample with MNE's anti-aliasing implementation to 256 Hz.
5. Convert volts to microvolts explicitly.
6. Generate starting samples separated by the configured stride.
7. Slice exact 1,024-sample arrays.

Optional artifact rules calculate per-window maxima or average channel standard
deviation. The boolean `keep` array ensures starts and windows remain aligned.

Windows are cast to float16 only after signal processing. The smaller storage
type does not contaminate filtering calculations. Start seconds remain float32.

### Cohort-safe persistence

The signature stores channels, reference, frequency, duration, stride, filter,
crop and units. `_load_existing_count` permits resume only when this signature
matches.

Each new recording is saved to a temporary filename and then atomically moved.
A power interruption therefore leaves either the previous complete tensor or no
new tensor, not a partially serialized file.

The manifest adds tensor path and window count to one recording row. Every
exception becomes one row in `preprocessing_failures.csv`. After all attempts,
the script verifies official train/evaluation counts and raises if strict mode
detects any failure.

## 6. `data.py`

`resolve_tensor_path` interprets relative manifest paths against the project.
POSIX separators remain readable on Windows through `pathlib`.

`load_recording_tensor` uses PyTorch's restricted `weights_only` deserializer,
extracts windows/start times, converts signals to float32, and enforces the
three-dimensional `[N,21,1024]` contract.

`RecordingBagDataset` filters a recording-level manifest by exactly one
experiment split. `set_epoch` changes random training selections without making
them irreproducible.

`_choose_indices` has three paths:

- zero requested windows returns all indices for final aggregation;
- a short record returns all available windows;
- training samples without replacement using seed + epoch + row index;
- validation uses evenly spaced deterministic indices.

`__getitem__` loads one tensor once, applies selected indices, and pads only a
record shorter than the fixed bag size. The mask is false on padding, so pooling
and attention ignore it. The returned label is one float because
`BCEWithLogitsLoss` expects a binary target for one recording logit.

`recording_class_counts` counts labels from recording rows, not windows. Its
negative/positive ratio becomes `pos_weight` in training.

## 7. `eegpt_backbone.py`

This file is the EEGPT implementation inherited from the supplied project and
official model code. It is kept structurally compatible with the released
checkpoint. The downstream v2 code uses only these relevant components:

- `CHANNEL_DICT` maps named electrodes to pretrained embedding rows.
- `PatchEmbed` uses a convolution of width 64 and stride 64. A 1,024-sample
  input therefore creates 16 temporal patches.
- `Attention`, `MLP`, and `Block` implement each transformer layer.
- `EEGTransformer` creates eight 512-dimensional layers with eight attention
  heads and four summary tokens per temporal patch.
- `prepare_chan_ids` converts the 19 output channel names into pretrained IDs.
- `forward` patchifies the signal, adds channel embeddings, processes the
  electrode tokens, retains summary tokens, and returns
  `[batch,16,4,512]`.
- `Conv1dWithConstraint` bounds the spatial adapter's channel weight norms.

The reconstructor and legacy generic classifier remain in this upstream file
for checkpoint/source traceability. The old file also retains an inert demo
inside its `if __name__ == "__main__"` block, but no v2 module imports or calls
that classifier during the pipeline.

## 8. `model.py`

### `EEGPTWindowEncoder`

The constructor validates the two supported adapters and requires an odd
temporal kernel. `chan_conv` maps 21 inputs to 19. Its weights are immediately
zeroed, and the diagonal for the first 19 inputs is set to one. The initial
mapping is therefore exactly the 19 scalp signals; A1/A2 start unused but remain
learnable.

For the temporal variant, a grouped convolution applies one temporal filter per
adapted channel. Batch norm, GELU and dropout follow. The entire branch is added
residually and multiplied by a trainable scalar initialized to zero. Thus an
untrained random temporal filter cannot corrupt the first forward pass.

`target_encoder` exactly matches EEGPT large's downstream encoder dimensions
and preserves its checkpoint key name. `chans_id` is a nonpersistent buffer: it
moves with the GPU but is derived rather than saved as learned state.

`forward` rejects every shape except `[B,21,1024]`. Adapted samples pass through
EEGPT. The returned 16×4 summary tokens are averaged into one 512-dimensional
window feature, normalized, and regularized.

### `EEGPTMILClassifier`

`window_head` maps each feature to an abnormality logit. Logits are used rather
than probabilities because BCE-with-logits combines sigmoid and logarithms
more stably.

The attention option learns 512→128→1 scores. `Tanh` permits positive and
negative hidden evidence. Softmax normalizes valid window scores within each
recording.

`_encode_in_chunks` limits how many raw windows enter EEGPT simultaneously but
concatenates all resulting features before aggregation. This changes memory,
not mathematics.

`forward` validates the mask, flattens only the first two dimensions, and
encodes only valid windows. Padded zeros therefore cannot alter adapter batch
normalization. It scatters valid features back into the record/bag layout,
calculates window logits, and then:

- averages valid logits for `mean_logit`;
- averages the highest configured fraction for `topk_mean`; or
- attention-weights features and applies the same head for `attention`.

It returns recording and window logits. Training loss uses only recording
logits; window values are available for later interpretation, not pseudo-labels.

`set_encoder_trainability` first freezes everything. Zero leaves it frozen,
negative opens it all, and a positive count opens only the highest blocks plus
the final normalization.

## 9. `checkpoint.py`

`_extract_state_dict` handles Lightning's `state_dict`, conventional
`model_state_dict`, and plain tensor dictionaries. Non-tensor metadata is never
mistaken for a weight.

`_key_variants` removes only known leading wrappers. Unlike unrestricted string
replacement, it cannot alter a legitimate substring in the middle of a layer
name. It maps `encoder` to EEGPT's `target_encoder` and accounts for v2's
`window_encoder` nesting.

`load_pretrained_eegpt` matches both name and shape. It counts scalar elements
inside the encoder, not merely tensors: loading ten tiny biases cannot appear
equivalent to loading ten large matrices. Missing keys and the first shape
mismatches are written to JSON. The new adapter and heads should be missing;
the pretrained encoder should exceed 95% coverage.

`atomic_torch_save` serializes to `.tmp` and replaces the destination only after
success. `load_experiment_checkpoint` later loads v2 strictly so an architecture
change cannot be silently evaluated with partial weights.

## 10. `train.py`

`set_reproducibility` covers Python, NumPy, PyTorch CPU and all CUDA devices.
cuDNN benchmark is disabled only in deterministic mode.

`model_options` creates a small exact dictionary saved in the checkpoint. Final
evaluation reconstructs the architecture from this dictionary rather than
assuming the current YAML still matches.

### Optimizer and schedule

`build_optimizer` distinguishes pretrained encoder parameters from new ones.
For an encoder block, `depth_from_output` is zero at the highest block and grows
toward the input. Multiplying by `layer_decay ** depth` makes lower learning
rates progressively smaller. One-dimensional parameters, biases, and norms get
zero weight decay; matrix/convolution weights get the configured decay.

All parameters enter the optimizer even while frozen. When `requires_grad`
becomes true at epoch six, the optimizer can update them without being rebuilt
or losing head momentum.

`build_scheduler` linearly warms from a small multiplier, then follows cosine
decay to a nonzero minimum ratio. It advances after optimizer updates, not after
every accumulated mini-batch.

### Validation inference

`predict_recordings` switches to evaluation mode and disables gradients. It
moves only windows and masks to GPU, applies AMP when CUDA is available,
sigmoids recording logits, and writes one row per ID. No threshold is needed to
calculate AUROC.

### Main training sequence

1. Load paths, seed everything and choose CUDA/CPU.
2. Refuse an existing non-resume run directory to prevent accidental overwrite.
3. Build internal train and validation datasets only.
4. Instantiate the model and either audit-load EEGPT or strictly resume v2.
5. Freeze/unfreeze the encoder according to the next epoch.
6. Calculate positive weight from training recordings.
7. Build optimizer, schedule and AMP scaler; restore them on resume.
8. For each epoch, change sampled training windows deterministically.
9. Compute one BCE loss per recording, divide for accumulation, backpropagate,
   unscale gradients, clip to 1.0, update optimizer/scaler, and step schedule.
10. Predict validation recordings and select a validation threshold.
11. Select the best epoch by validation AUROC, save history, `last.pt`, and
    `best.pt`, then apply patience.
12. Reload the best epoch and make one all-window validation pass. This finalizes
    its operating threshold with the same aggregation used on the test set.

At no point does `train.py` request `experiment_split="evaluation"`.

### Why `pos_weight` is recording-level

With 2,236 normal and 563 abnormal internal-training recordings, the positive
weight is about 3.97. This balances the contribution of the two recording
classes. Computing it from window counts would again let duration determine the
loss.

## 11. `metrics.py`

`binary_metrics` thresholds one probability per recording and requests
confusion-matrix labels `[0,1]`, so all four cells retain stable meaning. It
reports macro F1 and positive-class F1 separately; this avoids comparing two
papers that use the word “F1” for different averages.

AUROC and PR-AUC use probabilities and are computed only when both classes are
present. `select_threshold` scans 0.05–0.95 on validation. Ties prefer a value
closer to 0.5, making the choice deterministic and less extreme.

`bootstrap_confidence_intervals` samples recording indices with replacement and
uses the same sampled indices for labels and predictions. Resamples missing a
class are skipped because their AUROC is undefined. Percentiles 2.5 and 97.5
form the paired 95% intervals.

## 12. `evaluate.py`

The command requires `--confirm-final-test`; omission stops immediately. This is
a behavioral guard against running the official test during development.

The script loads architecture options from `best.pt`, restores every model key
strictly, and constructs only the `evaluation` dataset. It asserts exactly 1,000
recordings and uses the validation threshold stored in the checkpoint.

It reports both the validation-selected operating point and 0.5. It never calls
`select_threshold`. Predictions merge hospital/demographic metadata for later
error analysis but those attributes never enter the model.

Plots and JSON explicitly say recording-level. The JSON records checkpoint
epoch and threshold source so a table cannot later lose this provenance.

## 13. Safe ways to modify the project

- Change `run_name` before every new training configuration.
- Change one YAML factor at a time and keep the same split seed/fold.
- If preprocessing changes, also change `processed_tensor_dir` or set overwrite
  only after deliberately deciding to replace old tensors.
- Keep input shape 21×1,024. A ten-second experiment needs a genuinely different
  TUAB-compatible architecture; it must not use interpolation to impersonate
  four seconds.
- Keep evaluation out of training, early stopping, calibration, and ablation
  selection.
- Compare methods at recording level and state whether F1 is macro or positive.

## 14. Mistakes the assertions are designed to catch

| Mistake | Guard that stops it |
|---|---|
| Missing/extra metadata rows | Expected official counts |
| Same patient in two partitions | Patient-set intersections |
| Three missing evaluation EDFs | Strict preprocessing cohort |
| Reusing old 10-second tensors | Tensor shape and preprocessing signature |
| Wrong checkpoint | Encoder parameter-coverage threshold |
| Editing architecture before evaluation | Strict experiment checkpoint load |
| Tuning threshold on test | Evaluation contains no selection function |
| Accidental second run overwrite | Existing run-directory guard |
| Laptop interruption during save | Atomic `last.pt` replacement |
