# EEGPT–NMT v2

This is a corrected, runnable baseline for fine-tuning the released EEGPT base
model on NMT-4K EEG abnormality classification. Its primary prediction is one
label per EEG recording—not one label per four-second window.

The project deliberately separates three activities:

1. Internal training learns parameters from official training recordings.
2. Internal validation selects the epoch, aggregation behavior, and threshold.
3. Final evaluation is run once on the untouched official evaluation partition.

Read `docs/LINE_BY_LINE_GUIDE.md` before changing the scripts. It explains the
reason for every module and every nontrivial operation.

`docs/VALIDATION_REPORT.md` records which checks were completed before the ZIP
was packaged and which checkpoint/data checks must run on your workstation.

## What was corrected

- Ten-second windows are no longer resized to 1,024 samples. EEGPT receives its
  native four seconds at 256 Hz, exactly 1,024 samples.
- The official evaluation partition is never imported by the training script.
- A subject-disjoint validation fold is made only from official training data.
- All 1,000 evaluation recordings are required. Silent preprocessing failures
  block training instead of changing the benchmark cohort.
- Sampling is recording-balanced. Each training record supplies the same number
  of randomly selected windows per epoch.
- Multiple-instance learning produces one recording logit and one loss per EEG.
- Model selection uses recording-level validation AUROC.
- The operating threshold is selected only on validation recordings.
- Checkpoint loading reports parameter coverage and aborts below 95% EEGPT
  encoder coverage.
- Fine-tuning is staged: the encoder is initially frozen, then only its highest
  two blocks are opened with a much smaller learning rate.
- Checkpoints are complete and resumable after interruption.

## Folder layout

```text
EEGPT-NMT-v2/
├── checkpoints/                  released EEGPT checkpoint goes here
├── configs/default.yaml          all experimental choices
├── data/metadata.csv             attached 4,500-recording metadata
├── docs/                         audit and detailed teaching material
├── eegpt_nmt/                    implementation
├── scripts/                      double-clickable Windows launchers
├── tests/                        protocol and metric tests
└── requirements.txt
```

Large processed tensors, run checkpoints, and final outputs are created locally
and are intentionally not included in this ZIP.

## Windows setup

Open Command Prompt or PowerShell in the extracted `EEGPT-NMT-v2` folder.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Install a CUDA-enabled PyTorch build that matches your machine using the
[official PyTorch installer](https://pytorch.org/get-started/locally/), unless
your current EEGPT environment already has working CUDA PyTorch. Verify it with:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Do not install a second CPU-only `torch` package over a working CUDA build.

## Files to supply

1. Copy the released checkpoint to:

   `checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt`

2. Open `configs/default.yaml` and update only `paths.dataset_root` if the NMT
   dataset is not at `E:/Dataset/NMT-4K-EEG`.

The resolver expects a structure similar to:

```text
NMT-4K-EEG/
├── train/
│   ├── normal/edf/*.edf
│   └── abnormal/edf/*.edf
└── evaluation/
    ├── normal/edf/*.edf
    └── abnormal/edf/*.edf
```

It can also locate a uniquely named EDF underneath the configured root when an
extra extraction directory is present.

## Exact run order

Run these Windows launchers in order:

1. `scripts/00_verify_setup.bat`
2. `scripts/01_prepare_splits.bat`
3. `scripts/02_preprocess.bat`
4. `scripts/03_audit_data.bat`
5. `scripts/04_train.bat`
6. `scripts/05_final_evaluate.bat` only after choices are frozen

The equivalent commands are:

```powershell
python -m eegpt_nmt.verify_setup --config configs/default.yaml
python -m eegpt_nmt.prepare_splits --config configs/default.yaml
python -m eegpt_nmt.preprocess --config configs/default.yaml
python -m eegpt_nmt.audit_protocol --config configs/default.yaml
python -m eegpt_nmt.train --config configs/default.yaml
python -m eegpt_nmt.evaluate --config configs/default.yaml --confirm-final-test
```

The supplied metadata produces:

| Experiment partition | Normal | Abnormal | Total |
|---|---:|---:|---:|
| Internal training | 2,236 | 563 | 2,799 |
| Internal validation | 560 | 141 | 701 |
| Official evaluation | 540 | 460 | 1,000 |

The exact 2,799/701 fold sizes result from subject grouping. No patient occurs
in more than one row of this dataset, but the grouping guard remains important
for future datasets containing multiple recordings per patient.

## What preprocessing writes

Every recording becomes one `.pt` dictionary containing:

- `windows`: `[number_of_windows, 21, 1024]`, stored as float16 in microvolts
- `window_start_seconds`: start time of each window
- channel order and recording identifier
- preprocessing signature
- basic repair/rejection counts

Filtering and reference operations run in float32 before storage. Training
converts selected windows back to float32. No temporal interpolation occurs.

If even one recording fails, inspect:

`data/preprocessing_failures.csv`

The strict preprocessing command reports an incomplete cohort whenever any EDF
fails. In the old run, `mh_2024_0004036`, `mh_2024_0004037`, and
`mh_2024_0004038` were absent. Version 2.0.3 may train on the complete official
training partition while these evaluation-only files are deferred, but it will
not accept a 997-record final test.

Version 2.0.2 pins MNE 1.9.0 because MNE 1.8.0 could fail when an EDF stored a
blank optional patient height, weight, or handedness value. Upgrade an existing
environment once from the project directory:

```bat
python -m pip install --upgrade "mne==1.9.0"
```

After a failed full pass, run `scripts/02a_diagnose_failed_edfs.bat`. It writes
`data/edf_diagnostics.csv` and reports header size, available waveform records,
optional patient-metadata problems, and MNE's readable sample count. It never
changes the source EDF. A file with zero complete data records must be replaced
from a verified NMT-4K copy; preprocessing cannot reconstruct missing EEG.

After upgrading MNE and replacing any truncated EDF, run
`scripts/02b_retry_failed_preprocessing.bat`. This processes only the IDs still
listed in `preprocessing_failures.csv`, merges successes into the existing
4,497-row manifest, and rechecks the required 3,500/1,000 cohort. Do not rerun
the three-hour full pass.

If only official-evaluation EDFs are missing while all 3,500 training EDFs are
complete, version 2.0.3 can train before those files are restored. The explicit
command-line setting is already included in `scripts/03_audit_data.bat` and
`scripts/04_train.bat`:

```bat
--allow-incomplete-evaluation-for-training
```

`scripts/03_audit_data.bat` then verifies the complete 3,500-record training
cohort, patient separation, and split assignments before allowing training.
It prints a warning that the evaluation cohort is 997/1,000. This option does
not weaken `evaluate.py`: `scripts/05_final_evaluate.bat` still requires exactly
1,000 evaluation recordings. Restore and preprocess the three EDFs before
reporting final test metrics.

Version 2.0.3 also records the failing stage, resolved EDF path, file size, and
traceback. It prints each preprocessing exception immediately and stops after
ten consecutive failures, preventing a systematic path, channel, or API issue
from silently iterating over all 4,500 rows.

## Memory behavior

The default training batch is two recordings × eight windows = sixteen EEGPT
inputs. If CUDA memory is insufficient, first reduce `recording_batch_size` from
2 to 1 and increase `gradient_accumulation_steps` from 4 to 8. Do not reduce the
number of EEGPT encoder blocks.

Validation and final evaluation encode long recordings in chunks of sixteen
windows, then aggregate their small 512-dimensional features. This avoids
placing every window of a long recording on the GPU simultaneously.

## Interrupting and resuming training

The latest complete epoch is written atomically to:

`outputs/runs/eegpt_nmt_v2_attention_seed2026/last.pt`

To resume, edit `configs/default.yaml`:

```yaml
training:
  resume_checkpoint: "outputs/runs/eegpt_nmt_v2_attention_seed2026/last.pt"
```

Then run `scripts/04_train.bat` again. Keep the same `run_name`, model settings,
epoch count, and learning-rate schedule. After a completed run, set this field
back to `null` before starting a separate experiment.

## Outputs to report

Training writes:

- `best.pt`: epoch selected by internal validation AUROC
- `last.pt`: latest resumable epoch
- `history.csv`: recording-level validation history
- `pretrained_load_report.json`: exact EEGPT restoration coverage
- per-epoch and final all-window validation predictions
- `config_used.yaml`: immutable record of the run settings

Final evaluation writes:

- `recording_predictions.csv`
- `final_metrics.json`
- recording-level confusion matrix, ROC curve, and PR curve
- 95% paired-bootstrap confidence intervals

These files are placed under `outputs/final_evaluation/<run_name>/`, so results
from different seeds do not overwrite one another.

Report the metrics under `metrics_at_validation_threshold` as the primary result.
Also report the threshold-0.5 result for transparency. Do not search for a new
threshold after viewing official evaluation labels.

## Recommended experiment sequence

Run the supplied attention-MIL configuration first. Afterwards, follow
`docs/EXPERIMENT_PLAN.md`. Change the `run_name` for every experiment and use
three seeds before drawing conclusions.
