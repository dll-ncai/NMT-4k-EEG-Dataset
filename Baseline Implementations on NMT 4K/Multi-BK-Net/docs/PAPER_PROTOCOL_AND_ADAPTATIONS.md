# Paper protocol and NMT-4K adaptation

## What the paper actually did

### Data used for model development

The final Multi-BK-Net was developed on TUABCOMB, the union of the TUAB and
TUABEXB training partitions:

- 10,707 recordings
- 5,386 non-pathological
- 5,321 pathological
- 8,549 patients

The TUAB and TUABEXB predefined evaluation partitions remained final test sets.

### Preprocessing

For every recording, the authors:

1. Selected 21 common positions: A1, A2, C3, C4, Cz, F3, F4, F7, F8, Fp1,
   Fp2, Fz, O1, O2, P3, P4, Pz, T3, T4, T5, and T6.
2. Discarded the first 60 seconds.
3. Kept at most the next 20 minutes.
4. Converted signals to microvolts.
5. Clipped them to ±800 µV.
6. Applied common-average reference.
7. Resampled to 100 Hz.
8. Created non-overlapping 60-second windows (6,000 samples).
9. Assigned the EDF's recording label to every window.

There was no band-pass filter, notch filter, ICA, artifact rejection, or
per-window z-normalization in this described pipeline.

### Architecture

Input shape was `(batch, 21, 6000)`. The first block had five parallel branches.
Each branch applied:

1. A temporal convolution with one of five kernels: 200, 25, 13, 7, or 3.
2. A spatial convolution spanning all electrodes.
3. Group normalization.
4. GELU.
5. Mean pooling of length 50 and stride 15.

Each branch used seven temporal filters, producing 35 concatenated features.
Three later convolution/mean-pooling blocks increased the filters to 70, 140,
and 280. The model ended with a two-class convolutional classifier and softmax.
The reported 21-channel model contains 1,038,683 trainable parameters.

### Training

- Trial-wise training: one complete 60-second window produces one loss.
- Weighted negative-log-likelihood.
- AdamW.
- Learning rate: 0.0031414364096615.
- Betas: (0.5, 0.999).
- Weight decay: 1.8397405899531204e-05.
- Batch size: 64.
- Epochs: 42.
- Cosine annealing without restarts.
- Xavier weight initialization and zero biases as described in the paper.
- Ten independent final training runs for mean and standard deviation.

### Hyperparameter selection

The authors did not guess the final values. They ran 100 Optuna trials with
five-fold stratified, patient-grouped cross-validation on TUABCOMB and optimized
validation accuracy and sensitivity. The top configurations were repeated.
This search is much more expensive than final training and is intentionally not
repeated by the laptop package. The published selected configuration is used.

### Final prediction and metrics

Softmax probabilities were averaged across all windows of an EDF. The recording
was classified from the mean probability. The paper reported recording-level
accuracy, balanced accuracy, sensitivity, specificity, F2, ROC-AUC, and PR-AUC.

## What the paper's NMT result means

The paper also tested TUH-trained models on an older NMT cohort without training
on NMT. It describes 2,417 recordings, with 2,232 training recordings and only
185 evaluation recordings (95 normal, 90 pathological). Multi-BK-Net obtained
67.32% mean accuracy in that cross-institutional transfer experiment.

The current NMT-4K release is a different, expanded benchmark:

- 4,500 unique-subject recordings
- 3,500 official training recordings (2,796 normal, 704 abnormal)
- 1,000 official evaluation recordings (540 normal, 460 abnormal)
- 19 scalp channels
- 200 Hz native sampling rate

Training Multi-BK-Net on the current NMT-4K training set and evaluating it on
the current evaluation set is an in-domain NMT experiment. It is not directly
comparable to the paper's TUH-to-old-NMT result.

## Necessary adaptations

| Change | Reason | Scientific effect |
| --- | --- | --- |
| 21 to 19 spatial channels | Current NMT-4K documents 19 scalp channels and linked-ear acquisition reference | The spatial convolution spans 19 inputs; temporal architecture is unchanged |
| CAR across 19 scalp channels | A1/A2 are acquisition references, not released scalp channels | Closest defined application of the paper's CAR step to the current release |
| Native PyTorch | Legacy Braindecode 0.7/Linux code is not a stable Windows path | Mathematical layers and hyperparameters are retained |
| Micro-batch 16 x accumulation 4 | Protect 8 GB laptop VRAM | Effective batch 64; GroupNorm avoids batch-statistic changes |
| AMP | Lower activation memory and improve RTX throughput | Small numerical differences from full FP32 are possible |
| Internal 80/20 validation | Evaluation must remain untouched | Checkpoint/epoch choice is scientifically valid |
| Retrain on all successfully preprocessed training recordings | Recover training data held out during development | Final evaluation model uses every training EDF that can provide a real 60-second input |
| Last complete 60 seconds for unusually short EDFs | Some current NMT-4K EDFs cannot satisfy both a 60-second lead-in crop and a complete 60-second input | No samples are padded or fabricated; every use is written to `short_recording_adaptations.csv` and must be reported |
| Probability-based AUROC/PR-AUC | Original released evaluator used hard predictions | Correct ranking metrics |
| One seed by default | Ten full runs are expensive on a laptop | A single result lacks the paper's run-to-run mean and SD |

## Why the evaluation split remains untouched

NMT-4K's repository explicitly states that the predefined evaluation partition
must not be used for model selection or hyperparameter tuning. This package does
not even construct an evaluation data loader during training. The complete
evaluation partition is checked only for preprocessing completeness. Its signal
arrays and labels are loaded by `05_evaluate.ps1` after final training.

## Class imbalance

The training partition has approximately four normal recordings for every
abnormal recording. The package reproduces the original inverse-frequency
window weighting:

```text
weight(class c) = number of training windows / (2 * windows in class c)
```

It does not combine a weighted sampler with weighted loss, which would
double-correct the imbalance.

## Recommended reporting statement

Use wording similar to:

> Multi-BK-Net was trained on the predefined NMT-4K training partition and
> evaluated once on the held-out evaluation partition. The first 60 seconds
> were removed, up to 20 subsequent minutes were retained, signals were clipped
> to ±800 µV, common-average referenced over 19 scalp channels, resampled to
> 100 Hz, and divided into non-overlapping 60-second windows. Recording
> probabilities were obtained by averaging window probabilities. Model epoch
> selection used an internal stratified split derived only from the training
> partition, followed by retraining on the successfully preprocessed training
> recordings. For EDFs too short to retain a full window after the planned
> lead-in crop, the last complete 60 seconds of real signal were used; no signal
> padding was applied.

Also report PyTorch/MNE versions, seed, effective batch size, use of AMP, the
number of successfully processed/excluded recordings, threshold, and both the
confusion matrix and continuous-probability AUROC.
