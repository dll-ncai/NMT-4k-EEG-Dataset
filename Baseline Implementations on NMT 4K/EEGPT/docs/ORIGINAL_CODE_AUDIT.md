# Audit of the previous implementation

## 1. The held-out evaluation partition became validation data

The former `finetune_nmt.py` constructed `NMTDataset(..., split="evaluation")`
and evaluated it after every epoch. It selected `best_eegpt_model.pt` by that
partition's accuracy and stopped training after two non-improving epochs.

This means the final test cohort influenced:

- the selected epoch;
- the decision to stop training;
- every architecture or optimizer choice made after viewing results.

The former evaluation script then swept 81 thresholds on the same labels. That
is a second form of test tuning. V2 creates validation only from official train,
and the train module never constructs an evaluation dataset.

## 2. The published number was a window score, not a recording score

The old confusion matrix contains 43,355 TN, 11,120 FP, 11,642 FN and 33,764 TP:
99,881 predictions. NMT-4K evaluation contains 1,000 EEG recordings, so the
77.21% accuracy and 0.8488 AUROC describe windows. Long EEGs contribute more
predictions than short EEGs.

V2 indexes one row per recording, returns a bag of windows, and computes one
recording logit before the loss and metrics.

## 3. Ten seconds were mislabeled as four seconds

The old preprocessor wrote `[21, 2560]` ten-second windows at 256 Hz. The model's
`temporal_interpolation` resized 2,560 samples to 1,024 using nearest-neighbor
interpolation. EEGPT still interpreted those 1,024 positions as four seconds.

This compresses physiological time by 2.5×, shifts frequency content, duplicates
or drops samples without an anti-aliasing filter, and breaks the pretrained
patch semantics. V2 produces native `[21, 1024]` four-second segments directly.
Its model throws an error for every other time length.

## 4. Every abnormal-record window was treated as independently abnormal

Clinical abnormality labels describe entire reports/recordings. An abnormal EEG
may contain many normal-looking intervals or sparse discharges. Independent
window cross-entropy therefore introduces false positive labels during training.

V2 samples several windows from the same recording and applies one
recording-level BCE loss after attention aggregation. The network is allowed to
decide which intervals support the recording diagnosis.

## 5. The implementation was not the TUAB-specific downstream architecture

The previous model used the generic EEGPT classifier: encoder, eight-layer
reconstructor, mean token pooling, and two-class head. The TUAB-specific code has
a spatial/temporal convolutional adapter and uses the EEGPT encoder for the
downstream representation.

V2 removes the reconstructor. It uses a 21→19 spatial adapter, an identity-safe
depthwise temporal residual, the eight-layer EEGPT encoder, and an MIL head.

## 6. Checkpoint loading could silently accept a weak match

The old loader stripped one string prefix, loaded matching shapes, and printed
only a tensor count. It did not prove that the encoder was substantially
pretrained. A wrong checkpoint could leave much of the backbone random.

V2 maps only known prefixes, verifies shapes, measures scalar-parameter coverage
inside the encoder, writes all missing keys, and aborts below 95% coverage.

## 7. Longer records had larger training weight

The old dataset had one item per window and shuffled all windows. A recording
with 126 windows produced 42 times as many updates as a three-window recording.

V2 samples records uniformly and chooses exactly eight windows from each record
per epoch. The chosen windows change reproducibly with the epoch number.

## 8. Three test recordings disappeared silently

The metadata has 1,000 official evaluation records, but the old processed index
contained 997. Missing normal recordings were `mh_2024_0004036` through
`mh_2024_0004038`. Only one example error was printed.

V2 writes every failure with its exception and split, checks the expected 3,500
and 1,000 counts, and blocks both training and evaluation on mismatch.

## 9. The optimization schedule was too abrupt

The former run trained encoder, reconstructor, adapter and head together at
`1e-4`, used only five epochs, had no warm-up, and stopped with patience two.

V2 trains the adapter/MIL head for five frozen-backbone epochs, then opens the
last two EEGPT blocks at `2e-5` with layer-wise decay. The head uses `5e-4`, the
schedule has five warm-up epochs followed by cosine decay, clipping is 1.0, and
patience is ten.

## 10. Reproducibility and recovery were incomplete

The old run did not persist the optimizer, scheduler, AMP scaler, threshold,
configuration, or random states. It could not faithfully resume after a reboot.

V2 stores all of these in `last.pt` and saves the checkpoint atomically. It also
records the precise data, model and learning choices in `config_used.yaml`.

