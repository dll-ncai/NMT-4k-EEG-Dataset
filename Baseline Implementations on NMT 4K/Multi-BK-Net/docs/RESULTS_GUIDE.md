# Results guide

## Which results are primary?

Use `evaluation/recording_metrics.json` as the primary result. The task label is
assigned to the complete EDF, and the paper's protocol first averages window
probabilities within each recording. Window-level values are useful diagnostics,
but they are not independent patient-level observations and should not replace
the recording-level benchmark.

## Requested metrics

For the abnormal class as positive:

- Accuracy = `(TP + TN) / (TP + TN + FP + FN)`
- F1 = `2 * precision * sensitivity / (precision + sensitivity)`
- Sensitivity = `TP / (TP + FN)`
- Specificity = `TN / (TN + FP)`
- AUROC = area under the sensitivity-versus-false-positive-rate curve while the
  decision threshold varies

The package also saves balanced accuracy, F2, and PR-AUC because the training
set is imbalanced and the paper emphasizes abnormal-case recall.

## Output files

| File | Purpose |
| --- | --- |
| `recording_metrics.json` | Primary exact metrics and confusion counts |
| `recording_metrics.csv` | Same metrics in a spreadsheet-friendly row |
| `recording_predictions.csv` | EDF ID, true class, mean abnormal probability, predicted class, and number of windows |
| `confusion_matrix.csv` | TN, FP, FN, and TP matrix |
| `roc_pr_curves.png` | Recording-level ROC and precision-recall plots |
| `roc_curve.csv` | Raw ROC points |
| `precision_recall_curve.csv` | Raw PR points |
| `window_metrics.json` | Secondary window-level metrics |
| `window_predictions.csv` | Secondary probability for every 60-second window |
| `evaluation_summary.json` | Runtime, GPU memory, checkpoint epoch, and both metric levels |

## AUROC correctness

AUROC must be computed from `abnormal_probability`, not `predicted_label`.
Thresholding at 0.5 first reduces the curve to a few points and produces a value
that is not the intended ranking AUROC. The adapted evaluator uses the continuous
mean probability for every EDF.

## Interpreting sensitivity and specificity

- Low sensitivity means too many abnormal EDFs are classified as normal (false
  negatives).
- Low specificity means too many normal EDFs are classified as abnormal (false
  positives).
- Because evaluation has 540 normal and 460 abnormal recordings, accuracy is
  interpretable but should still be reported beside both sensitivity and
  specificity.

## Threshold policy

The default threshold is 0.5 and must be fixed before looking at final
evaluation labels. If a different threshold is clinically desired, select it
using only internal validation predictions, document the rule, freeze it, and
then evaluate once. Do not optimize a threshold on the 1,000-recording
evaluation partition.

## Comparing runs

One seed gives one point estimate. The paper repeated training ten times and
reported mean ± standard deviation. For a publication-quality stochasticity
analysis, repeat the complete training with multiple fixed seeds, retain the
same preprocessing and internal-split policy, and summarize recording-level
metrics across runs. Do not average window-level metrics as a substitute.
