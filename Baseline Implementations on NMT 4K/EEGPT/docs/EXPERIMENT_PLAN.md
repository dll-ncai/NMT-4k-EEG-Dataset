# Controlled experiment plan

First run the supplied configuration unchanged. It is the reference experiment,
not a claim that attention is guaranteed to be best.

## Rules applying to every experiment

1. Never change the official evaluation partition or inspect it between runs.
2. Use the same internal fold for comparable ablations.
3. Give every run a unique `training.run_name`.
4. Run seeds 2026, 2027 and 2028 for the final candidates.
5. Compare recording-level validation AUROC, balanced accuracy, macro F1,
   sensitivity and specificity.
6. Change one scientific factor at a time.

## Phase A: aggregation ablation

Keep preprocessing and learning settings fixed.

| Run | `model.aggregation` | Additional change | Hypothesis |
|---|---|---|---|
| A1 | `mean_logit` | none | Stable simple baseline |
| A2 | `attention` | none | Learns sparse diagnostic intervals |
| A3 | `topk_mean` | `topk_fraction: 0.25` | Strong sparse-event prior |

Choose by internal validation AUROC, not by final evaluation.

## Phase B: fine-tuning depth

Use the best Phase A aggregation.

| Run | Frozen epochs | Unfrozen blocks |
|---|---:|---:|
| B1 | all epochs | 0 |
| B2 | 5 | 2 |
| B3 | 5 | 4 |
| B4 | 5 | all (`-1`) |

If deeper variants overfit, do not try to repair them by tuning on evaluation.
Prefer the simpler validation winner.

## Phase C: channel adapter

Compare `spatial_only` to `spatial_temporal`. The spatial adapter begins as an
identity mapping for the 19 scalp channels, while A1/A2 begin at zero. This makes
the comparison interpretable.

## Phase D: window sampling

Compare 8, 16 and 32 training windows per recording while keeping total windows
per optimizer update similar by adjusting recording batch size and accumulation.
More windows may help sparse abnormalities but increase compute.

## Phase E: preprocessing

Only after A–D are stable, test reference/filter changes. Every preprocessing
change needs a different `processed_tensor_dir`; the signature guard will reject
stale tensors. Suggested controlled comparisons:

- scalp common-average reference versus a clearly implemented linked-ear reference;
- 0.5–40 Hz versus the exact filter used by a comparison method;
- non-overlapping four seconds versus 50% overlap, while retaining record-balanced sampling.

Do not introduce per-window z-normalization without an ablation. It can remove
amplitude information relevant to low-voltage or high-amplitude abnormalities.

## Selecting a final model

After the design is fixed, run three seeds. Prefer the candidate with consistent
validation performance and clinically acceptable sensitivity/specificity, not
the single highest lucky seed. Evaluate the official 1,000 recordings once per
final seed, report mean ± standard deviation, and include recording-bootstrap
confidence intervals.

