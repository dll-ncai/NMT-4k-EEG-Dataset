# Package validation report

The following checks were completed before packaging.

## Python 3.13 dependency resolution (2026-08-29)

The Windows environment was dry-run resolved for CPython 3.13 x64 without
installing packages. The CUDA 12.8 index selected
`torch-2.11.0+cu128-cp313-cp313-win_amd64.whl`, and every dependency in
`requirements.txt` resolved to a compatible Windows x64 wheel. This corrects
the overly conservative Python 3.13 rejection in package version 1.0.0.

## Static validation

- Every Python module passed bytecode compilation.
- Ruff static analysis passed with no findings.
- All PowerShell scripts use project-relative `.venv` and module paths; no
  `/home/...` or machine-specific source paths remain.
- The only configured data path is the user's requested
  `G:\Dataset\NMT-4K-EEG` in `config.yaml`.

## Unit tests

Sixteen tests are supplied. The original eleven-test suite passed, and five
additional pure crop-boundary tests cover the short-recording adaptation:

- common EDF channel aliases and legacy/modern temporal channel mapping
- ordered channel selection
- blocking behavior for a missing channel
- exact non-overlapping windows
- end-aligned final window behavior
- short-recording rejection
- recording probability aggregation
- accuracy/F1/sensitivity/specificity/AUROC correctness
- 19-channel model input/output shape
- final feature length of three samples
- exact 21-channel paper parameter count
- standard 60-second lead-in crop preservation
- last-complete-window selection for a 60-120 second EDF
- exact 60-second EDF handling without padding
- rejection below 60 seconds
- strict-policy backward compatibility

## Architecture parity

| Model | Trainable parameters | Input | Output | Final feature samples |
| --- | ---: | --- | --- | ---: |
| Paper/TUH channel configuration | 1,038,683 | `(B, 21, 6000)` | `(B, 2)` | 3 |
| NMT-4K adaptation | 1,038,193 | `(B, 19, 6000)` | `(B, 2)` | 3 |

The 21-channel count exactly equals the count reported by the paper and original
repository. The 490-parameter reduction is solely due to the two omitted spatial
input channels across five seven-filter spatial convolutions.

## Synthetic EDF end-to-end validation

A fresh temporary dataset of 12 standards-compliant EDF recordings was created
at 200 Hz with the 19 required channels and the same directory nesting used by
NMT-4K. The following completed successfully:

1. Recursive discovery and exact split/class count validation.
2. `recordings.tsv` identifier matching.
3. EDF header and channel inspection.
4. Full preprocessing from raw EDF to 100 Hz cached arrays.
5. A second preprocessing run that reused every valid cache entry.
6. Stratified train-only development/validation splitting.
7. Development training and recording-level validation aggregation.
8. Full-training retraining and final checkpoint export.
9. Resume invocation after both phases were complete.
10. Final recording-level evaluation, prediction CSVs, metrics JSON/CSV,
    confusion matrix, and ROC/PR output generation.

This synthetic test validates execution and data flow, not clinical performance.
The actual NMT-4K files were not available in this execution environment, so the
user must run the supplied full dataset inspector before preprocessing/training.
