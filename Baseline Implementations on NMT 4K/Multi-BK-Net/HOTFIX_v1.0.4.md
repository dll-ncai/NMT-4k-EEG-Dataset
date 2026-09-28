# Multi-BK-Net v1.0.4 resume-performance hotfix

This hotfix addresses a GPU-starved Windows training pipeline while preserving
the completed-epoch checkpoint and configured effective batch size.

## Changes

- Adds two Windows DataLoader worker processes with background prefetching.
- Caches recently opened NumPy memory maps independently inside each worker.
- Avoids repeated Pandas row lookup in every window request.
- Uses a physical batch of 64 with one accumulation step. The configured
  effective batch remains exactly 64 windows (`16 x 4` becomes `64 x 1`).
- Uses validation batches of 64.
- Records all runtime settings in every new history row and the phase summary.
- Leaves `config.yaml`, its checkpoint hash, optimizer state, scheduler state,
  model state, split, sample order, class weights, and best checkpoint intact.

The physical grouping of the same 64-window effective batch can cause small
floating-point differences from four 16-window micro-batches, but it does not
change the conceptual optimizer batch size or selection/evaluation protocol.

## Apply after a completed epoch

1. Wait until the epoch-4 validation result and `Epoch 04/42` summary appear.
2. Once epoch 5 begins, press `Ctrl+C` and wait for the PowerShell prompt.
3. Extract this ZIP directly into `E:\Multi-BK-Net-NMT4K-Windows` and replace
   existing files.
4. Do not edit `config.yaml`.
5. Run `08_run_unit_tests.ps1`; the expected result is 21 passed tests.
6. Run `09_resume_optimized.ps1`.

The resume command should report `Resuming development at epoch 5/42`, followed
by:

```text
runtime batching: micro=64, accumulation=1, effective=64
runtime loading: workers=2, prefetch=2, validation_batch=64
```

Training progress will contain approximately 708 batches per epoch instead of
2,829. If CUDA reports out-of-memory, set the launcher to micro-batch 32 and
accumulation 2; the effective batch will remain 64.
