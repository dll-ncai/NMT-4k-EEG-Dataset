# Version 1.0.2 short-recording hotfix

This update handles NMT-4K EDFs that contain a complete 60-second model input
but are too short to discard the full first minute first. It selects the last
complete 60 seconds of real EEG and records every adaptation in the audit
output. It never pads or repeats a signal. EDFs shorter than 60 seconds remain
excluded.

To apply the hotfix archive over an existing installation:

```powershell
Expand-Archive `
  -Path .\Multi-BK-Net-v1.0.2-short-recording-hotfix.zip `
  -DestinationPath E:\Multi-BK-Net-NMT4K-Windows `
  -Force
Set-Location E:\Multi-BK-Net-NMT4K-Windows
.\03_preprocess.ps1
```

Do not pass `--overwrite`. Existing valid cached arrays retain the same verified
preprocessing signature, so the script reuses them and retries only previously
failed recordings. The hotfix archive does not replace your `config.yaml` or
your `.venv`.

Before training, confirm that `evaluation_failures` is `0`. Review:

```text
G:\Dataset\NMT-4K-EEG\multibknet_nmt_outputs\audit\short_recording_adaptations.csv
G:\Dataset\NMT-4K-EEG\multibknet_nmt_outputs\audit\preprocessing_failures.csv
```
