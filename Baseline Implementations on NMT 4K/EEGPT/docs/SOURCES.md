# Upstream sources and scientific references

- EEGPT official repository: <https://github.com/BINE022/EEGPT>
- EEGPT TUAB downstream model: <https://github.com/BINE022/EEGPT/blob/main/downstream_tueg/Modules/models/EEGPT_mcae_finetune_change.py>
- EEGPT TUAB training runner: <https://github.com/BINE022/EEGPT/blob/main/downstream_tueg/run_class_finetuning_EEGPT_change.py>
- EEGPT TUAB configuration: <https://github.com/BINE022/EEGPT/blob/main/downstream_tueg/finetune_TUAB_EEGPT.sh>
- NMT-4K official repository: <https://github.com/dll-ncai/NMT-4k-EEG-Dataset>

`eegpt_nmt/eegpt_backbone.py` is the EEGPT model file supplied in the attached
project and structurally derived from the official EEGPT implementation. V2
keeps that file checkpoint-compatible and places all new downstream behavior in
`model.py`, making the boundary between upstream backbone and new experiment
code explicit.

