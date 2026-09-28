"""Constants shared by preprocessing, model construction, and validation."""

# The 19 scalp electrodes are channels that EEGPT knows by name from pretraining.
EEGPT_SCALP_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T7", "C3", "CZ", "C4", "T8",
    "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
]

# NMT-4K additionally supplies the left and right auricular electrodes.
NMT_INPUT_CHANNELS = EEGPT_SCALP_CHANNELS + ["A1", "A2"]

# Historical 10-20 names and mastoid aliases found in clinical EDF headers.
CHANNEL_ALIASES = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    "M1": "A1",
    "M2": "A2",
}

# EEGPT was pretrained on four seconds at 256 Hz: 4 * 256 = 1,024 samples.
EEGPT_SFREQ = 256.0
EEGPT_WINDOW_SECONDS = 4.0
EEGPT_WINDOW_SAMPLES = 1024

