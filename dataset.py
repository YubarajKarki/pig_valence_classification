"""
dataset.py — Data loading and train/val/test splitting
=======================================================

Raw-waveform pipeline
---------------------
Unlike the spectrogram pipeline (where images are pre-extracted),
here we load .wav files at runtime and return fixed-length *chunks*
of the raw waveform to the model.

Chunking strategy
-----------------
Pig vocalisations are short (typically 0.2–0.4 s).  We pad or truncate
every file to exactly SAMPLE_LENGTH samples so all tensors in a batch
have the same shape.

  • If the call is *shorter* than SAMPLE_LENGTH → zero-pad on the right.
  • If the call is *longer*  than SAMPLE_LENGTH → take the first chunk.

An alternative (used in the original SampleCNN for music) is to cut one
long audio file into many overlapping segments and average their
predictions at test time (see train.py for this option).

Split strategy — grouped by Recording Team
------------------------------------------
Pig IDs are encoded differently across the 6 recording teams
(ETHZ: 'pig15', IASPA: 'Pig8', FBN: 'VT13', …).
We therefore group by 'Recording Team' (a clean dataset-wide identifier)
which prevents:
  1. Within-pig leakage   — calls from the same animal in train AND test.
  2. Within-site leakage  — recordings from the same lab in train AND test.

If you add a standardised per-pig ID column later, just swap
Config.GROUP_COLUMN from 'Recording Team' to that column name.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit

try:
    import librosa           # preferred: handles resampling automatically
    LIBROSA_AVAILABLE = True
except ImportError:
    import soundfile as sf   # fallback: faster, no resampling
    LIBROSA_AVAILABLE = False

from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = Config.SAMPLE_RATE) -> np.ndarray:
    """
    Load a .wav file and return a 1-D float32 array at *target_sr* Hz.

    librosa is preferred because it resamples automatically if the file's
    native sample rate differs from target_sr.  soundfile is a fast fallback
    but will raise an error if the sample rates don't match.
    """
    if LIBROSA_AVAILABLE:
        y, _ = librosa.load(path, sr=target_sr, mono=True)
    else:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:               # stereo → mono
            y = y.mean(axis=1)
        if sr != target_sr:
            raise ValueError(
                f"File {path} has sr={sr}, expected {target_sr}. "
                "Install librosa for automatic resampling."
            )
    return y.astype(np.float32)


def pad_or_truncate(y: np.ndarray, length: int) -> np.ndarray:
    """
    Force a 1-D array to exactly *length* samples.
    Short signals are zero-padded; long ones are truncated.
    """
    if len(y) >= length:
        return y[:length]
    # Zero-pad on the right
    return np.pad(y, (0, length - len(y)), mode="constant")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PigAudioDataset(Dataset):
    """
    Loads raw pig vocalisations from .wav files and returns
    (waveform_tensor, label) pairs.

    Parameters
    ----------
    dataframe     : pd.DataFrame — rows from annotations.xlsx
    audio_dir     : str          — folder containing the .wav files
    sample_length : int          — fixed number of samples per clip
    augment       : bool         — apply training-time waveform augmentation
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        audio_dir: str = Config.AUDIO_DIR,
        sample_length: int = Config.SAMPLE_LENGTH,
        augment: bool = False,
    ):
        self.df            = dataframe.reset_index(drop=True)
        self.audio_dir     = audio_dir
        self.sample_length = sample_length
        self.augment       = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Build the full file path
        # The 'Audio Filename' column already contains the .wav filename.
        filepath = os.path.join(self.audio_dir, row["Audio Filename"])

        # Load and normalise waveform
        y = load_audio(filepath, target_sr=Config.SAMPLE_RATE)
        y = pad_or_truncate(y, self.sample_length)

        # ── Training augmentation ─────────────────────────────────────────
        if self.augment:
            y = self._augment(y)

        # Normalise amplitude to [-1, 1] to help training stability
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak

        # Add channel dimension: (1, sample_length) — same as (C, T) convention
        waveform = torch.from_numpy(y).unsqueeze(0)   # (1, sample_length)
        label    = torch.tensor(row["label"], dtype=torch.long)
        return waveform, label

    # ── Waveform augmentations ────────────────────────────────────────────
    def _augment(self, y: np.ndarray) -> np.ndarray:
        """
        Lightweight augmentations that make acoustic sense for vocalisations:

        1. Random gain   — simulate microphone distance / volume variation.
        2. Random flip   — time reversal is unusual but acts as regularisation.
           (Use with caution: reversed pig calls may not sound natural.)
        3. Additive noise — simulate background noise in the barn.

        We deliberately do NOT shift pitch (would change fundamental frequency)
        or stretch time (would change call duration statistics).
        """
        # 1. Random gain in [0.8, 1.2]
        gain = np.random.uniform(0.8, 1.2)
        y = y * gain

        # 2. Randomly flip the waveform in time (~50 % chance)
        if np.random.rand() < 0.5:
            y = y[::-1].copy()

        # 3. Add very low-level Gaussian noise (SNR ≈ 30 dB)
        noise_std = np.abs(y).mean() * 0.03
        y = y + np.random.randn(len(y)).astype(np.float32) * noise_std

        return y


# ─────────────────────────────────────────────────────────────────────────────
# Split builder
# ─────────────────────────────────────────────────────────────────────────────

def build_splits(annotations_file: str = Config.ANNOTATIONS_FILE):
    """
    Load the annotation spreadsheet and return three DataFrames:
    (df_train, df_val, df_test) with no group appearing in more than
    one split.

    Returns
    -------
    df_train, df_val, df_test : pd.DataFrame
    """
    df = pd.read_excel(annotations_file)

    # Binary label: Positive → 1, Negative → 0
    df["label"] = (df["Valence"] == "Pos").astype(int)

    groups = df[Config.GROUP_COLUMN].values

    # Step 1: hold out 20 % as (val + test)
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.20,
                                  random_state=Config.SEED)
    train_idx, temp_idx = next(gss_outer.split(df, groups=groups))

    # Step 2: split the 20 % evenly into val and test
    temp_groups = groups[temp_idx]
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.50,
                                  random_state=Config.SEED)
    val_rel, test_rel = next(
        gss_inner.split(df.iloc[temp_idx], groups=temp_groups)
    )
    val_idx  = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    # Print a summary so students can verify the split is clean
    print(f"Split summary:")
    print(f"  Train  {len(df_train):>5} samples | "
          f"groups: {sorted(df_train[Config.GROUP_COLUMN].unique())}")
    print(f"  Val    {len(df_val):>5} samples | "
          f"groups: {sorted(df_val[Config.GROUP_COLUMN].unique())}")
    print(f"  Test   {len(df_test):>5} samples | "
          f"groups: {sorted(df_test[Config.GROUP_COLUMN].unique())}")

    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────────────────────
# Quick test (run this file directly)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_train, df_val, df_test = build_splits()

    # Build datasets (augment only the training set)
    train_ds = PigAudioDataset(df_train, augment=True)
    val_ds   = PigAudioDataset(df_val,   augment=False)
    test_ds  = PigAudioDataset(df_test,  augment=False)

    print(f"\nDataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Load the first training sample to verify shapes
    waveform, label = train_ds[0]
    print(f"Waveform tensor shape: {tuple(waveform.shape)}")
    print(f"Label: {label.item()}  (0=Neg, 1=Pos)")
