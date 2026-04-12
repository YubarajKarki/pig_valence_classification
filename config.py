"""
config.py — Centralised hyperparameters and paths
==================================================
Keeping all settings in one place makes it easy to run experiments:
just change a value here and every other module picks it up.
"""


class Config:
    # ── Audio ─────────────────────────────────────────────────────────────
    SAMPLE_RATE   = 44_100        # Hz — must match your recording equipment
    # SampleCNN processes fixed-length chunks of raw audio.
    # 16 384 = 2^14 samples ≈ 0.37 s at 44 100 Hz.
    # Choosing a power of 2 keeps the architecture clean:
    # one strided-conv + 13 MaxPool(2) layers = exactly 1 time-step at output.
    SAMPLE_LENGTH = 16_384

    # ── Paths ──────────────────────────────────────────────────────────────
    ANNOTATIONS_FILE = "annotations.xlsx"   # annotation spreadsheet
    AUDIO_DIR        = "audio/"             # folder with .wav files
                                             # names must match 'Audio Filename'
    BEST_MODEL_PATH  = "best_model.pth"

    # ── Model ──────────────────────────────────────────────────────────────
    NUM_CLASSES = 2   # binary: Positive (1) vs Negative (0) valence

    # ── Training ───────────────────────────────────────────────────────────
    BATCH_SIZE    = 32
    NUM_EPOCHS    = 50
    # The original SampleCNN paper uses SGD with an initial lr of 0.01,
    # then manually steps it down (0.01 → 0.002 → 0.0004 → …).
    # Here we replace that with ReduceLROnPlateau for simplicity.
    LEARNING_RATE = 0.01
    MOMENTUM      = 0.9
    WEIGHT_DECAY  = 1e-7    # L2 regularisation on all weights
    PATIENCE      = 5       # epochs without val-loss improvement before LR drop
    NUM_WORKERS   = 4       # DataLoader workers (set 0 on Windows)

    # ── Split ──────────────────────────────────────────────────────────────
    # Groups by Recording Team — see dataset.py for the full rationale.
    GROUP_COLUMN  = "Recording Team"
    SEED          = 42
