# Pig Valence Classification — SampleCNN (PyTorch)

Binary classification of pig vocalisations (**Positive** vs **Negative** valence)
from **raw audio waveforms** using a PyTorch re-implementation of **SampleCNN**
(Lee et al., 2017).

---
## Soundwel dataset [download](https://data.niaid.nih.gov/resources?id=zenodo_8252482)


## Why raw waveform instead of spectrogram?

| Approach | Representation | Feature extraction |
|---|---|---|
| Spectrogram CNN | Log-Mel image | Hand-crafted (STFT → Mel filterbank → log) |
| **SampleCNN** | Raw samples | **Learned** by the first strided convolution |

SampleCNN's first layer learns its own filterbank from data — it can discover
frequency patterns that a fixed Mel filterbank might miss for non-musical sounds
like pig vocalisations.

---

## Project structure

```
pig_samplecnn/
├── config.py     ← all paths and hyper-parameters (edit here)
├── model.py      ← SampleCNN architecture
├── dataset.py    ← data loading, augmentation, group-aware split
├── evaluate.py   ← metrics (accuracy, AUC-ROC, average precision)
└── train.py      ← training + evaluation entry point
```

---

## Quick start

### 1. Install dependencies
```bash
pip install torch torchvision torchaudio librosa pandas openpyxl \
            scikit-learn matplotlib seaborn soundfile
```

### 2. Set your paths in `config.py`
```python
ANNOTATIONS_FILE = "annotations.xlsx"   # path to annotation spreadsheet
AUDIO_DIR        = "audio/"             # folder containing .wav files
```

### 3. Verify the architecture
```bash
python model.py
```

### 4. Verify data loading
```bash
python dataset.py
```

### 5. Train
```bash
python train.py
```

---

## Reference

> Lee, J., Park, J., Kim, K. L., & Nam, J. (2017).
> *Sample-level deep convolutional neural networks for music auto-tagging
> using raw waveforms.*
> arXiv:1703.01789
