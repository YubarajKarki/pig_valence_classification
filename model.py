"""
model.py — SampleCNN architecture (PyTorch)
============================================
Original paper:
    Lee et al. (2017) "Sample-level Deep Convolutional Neural Networks
    for Music Auto-tagging Using Raw Waveforms"
    https://arxiv.org/abs/1703.01789

Key idea
--------
Instead of hand-crafting a spectrogram (STFT → Mel filterbank → log),
SampleCNN learns its own time-frequency representation directly from
the raw waveform samples.

How it works
------------
1. A *strided* first convolution (kernel=2, stride=2) acts as a learned
   filterbank — comparable to what a Mel filterbank does, but data-driven.

2. A stack of identical blocks (Conv → BN → ReLU → MaxPool) progressively
   compress the time axis while expanding the feature (channel) axis.

3. After all pooling the temporal dimension collapses to 1, so the model
   is invariant to the exact position of events within the chunk.

4. A 1×1 convolution acts as a channel-mixing fully-connected layer while
   keeping the spatial structure (more parameter-efficient than Flatten→Dense).

5. Dropout + a final linear layer produce the class scores.

Architecture summary (power-of-2 variant, SAMPLE_LENGTH=16384)
----------------------------------------------------------------
Input         :  (B, 1, 16384)   ← 1 channel = mono waveform
Conv0 s=2     :  (B, 128, 8192)
Block 1–2     :  (B, 128, ?)     ← 128 filters, MaxPool(2) each
Block 3–11    :  (B, 256, ?)     ← 256 filters, MaxPool(2) each
Block 12      :  (B, 256, ?)
Block 13      :  (B, 512, ?)     ← 512 filters, MaxPool(2)
Conv1×1       :  (B, 512, 1)
Dropout(0.5)
Flatten       :  (B, 512)
Linear        :  (B, num_classes)

Note: 1 initial stride-2 + 13 MaxPool(2) = 2^14 = 16384 total downsampling,
which exactly collapses SAMPLE_LENGTH=16384 to a single time step.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Building block
# ─────────────────────────────────────────────────────────────────────────────

class SampleCNNBlock(nn.Module):
    """
    One convolutional block: Conv1d → BatchNorm → ReLU → MaxPool.

    Parameters
    ----------
    in_channels  : number of input feature channels
    out_channels : number of output feature channels (filters to learn)
    kernel_size  : convolution kernel width in samples (default 2)
    pool_size    : MaxPool downsampling factor          (default 2)

    BatchNorm is applied *before* the activation — this is the
    'pre-activation' ordering used in the original SampleCNN code and
    helps stabilise training of deep networks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        pool_size: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding="same",   # keep temporal dimension before pooling
            bias=False,       # bias is redundant when followed by BatchNorm
        )
        self.bn   = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C_in, T)
        x = self.conv(x)   # → (B, C_out, T)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)   # → (B, C_out, T // pool_size)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full SampleCNN
# ─────────────────────────────────────────────────────────────────────────────

class SampleCNN(nn.Module):
    """
    SampleCNN — power-of-2 variant adapted for binary pig valence
    classification.

    Parameters
    ----------
    num_classes   : number of output classes (2 for Pos/Neg)
    sample_length : number of raw audio samples per input chunk.
                    Must be divisible by 2^14 = 16384 for the default
                    architecture to reduce to exactly 1 time step.
    """

    def __init__(self, num_classes: int = 2, sample_length: int = 16_384):
        super().__init__()

        # ── Layer 0: strided convolution (learned filterbank) ──────────────
        # kernel=2, stride=2 → halves the temporal dimension immediately.
        # This is the first 'power of 2' in the downsampling cascade.
        self.conv0 = nn.Conv1d(1, 128, kernel_size=2, stride=2, bias=False)
        self.bn0   = nn.BatchNorm1d(128)
        self.relu0 = nn.ReLU(inplace=True)

        # ── Layers 1–13: stacked SampleCNN blocks ─────────────────────────
        # Each block halves time (MaxPool(2)) and may double channels.
        # Channel progression mirrors the original paper:
        #   128 × 2 blocks → 256 × 11 blocks → 512 × 1 block → 512 × 1 (1×1)
        self.blocks = nn.Sequential(
            SampleCNNBlock(128, 128),   # block 1
            SampleCNNBlock(128, 128),   # block 2
            SampleCNNBlock(128, 256),   # block 3  ← channel expansion
            SampleCNNBlock(256, 256),   # block 4
            SampleCNNBlock(256, 256),   # block 5
            SampleCNNBlock(256, 256),   # block 6
            SampleCNNBlock(256, 256),   # block 7
            SampleCNNBlock(256, 256),   # block 8
            SampleCNNBlock(256, 256),   # block 9
            SampleCNNBlock(256, 256),   # block 10
            SampleCNNBlock(256, 256),   # block 11
            SampleCNNBlock(256, 256),   # block 12
            SampleCNNBlock(256, 512),   # block 13 ← channel expansion
        )

        # ── Layer 14: 1×1 convolution ─────────────────────────────────────
        # A kernel_size=1 conv mixes channels without touching time.
        # At this point the temporal dimension is already 1, so this is
        # equivalent to a fully-connected layer over the 512 features.
        self.conv_final = nn.Conv1d(512, 512, kernel_size=1, bias=False)
        self.bn_final   = nn.BatchNorm1d(512)
        self.relu_final = nn.ReLU(inplace=True)

        # ── Head ──────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(512, num_classes)

        # ── Weight initialisation (He/Kaiming — good for ReLU networks) ───
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, sample_length)  — raw mono waveform batch

        Returns
        -------
        logits : (B, num_classes)  — raw (pre-softmax) class scores
        """
        # Strided entry convolution
        x = self.relu0(self.bn0(self.conv0(x)))   # (B, 128, sample_length/2)

        # Stacked blocks — each halves the time axis
        x = self.blocks(x)                         # (B, 512, 1)

        # Channel mixing
        x = self.relu_final(self.bn_final(self.conv_final(x)))  # (B, 512, 1)

        # Classification head
        x = self.dropout(x)
        x = self.flatten(x)     # (B, 512)
        x = self.fc(x)          # (B, num_classes)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Quick architecture sanity check (run this file directly)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import Config

    model = SampleCNN(
        num_classes=Config.NUM_CLASSES,
        sample_length=Config.SAMPLE_LENGTH,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"SampleCNN — total parameters: {total_params:,}")

    # Feed a random batch through and print each layer's output shape
    dummy = torch.randn(4, 1, Config.SAMPLE_LENGTH)
    print(f"\nInput shape : {tuple(dummy.shape)}")

    with torch.no_grad():
        x = model.relu0(model.bn0(model.conv0(dummy)))
        print(f"After conv0 : {tuple(x.shape)}")
        for i, block in enumerate(model.blocks):
            x = block(x)
            print(f"After block {i+1:2d}: {tuple(x.shape)}")
        x = model.relu_final(model.bn_final(model.conv_final(x)))
        print(f"After conv1x1: {tuple(x.shape)}")
        x = model.flatten(model.dropout(x))
        logits = model.fc(x)
        print(f"Logits shape : {tuple(logits.shape)}")
