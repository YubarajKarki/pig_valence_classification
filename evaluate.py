"""
evaluate.py — Evaluation metrics
=================================
This module is a clean Python-3 / PyTorch rewrite of the original
eval_tags.py (which was Python 2 and Theano-only).

For binary valence classification we report:
  • Accuracy          — fraction of correct predictions
  • AUC-ROC           — area under the ROC curve; 0.5 = random, 1.0 = perfect
                        robust to class imbalance (unlike raw accuracy)
  • Average Precision — area under the Precision-Recall curve

Why AUC-ROC?
------------
The pig dataset is imbalanced (~2:1 Neg/Pos).  AUC-ROC evaluates the
model's ability to *rank* positive examples above negative ones,
regardless of the decision threshold, making it more informative than
accuracy alone.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def evaluate(model, loader: DataLoader, criterion, device: torch.device):
    """
    Run the model on *loader* in eval mode (no gradient updates).

    Returns
    -------
    avg_loss : float
    accuracy : float
    labels   : np.ndarray — ground-truth class indices
    probs    : np.ndarray — predicted probability of the Positive class (P(y=1))
    """
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels    = labels.to(device)

            logits = model(waveforms)               # (B, num_classes)
            loss   = criterion(logits, labels)

            # Convert logits → probabilities with softmax, keep P(Positive)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)
    probs_arr  = np.array(all_probs)

    accuracy = accuracy_score(labels_arr, preds_arr)
    return avg_loss, accuracy, labels_arr, probs_arr


def print_test_report(labels: np.ndarray, probs: np.ndarray,
                       loss: float, acc: float):
    """Print a full evaluation report to stdout."""
    preds  = (probs >= 0.5).astype(int)
    auc    = roc_auc_score(labels, probs)
    ap     = average_precision_score(labels, probs)

    print("── Test-set results ──────────────────────────────────────────")
    print(f"  Loss               : {loss:.4f}")
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  AUC-ROC            : {auc:.4f}   (1.0 = perfect, 0.5 = random)")
    print(f"  Average Precision  : {ap:.4f}")
    print()
    print(classification_report(labels, preds, target_names=["Neg", "Pos"]))
    return auc, ap
