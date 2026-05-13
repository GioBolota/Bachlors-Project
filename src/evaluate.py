"""
evaluate.py — Evaluate a trained FusionDetector on the test set.

Usage:
    python evaluate.py
    python evaluate.py --data_dir data --checkpoint checkpoints/best_model.pth
"""

import argparse
import os

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from dataset import get_loaders
from model import load_checkpoint


# argument parsing

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate FusionDetector on test set")
    p.add_argument("--data_dir",   type=str, default="data")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--fft_dim",    type=int, default=256)
    return p.parse_args()


# evaluation loop

def evaluate(model, loader, device):
    """Run inference on the full loader; return arrays of labels and probabilities."""
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, fft_feats, labels in loader:
            images    = images.to(device)
            fft_feats = fft_feats.to(device)

            probs = model(images, fft_feats).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_probs)


# pretty print helpers

def print_confusion_matrix(cm):
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion Matrix:")
    print(f"              Predicted Real  Predicted AI")
    print(f"  Actual Real     {tn:>6}          {fp:>6}")
    print(f"  Actual AI       {fn:>6}          {tp:>6}")


# main
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = load_checkpoint(args.checkpoint, fft_feature_dim=args.fft_dim, device=device)

    _, _, test_loader = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        fft_dim=args.fft_dim,
        num_workers=0,    # safer for evaluation on CPU
    )

    print("\nRunning evaluation on test set...")
    labels, probs = evaluate(model, test_loader, device)
    preds = (probs >= 0.5).astype(int)

    # metrics
    acc       = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)
    auc       = roc_auc_score(labels, probs)
    cm        = confusion_matrix(labels, preds)

    print("\n" + "=" * 45)
    print(f"  Test Results")
    print("=" * 45)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("=" * 45)

    print_confusion_matrix(cm)

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Real", "AI Generated"]))


if __name__ == "__main__":
    main()
