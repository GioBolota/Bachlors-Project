"""
train.py — Full training loop with validation, early stopping, and checkpointing.

Usage:
    python train.py
    python train.py --data_dir data --epochs 30 --batch_size 32 --lr 1e-4
"""

import argparse
import os
import time

import torch
import torch.nn as nn

from dataset import get_loaders
from model import FusionDetector, save_checkpoint


#argument parsing

def parse_args():
    p = argparse.ArgumentParser(description="Train FusionDetector")
    p.add_argument("--data_dir",    type=str,   default="data")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--fft_dim",     type=int,   default=256)
    p.add_argument("--patience",    type=int,   default=5,
                   help="Early-stopping patience (epochs without val improvement)")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--max_samples", type=int, default=None,
               help="Cap dataset size for quick testing")
    return p.parse_args()


# one epoch

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, fft_feats, labels in loader:
            images    = images.to(device)
            fft_feats = fft_feats.to(device)
            labels    = labels.to(device)

            outputs = model(images, fft_feats)
            loss    = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds  = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return total_loss / total, correct / total


# main

def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        fft_dim=args.fft_dim,
        max_samples=args.max_samples,
    )

    model     = FusionDetector(fft_feature_dim=args.fft_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2 #, verbose=True
    )

    best_val_acc = 0.0
    epochs_no_improve = 0

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # freeze CNN for first N epochs so FFT branch can warm up
        if epoch <= model.freeze_cnn_epochs:
            model.freeze_cnn()
        else:
            model.unfreeze_cnn()

        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device, training=False)
        elapsed = time.time() - t0

        scheduler.step(val_acc)

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  {val_loss:>8.4f}  {val_acc:>6.2%}  {elapsed:>5.1f}s")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(
                model,
                path=os.path.join(args.checkpoint_dir, "best_model.pth"),
                metadata={"epoch": epoch, "val_acc": val_acc}
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs).")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2%}")
    print(f"Best model saved to: {args.checkpoint_dir}/best_model.pth")


if __name__ == "__main__":
    main()
