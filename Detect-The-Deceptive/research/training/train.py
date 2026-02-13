# research/training/train.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from model import get_model
from dataset import get_dataloaders
from utils import train_one_epoch, validate


def main():

    # -----------------------------
    # Device Setup
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)
    print("-" * 50)

    # -----------------------------
    # Mount Google Drive (Colab only)
    # -----------------------------
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        save_dir = "/content/drive/MyDrive/deepfake_checkpoints"
    except:
        save_dir = "models"

    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Dataset Paths
    # -----------------------------
    train_dir = "data/real_vs_fake/real-vs-fake/train"
    valid_dir = "data/real_vs_fake/real-vs-fake/valid"

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    if not os.path.exists(valid_dir):
        raise FileNotFoundError(f"Valid directory not found: {valid_dir}")

    # -----------------------------
    # Data Loaders
    # -----------------------------
    print("Loading datasets...")
    train_loader, valid_loader = get_dataloaders(train_dir, valid_dir)
    print("Datasets loaded successfully.")
    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    print("-" * 50)

    # -----------------------------
    # Model
    # -----------------------------
    model = get_model()
    model.to(device)

    print("Model moved to device:", next(model.parameters()).device)
    print("-" * 50)

    # -----------------------------
    # Loss & Optimizer
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 5
    best_val_acc = 0.0
    start_epoch = 0

    # -----------------------------
    # Resume Logic
    # -----------------------------
    latest_checkpoint = os.path.join(save_dir, "latest_checkpoint.pth")

    if os.path.exists(latest_checkpoint):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}")
        print("-" * 50)

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(start_epoch, epochs):

        print(f"\n===== Epoch {epoch+1}/{epochs} =====")
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_acc = validate(model, valid_loader, device)

        epoch_time = time.time() - start_time

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Epoch Time: {epoch_time/60:.2f} minutes")
        print("=" * 50)

        # -----------------------------
        # Save Latest Checkpoint
        # -----------------------------
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc
        }

        torch.save(checkpoint, latest_checkpoint)

        # -----------------------------
        # Save Best Model
        # -----------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print("Best model updated.")
            print("-" * 50)

    print("\nTraining completed.")


if __name__ == "__main__":
    main()
