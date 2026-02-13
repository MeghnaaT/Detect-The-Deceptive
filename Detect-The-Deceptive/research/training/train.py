# research/training/train.py

import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from model import get_model
from dataset import get_dataloaders
from utils import train_one_epoch, validate


def main():

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("-" * 50)

    # -----------------------------
    # Mount Drive (Colab)
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

    # Load full dataset
    print("Loading datasets...")
    train_loader_full, valid_loader = get_dataloaders(
        train_dir,
        valid_dir,
        batch_size=32
    )

    full_train_dataset = train_loader_full.dataset

    print("Datasets loaded.")
    print("Total training samples:", len(full_train_dataset))
    print("-" * 50)

    # -----------------------------
    # Model
    # -----------------------------
    model = get_model()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # -----------------------------
    # Training Settings
    # -----------------------------
    epochs = 30
    subset_size = 5000
    best_val_acc = 0.0
    start_epoch = 0

    latest_checkpoint = os.path.join(save_dir, "latest_checkpoint.pth")

    # -----------------------------
    # Resume if checkpoint exists
    # -----------------------------
    if os.path.exists(latest_checkpoint):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        print(f"Resumed from epoch {start_epoch}")
        print("-" * 50)

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(start_epoch, epochs):

        print(f"\n===== Epoch {epoch+1}/{epochs} =====")

        # ---- Random 5K subset ----
        dataset_size = len(full_train_dataset)
        indices = random.sample(range(dataset_size), subset_size)
        subset = Subset(full_train_dataset, indices)

        subset_loader = DataLoader(
            subset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )

        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            subset_loader,
            criterion,
            optimizer,
            device
        )

        val_acc = validate(model, valid_loader, device)

        epoch_time = time.time() - start_time

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Epoch Time: {epoch_time/60:.2f} minutes")
        print("-" * 50)

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
