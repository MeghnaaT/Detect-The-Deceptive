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
    print("Script started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dir = "data/real_vs_fake/real-vs-fake/train"
    valid_dir = "data/real_vs_fake/real-vs-fake/valid"

    print("Loading datasets...")
    train_loader, valid_loader = get_dataloaders(train_dir, valid_dir)
    print("Datasets loaded.")

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

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(epochs):

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
    # Save Model
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    save_path = "models/convnext_small_v1.pth"

    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved successfully at: {save_path}")



if __name__ == "__main__":
    main()
