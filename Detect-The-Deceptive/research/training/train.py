# research/training/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from dataset import get_dataloaders
from utils import train_one_epoch, validate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = "data/train"
    valid_dir = "data/valid"

    train_loader, valid_loader = get_dataloaders(train_dir, valid_dir)

    model = get_model()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 5

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = validate(model, valid_loader, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print("-" * 30)

    torch.save(model.state_dict(), "../../models/convnext_small_v1.pth")
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
