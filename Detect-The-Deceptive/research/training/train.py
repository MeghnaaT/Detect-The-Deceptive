# research/training/train.py
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

    # 🔥 SAVE CHECKPOINT AFTER EACH EPOCH
    os.makedirs("models", exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, f"models/checkpoint_epoch_{epoch+1}.pth")

    print(f"\nModel saved successfully at: {save_path}")


if __name__ == "__main__":
    main()
