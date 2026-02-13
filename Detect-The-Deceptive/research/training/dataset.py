# research/training/dataset.py

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def get_dataloaders(train_dir, valid_dir, batch_size=32):
    train_transform, valid_transform = get_transforms()

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    valid_dataset = ImageFolder(valid_dir, transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, valid_loader
