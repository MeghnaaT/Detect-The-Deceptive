# backend/app/ml/image/model.py

import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH = "app/ml/image/models/best_model.pth"

def get_model():

    model = models.convnext_small(pretrained=False)

    # Replace classifier for binary classification
    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features, 2
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    model.eval()

    return model
