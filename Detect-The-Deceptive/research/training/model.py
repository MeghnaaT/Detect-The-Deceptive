# research/training/model.py

import torch
import torch.nn as nn
import torchvision.models as models


def get_model(num_classes=2, pretrained=True):
    weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_small(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model
