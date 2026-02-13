import torch
import cv2
import numpy as np

def generate_gradcam(model, input_tensor, class_idx):
    # Forward pass
    output = model(input_tensor)

    # Zero grads
    model.zero_grad()

    # Backward for target class
    output[0, class_idx].backward()

    gradients = model.gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activations = model.model.features[-1](input_tensor).detach()

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap)

    return heatmap.cpu().numpy()
