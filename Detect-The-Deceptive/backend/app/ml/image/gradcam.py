import torch
import numpy as np
import cv2

def generate_gradcam(model, original_image):
    gradients = model.gradients
    activations = model.activations

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.relu(heatmap)

    if torch.max(heatmap) != 0:
        heatmap /= torch.max(heatmap)

    heatmap = heatmap.detach().cpu().numpy()

    # Resize to original image size
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # Convert to 0–255
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay on original image
    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    return overlay
