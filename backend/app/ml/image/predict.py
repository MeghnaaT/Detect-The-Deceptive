# Image prediction logic
import torch
import cv2
import numpy as np
from app.ml.image.model import load_model
from app.ml.image.preprocess import preprocess_image
from app.ml.image.gradcam import generate_gradcam

_model = load_model()

def predict_image(upload_file):
    img_tensor, original_img = preprocess_image(upload_file, return_original=True)
    tensor = torch.from_numpy(img_tensor).float()

    with torch.no_grad():
        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    label = "fake" if pred.item() == 1 else "real"

    # Grad-CAM
    heatmap = generate_gradcam(_model, tensor, pred.item())

    return {
        "label": label,
        "confidence": round(confidence.item(), 3),
        "heatmap": heatmap.tolist()
    }