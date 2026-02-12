import torch
import base64
import cv2
from app.ml.image.model import load_model
from app.ml.image.preprocess import preprocess_image
from app.ml.image.gradcam import generate_gradcam

_model = load_model()

def predict_image(upload_file):
    img_tensor, original_img = preprocess_image(upload_file, return_original=True)
    tensor = torch.from_numpy(img_tensor).float()

    outputs = _model(tensor)
    probs = torch.softmax(outputs, dim=1)
    confidence, pred = torch.max(probs, dim=1)

    # Backward pass
    _model.zero_grad()
    outputs[0, pred.item()].backward()

    overlay = generate_gradcam(_model, original_img)

    # Convert overlay to base64
    _, buffer = cv2.imencode(".png", overlay)
    overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "label": "fake" if pred.item() == 1 else "real",
        "confidence": round(confidence.item(), 3),
        "heatmap_image": overlay_base64
    }
