# backend/app/api/image.py

import io
import os
import torch
import torch.nn.functional as F

from fastapi import APIRouter, UploadFile, File, HTTPException
from torchvision import transforms
from PIL import Image

from app.ml.image.model import get_model  # your model loader

router = APIRouter()

# -------------------------------------------------
# Device Setup
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Load Trained Model Once (on startup)
# -------------------------------------------------
MODEL_PATH = os.path.join(
    "app", "ml", "image", "models", "best_model.pth"
)

model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print(f"[INFO] Model loaded on {device}")

# -------------------------------------------------
# Image Transform (must match training)
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------------------------
# Endpoint
# -------------------------------------------------
@router.post("/image")
async def analyze_image(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Preprocess
    tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        
        confidence, pred = torch.max(probs, dim=1)
        print("Raw logits:", outputs.cpu().numpy())
        print("Probabilities:", probs.cpu().numpy())
        print("Predicted index:", pred.item())
    class_names = ["fake", "real"]  # must match training folder order
    predicted_label = class_names[pred.item()]

    return {
        "label": predicted_label,
        "confidence": round(confidence.item(), 4)
    }
