from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
import io
import cv2
import torch

from app.services.image_service import analyze_image
from app.ml.image.model import load_model
from app.ml.image.preprocess import preprocess_image
from app.ml.image.gradcam import generate_gradcam

router = APIRouter()   # 👈 THIS MUST COME BEFORE DECORATORS

_model = load_model()


@router.post("/image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    return analyze_image(file)


@router.post("/image/visualize")
async def visualize_image(file: UploadFile = File(...)):
    img_tensor, original_img = preprocess_image(file, return_original=True)
    tensor = torch.from_numpy(img_tensor).float()

    outputs = _model(tensor)
    probs = torch.softmax(outputs, dim=1)
    _, pred = torch.max(probs, dim=1)

    _model.zero_grad()
    outputs[0, pred.item()].backward()

    overlay = generate_gradcam(_model, original_img)

    _, buffer = cv2.imencode(".png", overlay)
    io_buf = io.BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/png")
