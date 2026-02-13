# Image API endpoints
from fastapi import APIRouter, UploadFile, File
from app.services.image_service import analyze_image

router = APIRouter()

@router.post("/image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    return analyze_image(file)
