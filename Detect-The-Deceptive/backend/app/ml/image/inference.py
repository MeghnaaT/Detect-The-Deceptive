import torch
from torchvision import transforms
from PIL import Image
import io
import os
from .model import get_model

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "best_model.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

    label = "fake" if prediction.item() == 1 else "real"

    return {
        "prediction": label,
        "confidence": float(confidence.item())
    }
