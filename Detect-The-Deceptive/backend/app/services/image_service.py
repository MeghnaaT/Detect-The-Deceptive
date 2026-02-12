# Image service logic
from app.ml.image.predict import predict_image

def analyze_image(file):
    return predict_image(file)