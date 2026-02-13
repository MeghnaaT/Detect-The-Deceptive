from fastapi import FastAPI
from app.api import health, image

app = FastAPI(
    tittle = "Detect the Deceptive API",
    version = "1.0.0"
)

app.include_router(health.router)
app.include_router(image.router, prefix = "/analyze")

