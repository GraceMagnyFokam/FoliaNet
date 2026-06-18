"""
FoliaNet API.

Run:
    uvicorn api.main:app --reload

Configure the checkpoint via the FOLIANET_CHECKPOINT env var
(default: checkpoints/folianet_best.pt).

POST /predict  (multipart/form-data):
    image: file (required)
    lat:   float (required)
    lon:   float (required)
    date:  YYYY-MM-DD (required)
    ndvi:  float (optional override)
"""

import io
import os

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from PIL import Image

app = FastAPI(title="FoliaNet", version="0.1.0",
              description="AI crop disease detection for corn & wheat fungal diseases.")

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        # Imported lazily so /health works even before torch/model are ready.
        from folianet.inference import FoliaNetPredictor
        ckpt = os.environ.get("FOLIANET_CHECKPOINT", "checkpoints/folianet_best.pt")
        if not os.path.exists(ckpt):
            raise HTTPException(503, f"Model checkpoint not found at '{ckpt}'. Train first.")
        _predictor = FoliaNetPredictor(ckpt)
    return _predictor


@app.get("/health")
def health():
    return {"status": "ok", "service": "folianet"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    date: str = Form(...),
    ndvi: float = Form(None),
):
    try:
        img = Image.open(io.BytesIO(await image.read()))
    except Exception:
        raise HTTPException(400, "Could not read the uploaded image.")
    predictor = get_predictor()
    try:
        return predictor.predict(img, lat=lat, lon=lon, obs_date=date, ndvi_override=ndvi)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
