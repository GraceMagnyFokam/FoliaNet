"""
FoliaNet inference.

Pipeline:
  image  --CNN-->  disease class probabilities
  lat/lon/date  --weather + satellite-->  environmental favorability per disease
  late fusion  -->  final risk score
  + confidence, likely stressor, recommended action

This produces the product's four required outputs (risk score, confidence,
likely stressor, recommended action) plus supporting detail.
"""

from dataclasses import asdict
from datetime import date
from typing import Optional, Union
import math

from PIL import Image
import torch
import torch.nn.functional as F

from folianet.diseases import get_disease
from folianet.features.weather import fetch_weather
from folianet.features.satellite import fetch_ndvi
from folianet.features.environment import crop_favorability, WeatherFeatures
from folianet.recommend import recommend_action
from folianet.models.fusion_model import FusionModel
from folianet.data.dataset import build_transforms

MODEL_VERSION = "folianet-0.1.0"


def _entropy_confidence(probs: torch.Tensor) -> float:
    """1 - normalized entropy of the class distribution, in [0, 1]."""
    p = probs.clamp_min(1e-9)
    ent = -(p * p.log()).sum().item()
    max_ent = math.log(len(p))
    return float(1.0 - ent / max_ent) if max_ent > 0 else 1.0


def _confidence_level(score: float) -> str:
    return "high" if score >= 0.66 else "medium" if score >= 0.4 else "low"


class FoliaNetPredictor:
    def __init__(self, checkpoint_path: str, image_weight: float = 0.6,
                 env_weight: float = 0.4, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.classes = ckpt["classes"]
        self.img_size = ckpt.get("img_size", 224)
        self.model = FusionModel(
            num_classes=len(self.classes),
            tabular_dim=ckpt.get("tabular_dim", 0),
            backbone=ckpt.get("backbone", "efficientnet_b0"),
            pretrained=False,
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.transform = build_transforms(self.img_size, train=False)
        self.image_weight = image_weight
        self.env_weight = env_weight

    @torch.no_grad()
    def _image_probs(self, image: Image.Image) -> torch.Tensor:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        return F.softmax(self.model(x), dim=1).squeeze(0).cpu()

    def predict(
        self,
        image: Union[str, Image.Image],
        lat: float,
        lon: float,
        obs_date: Union[str, date],
        ndvi_override: Optional[float] = None,
    ) -> dict:
        if isinstance(image, str):
            image = Image.open(image)
        probs = self._image_probs(image)

        # ---- interpret image classifier ----
        diseases = [get_disease(c) for c in self.classes]
        healthy_mass = sum(float(probs[i]) for i, d in enumerate(diseases) if d.is_healthy)
        image_disease_prob = 1.0 - healthy_mass

        # top NON-healthy class drives the stressor identity
        disease_idxs = [i for i, d in enumerate(diseases) if not d.is_healthy]
        top_idx = max(disease_idxs, key=lambda i: float(probs[i])) if disease_idxs \
            else int(probs.argmax())
        top = diseases[top_idx]

        # ---- environmental context (late fusion) ----
        weather = fetch_weather(lat, lon, obs_date)
        sat = fetch_ndvi(lat, lon, obs_date)
        ndvi = ndvi_override if ndvi_override is not None else sat.ndvi
        fav_map = crop_favorability(weather, top.crop, ndvi=ndvi)
        env_favorability = float(fav_map.get(top.key, 0.0))
        env_available = weather.data_available or sat.data_available

        # ---- fuse ----
        if env_available:
            w_img, w_env = self.image_weight, self.env_weight
        else:
            # No real environmental data -> trust the image model only.
            w_img, w_env = 1.0, 0.0
        denom = w_img + w_env
        risk = (w_img * image_disease_prob + w_env * env_favorability) / denom

        # If the model is confident the plant is healthy, keep risk low.
        if top.is_healthy:
            risk = min(risk, image_disease_prob)

        # ---- confidence ----
        model_conf = float(probs[top_idx]) if not _is_healthy_pred(probs, diseases) \
            else healthy_mass
        confidence = 0.5 * model_conf + 0.5 * _entropy_confidence(probs)
        if not env_available:
            confidence *= 0.85  # discount when environmental signals are synthetic

        band, action = recommend_action(top.key, top.is_healthy, risk)

        return {
            "risk_score": round(float(risk), 4),
            "risk_score_pct": round(float(risk) * 100, 1),
            "risk_band": band,
            "confidence": {
                "score": round(float(confidence), 4),
                "level": _confidence_level(confidence),
            },
            "likely_stressor": {
                "name": top.display_name,
                "pathogen": top.pathogen,
                "crop": top.crop,
                "healthy": top.is_healthy,
            },
            "recommended_action": action,
            "details": {
                "image_disease_probability": round(image_disease_prob, 4),
                "environmental_favorability": round(env_favorability, 4),
                "class_probabilities": {
                    self.classes[i]: round(float(probs[i]), 4) for i in range(len(self.classes))
                },
                "weather": {k: round(v, 3) if isinstance(v, float) else v
                            for k, v in asdict(weather).items()},
                "ndvi": round(float(ndvi), 3),
                "data_sources": {
                    "weather": weather.data_available,
                    "satellite": sat.data_available,
                },
                "fusion_weights": {"image": w_img, "environment": w_env},
                "model_version": MODEL_VERSION,
            },
        }


def _is_healthy_pred(probs: torch.Tensor, diseases) -> bool:
    top = int(probs.argmax())
    return diseases[top].is_healthy
