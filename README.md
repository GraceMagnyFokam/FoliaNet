# FoliaNet v1

AI crop disease detection for **corn and wheat fungal diseases**. FoliaNet takes a
crop image plus location, date, and environmental/satellite context and returns a
**risk score**, a **confidence level**, the **likely stressor**, and a
**recommended next action** — in a single call.

---

## What it does

- **Identifies the likely stressor** from a leaf image using a transfer-learning CNN
  (EfficientNet-B0, PyTorch).
- **Scores disease risk** by fusing the image prediction with environmental context —
  temperature, humidity, leaf-wetness, rainfall, and satellite NDVI.
- **Reports a confidence level** alongside every prediction.
- **Recommends a next action** tailored to the detected stressor and risk level.
- **Serves over HTTP** via a FastAPI endpoint, ready to drop into a product or app.

## How it works

```
   leaf image ─────────────▶ CNN classifier ──▶ disease probabilities ─┐
                                                                        ├─▶ fusion ─▶ risk score
   lat / lon / date ─▶ weather + NDVI ─▶ agronomic favorability ────────┘            confidence
                                                                                      likely stressor
                                                                                      recommended action
```

FoliaNet combines two complementary signals:

1. **Image branch** — a fine-tuned CNN produces disease-class probabilities from a leaf photo.
2. **Environmental branch** — weather and satellite NDVI for the field's location and date
   are converted into a per-disease favorability index using agronomic rules.

These are fused into the final risk score, confidence level, likely stressor, and
recommended action. The model also supports **early fusion** (joint image + tabular
training) — set `model.tabular_dim > 0` once you have paired data.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

See [`data/README.md`](data/README.md) for preparing a labeled image dataset, then
set `data.root` in `configs/default.yaml`.

## Train

```bash
python -m folianet.train --config configs/default.yaml
```

Saves `checkpoints/folianet_best.pt` (weights + class list + metadata).

## Predict (CLI)

```bash
python scripts/predict_example.py \
  --image path/to/leaf.jpg --lat 40.0 --lon -88.2 --date 2024-07-15
```

## Serve (API)

```bash
uvicorn api.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F image=@path/to/leaf.jpg \
  -F lat=40.0 -F lon=-88.2 -F date=2024-07-15
```

### Output schema

```json
{
  "risk_score": 0.78,
  "risk_score_pct": 78.0,
  "risk_band": "high",
  "confidence": { "score": 0.81, "level": "high" },
  "likely_stressor": {
    "name": "Gray leaf spot",
    "pathogen": "Cercospora zeae-maydis",
    "crop": "Corn",
    "healthy": false
  },
  "recommended_action": "High pressure. Prioritize scouting now; ...",
  "details": {
    "image_disease_probability": 0.83,
    "environmental_favorability": 0.71,
    "class_probabilities": { "...": 0.0 },
    "weather": { "temp_mean_c": 26.1, "humidity_mean_pct": 92.0, "...": 0 },
    "ndvi": 0.62,
    "data_sources": { "weather": true, "satellite": false },
    "fusion_weights": { "image": 0.6, "environment": 0.4 },
    "model_version": "folianet-0.1.0"
  }
}
```

---

## Pluggable data sources

- **Weather** uses [Open-Meteo](https://open-meteo.com) (free, no API key) and works out of the box.
- **Satellite NDVI** is pluggable — add your provider (Sentinel Hub, Google Earth Engine, etc.)
  in `folianet/features/satellite.py`.

## Project layout

```
folianet/
  diseases.py            disease metadata + agronomic favorability params
  config.py              YAML config loader
  data/                  dataset + download helper
  features/              weather, satellite, environmental favorability
  models/fusion_model.py image (+ optional tabular) network
  train.py               training loop
  inference.py           predictor: assembles the risk/confidence/stressor/action output
  recommend.py           stressor + risk band -> action text
api/main.py              FastAPI service
scripts/predict_example.py
tests/test_logic.py      unit tests (pytest)
```

## Tests

```bash
pytest -q
```

## Responsible use

- FoliaNet is an advisory tool that supports scouting decisions; follow local agronomy /
  extension guidance for treatment timing, rates, and product choice.
- Tune and validate risk thresholds against your own field data, and back any performance
  figures with your own pilot methodology.

## License

MIT — see [LICENSE](LICENSE). © Grace Fokam.
