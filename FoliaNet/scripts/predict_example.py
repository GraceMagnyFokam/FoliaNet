"""
Run a single FoliaNet prediction from the command line.

    python scripts/predict_example.py --image path/to/leaf.jpg \
        --lat 40.0 --lon -88.2 --date 2024-07-15
"""

import argparse
import json

from folianet.inference import FoliaNetPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--ndvi", type=float, default=None)
    ap.add_argument("--checkpoint", default="checkpoints/folianet_best.pt")
    args = ap.parse_args()

    predictor = FoliaNetPredictor(args.checkpoint)
    result = predictor.predict(args.image, lat=args.lat, lon=args.lon,
                               obs_date=args.date, ndvi_override=args.ndvi)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
