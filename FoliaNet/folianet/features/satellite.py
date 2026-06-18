"""
Satellite-derived vegetation index (NDVI) for FoliaNet.

Real NDVI retrieval needs an authenticated provider (Sentinel Hub, Google Earth
Engine, Planet, etc.), so this module ships a *documented stub*: a stable
`fetch_ndvi()` interface plus a deterministic synthetic value so the pipeline
runs end-to-end today. Wire your provider into `_fetch_real_ndvi()` and the rest
of FoliaNet picks it up automatically.

Suggested integrations:
  - Sentinel Hub Statistical API  (https://docs.sentinel-hub.com)
  - Google Earth Engine  (ee.ImageCollection('COPERNICUS/S2_SR'))
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Union
import hashlib


@dataclass
class SatelliteFeatures:
    ndvi: float                 # -1..1 (vegetation density / vigor)
    data_available: bool = True
    source: str = "synthetic"


def _to_date(d: Union[str, date, datetime]) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()


def _synthetic_ndvi(lat: float, lon: float, target: date) -> float:
    seed = int(hashlib.sha256(f"ndvi{lat:.2f}{lon:.2f}{target}".encode()).hexdigest(), 16)
    return round(0.3 + 0.5 * ((seed % 1000) / 1000.0), 3)   # 0.30..0.80


def _fetch_real_ndvi(lat: float, lon: float, target: date) -> Optional[float]:
    """TODO: implement with your satellite provider. Return None if unavailable.

    Example (Sentinel Hub Statistical API): request a small bbox around (lat, lon)
    for a date window, evalscript computing (B08 - B04) / (B08 + B04), and return
    the mean. Keep credentials in environment variables, never in code.
    """
    return None


def fetch_ndvi(
    lat: float,
    lon: float,
    target_date: Union[str, date, datetime],
) -> SatelliteFeatures:
    target = _to_date(target_date)
    real = _fetch_real_ndvi(lat, lon, target)
    if real is not None:
        return SatelliteFeatures(ndvi=float(real), data_available=True, source="satellite")
    return SatelliteFeatures(ndvi=_synthetic_ndvi(lat, lon, target),
                             data_available=False, source="synthetic")
