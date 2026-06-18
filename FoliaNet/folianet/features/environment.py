"""
Agronomic environmental-risk module.

Turns weather + satellite features into a 0..1 "favorability" index per disease.
This is deliberately transparent and rule-based (no learned weights) so it stays
interpretable and runs even before you have any field-labeled risk data. Tune the
parameters in `folianet/diseases.py` against your own observations.

Pure Python / no torch -> unit testable.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from folianet.diseases import DISEASES, Disease, Favorability


@dataclass
class WeatherFeatures:
    temp_mean_c: float          # mean air temp over the recent window (C)
    humidity_mean_pct: float    # mean relative humidity (%)
    precip_7d_mm: float         # total precipitation, trailing 7 days (mm)
    leaf_wetness_hours: float   # estimated leaf-wetness hours in window
    gdd: float                  # accumulated growing-degree-days
    data_available: bool = True # False when synthetic/fallback values are used


def _temperature_suitability(temp: float, fav: Favorability) -> float:
    """Trapezoidal suitability: 0 outside [t_min, t_max], 1 across the optimal band."""
    if temp <= fav.t_min or temp >= fav.t_max:
        return 0.0
    if fav.t_opt_low <= temp <= fav.t_opt_high:
        return 1.0
    if temp < fav.t_opt_low:
        return (temp - fav.t_min) / max(fav.t_opt_low - fav.t_min, 1e-6)
    return (fav.t_max - temp) / max(fav.t_max - fav.t_opt_high, 1e-6)


def _humidity_factor(humidity: float, fav: Favorability) -> float:
    if fav.humidity_min <= 0:
        return 0.0
    if humidity <= fav.humidity_min:
        return 0.0
    # Linear ramp from humidity_min up to 100%.
    return min((humidity - fav.humidity_min) / max(100 - fav.humidity_min, 1e-6), 1.0)


def _wetness_factor(wetness_hours: float, fav: Favorability) -> float:
    if fav.wetness_hours <= 0:
        return 0.0
    return min(wetness_hours / fav.wetness_hours, 1.0)


def disease_favorability(weather: WeatherFeatures, disease: Disease) -> float:
    """Combine temperature, humidity, and leaf-wetness terms into a 0..1 index."""
    if disease.is_healthy:
        return 0.0
    fav = disease.favorability
    t = _temperature_suitability(weather.temp_mean_c, fav)
    h = _humidity_factor(weather.humidity_mean_pct, fav)
    w = _wetness_factor(weather.leaf_wetness_hours, fav)
    # Temperature is necessary; humidity and wetness reinforce each other.
    score = t * (0.5 * h + 0.5 * w)
    return float(max(0.0, min(score, 1.0)))


def crop_favorability(
    weather: WeatherFeatures,
    crop: str,
    ndvi: Optional[float] = None,
) -> Dict[str, float]:
    """Favorability for every (non-healthy) disease of a crop.

    A low NDVI (sparse/stressed canopy) nudges favorability up slightly, since
    stressed crops are generally more susceptible. NDVI is optional.
    """
    canopy_modifier = 1.0
    if ndvi is not None:
        # NDVI ~0.2 (stressed) -> 1.15x, NDVI ~0.8 (vigorous) -> ~0.95x.
        canopy_modifier = max(0.9, min(1.2, 1.25 - 0.4 * ndvi))

    out: Dict[str, float] = {}
    for class_name, disease in DISEASES.items():
        if disease.crop != crop or disease.is_healthy:
            continue
        out[disease.key] = min(1.0, disease_favorability(weather, disease) * canopy_modifier)
    return out
