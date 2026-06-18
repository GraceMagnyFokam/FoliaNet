"""
Weather features for FoliaNet.

Uses Open-Meteo (free, no API key) to pull the trailing weather window for a
location/date and derive the features the environmental-risk module needs:
mean temperature, mean humidity, 7-day precipitation, a leaf-wetness proxy, and
growing-degree-days. If the network call fails, a deterministic synthetic
fallback is returned with `data_available=False` so the pipeline never breaks.

Docs: https://open-meteo.com/en/docs  (archive + forecast endpoints)
"""

from datetime import date, datetime, timedelta
from typing import Union
import hashlib

from folianet.features.environment import WeatherFeatures

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = "temperature_2m,relative_humidity_2m,precipitation"
GDD_BASE_C = 10.0   # corn/wheat-ish base temperature for GDD


def _to_date(d: Union[str, date, datetime]) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()


def _synthetic(lat: float, lon: float, target: date) -> WeatherFeatures:
    """Deterministic pseudo-weather so demos/tests run with no network."""
    seed = int(hashlib.sha256(f"{lat:.2f}{lon:.2f}{target}".encode()).hexdigest(), 16)
    r = (seed % 1000) / 1000.0
    temp = 12 + 18 * r                     # 12..30 C
    humidity = 55 + 40 * ((seed >> 7) % 1000) / 1000.0   # 55..95 %
    precip = 30 * ((seed >> 11) % 1000) / 1000.0         # 0..30 mm
    wetness = 4 + 18 * ((seed >> 13) % 1000) / 1000.0    # 4..22 h
    gdd = max(0.0, (temp - GDD_BASE_C)) * 7
    return WeatherFeatures(temp, humidity, precip, wetness, gdd, data_available=False)


def fetch_weather(
    lat: float,
    lon: float,
    target_date: Union[str, date, datetime],
    window_days: int = 7,
) -> WeatherFeatures:
    """Fetch and aggregate the trailing weather window ending on `target_date`."""
    target = _to_date(target_date)
    start = target - timedelta(days=window_days)

    try:
        import requests  # imported lazily so the package loads without it

        use_archive = target <= date.today() - timedelta(days=5)
        url = ARCHIVE_URL if use_archive else FORECAST_URL
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": HOURLY_VARS,
            "start_date": start.isoformat(),
            "end_date": target.isoformat(),
            "timezone": "auto",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        temps = [t for t in hourly.get("temperature_2m", []) if t is not None]
        hums = [h for h in hourly.get("relative_humidity_2m", []) if h is not None]
        precs = [p for p in hourly.get("precipitation", []) if p is not None]
        if not temps or not hums:
            raise ValueError("empty weather response")

        temp_mean = sum(temps) / len(temps)
        hum_mean = sum(hums) / len(hums)
        precip_total = sum(precs)
        # Leaf-wetness proxy: hours with RH >= 90% or measurable precip.
        wetness = sum(
            1 for i in range(min(len(hums), len(precs)))
            if hums[i] >= 90 or precs[i] > 0.1
        )
        # GDD from daily mean temps (approx via hourly mean per day).
        gdd = max(0.0, temp_mean - GDD_BASE_C) * window_days
        return WeatherFeatures(temp_mean, hum_mean, precip_total, float(wetness), gdd, True)

    except Exception:
        # Network down, no `requests`, bad response, etc. -> safe fallback.
        return _synthetic(lat, lon, target)
