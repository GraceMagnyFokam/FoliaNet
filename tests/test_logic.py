"""Tests for FoliaNet's torch-free logic (run with `pytest`)."""

from folianet.diseases import DISEASES, get_disease
from folianet.features.environment import (
    WeatherFeatures, disease_favorability, crop_favorability,
)
from folianet.recommend import risk_band, recommend_action, LOW, MODERATE, HIGH


def _favorable_weather():
    return WeatherFeatures(temp_mean_c=26, humidity_mean_pct=95,
                           precip_7d_mm=20, leaf_wetness_hours=20, gdd=100)


def _unfavorable_weather():
    return WeatherFeatures(temp_mean_c=2, humidity_mean_pct=30,
                           precip_7d_mm=0, leaf_wetness_hours=0, gdd=0)


def test_favorability_in_bounds():
    w = _favorable_weather()
    for disease in DISEASES.values():
        f = disease_favorability(w, disease)
        assert 0.0 <= f <= 1.0


def test_healthy_has_zero_favorability():
    w = _favorable_weather()
    assert disease_favorability(w, DISEASES["Corn_(maize)___healthy"]) == 0.0


def test_favorable_beats_unfavorable():
    gls = DISEASES["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"]
    assert disease_favorability(_favorable_weather(), gls) > \
           disease_favorability(_unfavorable_weather(), gls)


def test_crop_favorability_keys():
    fav = crop_favorability(_favorable_weather(), "Corn", ndvi=0.4)
    assert "corn_gray_leaf_spot" in fav
    assert all(0.0 <= v <= 1.0 for v in fav.values())
    # no healthy class in the favorability map
    assert "corn_healthy" not in fav


def test_risk_bands():
    assert risk_band(0.1) == LOW
    assert risk_band(0.5) == MODERATE
    assert risk_band(0.9) == HIGH


def test_recommend_returns_text():
    band, text = recommend_action("corn_gray_leaf_spot", is_healthy=False, score=0.8)
    assert band == HIGH
    assert isinstance(text, str) and len(text) > 0


def test_unknown_class_fallback():
    d = get_disease("Soybean___some_new_disease")
    assert d.crop in ("Corn", "Wheat")
    band, text = recommend_action(d.key, d.is_healthy, 0.5)
    assert isinstance(text, str) and len(text) > 0
