"""
Recommendation layer: maps (stressor, risk band) -> a recommended next action.

These are advisory, extension-style messages. They are intentionally general;
real fungicide timing, rates, and product choice depend on local regulations,
hybrid/variety susceptibility, and growth stage. Always defer to a local
agronomist / extension service. Pure Python -> unit testable.
"""

from typing import Tuple

LOW, MODERATE, HIGH = "low", "moderate", "high"


def risk_band(score: float) -> str:
    if score < 0.33:
        return LOW
    if score < 0.66:
        return MODERATE
    return HIGH


# Per-stressor action text by band. Keyed by Disease.key.
_ACTIONS = {
    "corn_gray_leaf_spot": {
        LOW: "Conditions are not currently favorable. Continue routine scouting of the lower-mid canopy.",
        MODERATE: "Scout the mid-canopy for rectangular gray lesions. If a susceptible hybrid and disease is present below the ear leaf, plan a fungicide window around VT-R1.",
        HIGH: "High pressure. Prioritize scouting now; for susceptible hybrids consider a triazole/strobilurin application at VT-R1, and weigh resistant hybrids next season.",
    },
    "corn_common_rust": {
        LOW: "Low pressure. Common rust rarely warrants treatment in field corn — monitor only.",
        MODERATE: "Track pustule density on upper leaves. Treatment is seldom economical in field corn but may matter for seed/sweet corn.",
        HIGH: "Heavy rust. Treatment is usually justified only in seed or sweet corn, or on highly susceptible hybrids before tasseling.",
    },
    "corn_northern_leaf_blight": {
        LOW: "Conditions marginal. Scout the lower canopy for cigar-shaped lesions.",
        MODERATE: "Scout for elongated gray-green lesions on lower leaves. If lesions reach the ear leaf before tassel on a susceptible hybrid, a fungicide is often warranted.",
        HIGH: "High pressure. Strongly consider a fungicide if lesions are present below the ear leaf pre-tassel; plan resistant hybrids and residue management going forward.",
    },
    "wheat_leaf_rust": {
        LOW: "Low pressure. Monitor flag-leaf emergence.",
        MODERATE: "Scout the flag leaf and the leaf below. Protect the flag leaf with a fungicide if rust appears before/at heading on a susceptible variety.",
        HIGH: "High pressure. Protect the flag leaf promptly; flag-leaf health drives yield. Favor resistant varieties next season.",
    },
    "wheat_stripe_rust": {
        LOW: "Cool, wet conditions can flare stripe rust quickly — keep monitoring even at low risk.",
        MODERATE: "Scout for yellow striping on leaves. Stripe rust escalates fast in cool, moist weather; be ready to spray the flag leaf.",
        HIGH: "High, fast-moving pressure. Protect the flag leaf without delay and re-scout within days.",
    },
    "wheat_septoria": {
        LOW: "Low pressure. Monitor lower leaves after rain events.",
        MODERATE: "Wet-splash disease — scout lower/mid leaves for speckled lesions with pycnidia. Consider flag-leaf protection if wet weather persists.",
        HIGH: "High pressure under prolonged leaf wetness. Protect the flag leaf and prioritize residue management and variety selection.",
    },
}

_HEALTHY_ACTION = (
    "No disease detected and conditions are not strongly favorable. "
    "Continue your normal scouting schedule."
)

_GENERIC = {
    LOW: "Low estimated risk. Continue routine scouting.",
    MODERATE: "Moderate risk. Increase scouting frequency and reassess as conditions evolve.",
    HIGH: "High risk. Scout immediately and consult your agronomist about a protective fungicide.",
}

_DISCLAIMER = (
    " (Advisory only — confirm with a local agronomist/extension service before acting.)"
)


def recommend_action(stressor_key: str, is_healthy: bool, score: float) -> Tuple[str, str]:
    """Return (band, action_text)."""
    band = risk_band(score)
    if is_healthy:
        return band, _HEALTHY_ACTION + _DISCLAIMER
    table = _ACTIONS.get(stressor_key)
    text = table[band] if table else _GENERIC[band]
    return band, text + _DISCLAIMER
