"""
Disease metadata for FoliaNet.

Each entry is keyed by the dataset class-folder name.
`favorability` parameters drive the agronomic environmental-risk module in
`folianet/features/environment.py`. They are *heuristics* drawn from general
plant-pathology guidance (warm, humid, extended-leaf-wetness conditions favor
these foliar fungi) and are meant to be tuned against your own field data.

Classes are configured by crop. To add a crop or disease, register an entry
keyed by its class-folder name; training, inference, and recommendations pick it
up automatically.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class Favorability:
    """Simple agronomic suitability window for a foliar fungal disease."""
    t_min: float          # below this temp (C), little development
    t_opt_low: float      # lower bound of optimal temp band
    t_opt_high: float     # upper bound of optimal temp band
    t_max: float          # above this temp, little development
    humidity_min: float   # RH (%) below which risk is suppressed
    wetness_hours: float  # leaf-wetness hours that saturate the wetness term


_DEFAULT_FAVORABILITY = Favorability(15, 22, 28, 35, 75, 12)


@dataclass(frozen=True)
class Disease:
    key: str              # stable short id
    crop: str             # "Corn" | "Wheat"
    display_name: str     # human-readable stressor name
    pathogen: str         # causal organism
    is_healthy: bool = False
    favorability: Favorability = field(default_factory=lambda: _DEFAULT_FAVORABILITY)


# Keyed by dataset class-folder name.
DISEASES: Dict[str, Disease] = {
    # ---- Corn / maize ----
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": Disease(
        key="corn_gray_leaf_spot",
        crop="Corn",
        display_name="Gray leaf spot",
        pathogen="Cercospora zeae-maydis",
        favorability=Favorability(18, 24, 30, 35, 85, 12),
    ),
    "Corn_(maize)___Common_rust_": Disease(
        key="corn_common_rust",
        crop="Corn",
        display_name="Common rust",
        pathogen="Puccinia sorghi",
        favorability=Favorability(10, 16, 23, 30, 80, 6),
    ),
    "Corn_(maize)___Northern_Leaf_Blight": Disease(
        key="corn_northern_leaf_blight",
        crop="Corn",
        display_name="Northern leaf blight",
        pathogen="Exserohilum turcicum",
        favorability=Favorability(15, 18, 27, 32, 85, 6),
    ),
    "Corn_(maize)___healthy": Disease(
        key="corn_healthy",
        crop="Corn",
        display_name="Healthy",
        pathogen="—",
        is_healthy=True,
        favorability=Favorability(0, 0, 0, 0, 0, 0),
    ),

    # ---- Wheat ----
    "Wheat___Leaf_rust": Disease(
        key="wheat_leaf_rust",
        crop="Wheat",
        display_name="Leaf (brown) rust",
        pathogen="Puccinia triticina",
        favorability=Favorability(10, 15, 22, 30, 80, 6),
    ),
    "Wheat___Stripe_rust": Disease(
        key="wheat_stripe_rust",
        crop="Wheat",
        display_name="Stripe (yellow) rust",
        pathogen="Puccinia striiformis",
        favorability=Favorability(5, 10, 18, 25, 85, 6),
    ),
    "Wheat___Septoria": Disease(
        key="wheat_septoria",
        crop="Wheat",
        display_name="Septoria leaf blotch",
        pathogen="Zymoseptoria tritici",
        favorability=Favorability(10, 16, 22, 28, 90, 24),
    ),
    "Wheat___healthy": Disease(
        key="wheat_healthy",
        crop="Wheat",
        display_name="Healthy",
        pathogen="—",
        is_healthy=True,
        favorability=Favorability(0, 0, 0, 0, 0, 0),
    ),
}


def get_disease(class_name: str) -> Disease:
    """Look up disease metadata, tolerating unknown classes gracefully."""
    if class_name in DISEASES:
        return DISEASES[class_name]
    # Fallback for classes not yet registered: infer crop, treat as unknown stressor.
    crop = "Wheat" if "wheat" in class_name.lower() else "Corn"
    healthy = "healthy" in class_name.lower()
    return Disease(
        key=class_name.lower().replace(" ", "_"),
        crop=crop,
        display_name=class_name.split("___")[-1].replace("_", " ").strip() or class_name,
        pathogen="unknown",
        is_healthy=healthy,
        favorability=Favorability(15, 22, 28, 35, 80, 12),
    )
