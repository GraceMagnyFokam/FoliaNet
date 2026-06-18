"""Tiny YAML config loader with attribute access."""

from types import SimpleNamespace
from pathlib import Path
import yaml


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str = "configs/default.yaml") -> SimpleNamespace:
    data = yaml.safe_load(Path(path).read_text())
    return _to_namespace(data)
