# src/concepts/registry.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from src.registries import CONCEPTS


@dataclass
class ConceptConfig:
    name: str
    params: Dict[str, Any]

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        fn = d.pop("name")
        return cls(name=fn, params=d)

def split_by_concept(
    positions: List[dict],
    cfg: ConceptConfig,
) -> Tuple[List[dict], List[dict]]:
    """
    Applies a registered concept function to positions.

    Returns:
      X_plus: positions where concept is True
      X_minus: positions where concept is False
    """
    fn = CONCEPTS.get(cfg.name)

    X_plus, X_minus = [], []
    for p in positions:
        has_concept = bool(fn(p["grid"], p["player"], **cfg.params))
        (X_plus if has_concept else X_minus).append(p)

    return X_plus, X_minus
