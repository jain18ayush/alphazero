# src/concepts/registry.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from src.registries import CONCEPTS, COUNTERFACTUALS


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
    If counterfactual exists for this concept, returns matched pairs.
    Otherwise falls back to simple split.
    """
    check_fn = CONCEPTS.get(cfg.name)
    
    # Try to get counterfactual (returns None if not registered)
    try:
        cf_fn = COUNTERFACTUALS.get(cfg.name)
    except KeyError:
        cf_fn = None
    
    X_plus, X_minus = [], []
    
    for p in positions:
        has_concept = bool(check_fn(p["grid"], p["player"], **cfg.params))
        
        if has_concept:
            X_plus.append(p)
            if cf_fn:
                X_minus.append(cf_fn(p, **cfg.params))
    
    # If no counterfactual, fall back to unpaired negatives
    if cf_fn is None:
        X_minus = [p for p in positions if not check_fn(p["grid"], p["player"], **cfg.params)]
    
    return X_plus, X_minus
