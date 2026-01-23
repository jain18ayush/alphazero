# src/hooks/extract.py
from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch


def _to_model_input(positions: list[dict], device: torch.device) -> torch.Tensor:
    # Your convention: player-frame grid
    grids = np.array([p["player"] * p["grid"] for p in positions])
    x = torch.tensor(grids, dtype=torch.float32, device=device)
    return x


def _featurize(act: torch.Tensor) -> np.ndarray:
    # Default: flatten everything into (N, D)
    if isinstance(act, (tuple, list)):
        act = act[0]
    if act.dim() > 2:
        act = act.view(act.size(0), -1)
    return act.detach().cpu().numpy()


def extract_features_by_layer(
    net: torch.nn.Module,
    positions: list[dict],
    layers: List[dict],
) -> Dict[str, np.ndarray]:
    """
    One forward pass, capture all requested layers.

    Returns:
      {layer_name: np.ndarray of shape (N, D_layer)}
    """
    net.eval()
    modules = dict(net.named_modules())
    layer_names = [l["name"] for l in layers]

    # Build input
    device = next(net.parameters()).device
    x = _to_model_input(positions, device=device)

    feats: Dict[str, np.ndarray] = {}
    handles = []

    def make_hook(name: str):
        def hook_fn(module, inp, out):
            feats[name] = _featurize(out)
        return hook_fn

    # Register hooks
    for name in layer_names:
        if name not in modules:
            raise KeyError(f"Layer '{name}' not found.")
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    # Forward once
    try:
        with torch.no_grad():
            _ = net(x)
    finally:
        for h in handles:
            h.remove()

    # Ensure all layers captured
    missing = [n for n in layer_names if n not in feats]
    if missing:
        raise RuntimeError(f"Missing activations for layers: {missing}")

    return feats
