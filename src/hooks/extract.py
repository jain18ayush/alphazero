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

import numpy as np
import torch
from typing import List, Dict


def get_single_activation(
    net: torch.nn.Module,
    grid: np.ndarray,
    player: int,
    layer_name: str,
) -> np.ndarray:
    """
    Get activation for a single board state.
    
    Args:
        net: OthelloNet
        grid: (n, n) board grid
        player: 1 or -1
        layer_name: name of layer to extract from
    
    Returns:
        Flattened activation vector
    """
    net.eval()
    device = next(net.parameters()).device
    modules = dict(net.named_modules())
    
    if layer_name not in modules:
        available = [name for name, _ in net.named_modules() if name]
        raise KeyError(f"Layer '{layer_name}' not found. Available: {available}")
    
    # OthelloNet expects input from perspective of player 1
    # So we multiply by player to get player-relative view
    x = torch.tensor(player * grid, dtype=torch.float32, device=device)
    x = x.unsqueeze(0)  # (1, n, n)
    
    activation = None
    
    def hook(module, inp, out):
        nonlocal activation
        if isinstance(out, tuple):
            out = out[0]
        # Flatten spatial dimensions if present
        if out.dim() > 2:
            out = out.view(out.size(0), -1)
        activation = out.detach().cpu().numpy()
    
    handle = modules[layer_name].register_forward_hook(hook)
    
    try:
        with torch.no_grad():
            net(x)
    finally:
        handle.remove()
    
    return activation[0]  # (dim,)


def extract_trajectory_activations(
    net: torch.nn.Module,
    states: List[Dict],
    layer_name: str,
) -> np.ndarray:
    """
    Get activations for each state in a trajectory.
    
    Args:
        net: OthelloNet
        states: List of {'grid': np.ndarray, 'player': int}
        layer_name: Layer to extract from
    
    Returns:
        (T, dim) array of activations
    """
    activations = []
    for state in states:
        z = get_single_activation(
            net, 
            state['grid'], 
            state['player'], 
            layer_name
        )
        activations.append(z)
    return np.stack(activations)


def extract_paired_trajectory_activations(
    net: torch.nn.Module,
    pairs: List[Dict],
    layer_name: str,
) -> List[Dict[str, np.ndarray]]:
    """
    Extract activations for all trajectory pairs.
    
    Args:
        net: OthelloNet
        pairs: List of trajectory pairs from collect_trajectory_pairs
        layer_name: Layer to extract from
    
    Returns:
        List of {'chosen': (T, dim), 'rejected': (T, dim)}
    """
    results = []
    
    for pair in pairs:
        z_chosen = extract_trajectory_activations(
            net, pair['chosen'], layer_name
        )
        z_rejected = extract_trajectory_activations(
            net, pair['rejected'], layer_name
        )
        
        results.append({
            'chosen': z_chosen,
            'rejected': z_rejected,
        })
    
    return results