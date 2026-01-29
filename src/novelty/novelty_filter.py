# src/novelty/novelty_filter.py

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import json


def compute_svd_basis(Z: np.ndarray, max_rank: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SVD basis vectors for a set of activations.
    
    Args:
        Z: (n_samples, dim) activation matrix
        max_rank: Optional limit on number of basis vectors to keep
    
    Returns:
        U: (n_samples, rank) left singular vectors
        Sigma: (rank,) singular values
    """
    U, Sigma, Vt = np.linalg.svd(Z, full_matrices=False)
    
    if max_rank is not None:
        U = U[:, :max_rank]
        Sigma = Sigma[:max_rank]
    
    return U, Sigma


def compute_rank(Z: np.ndarray, threshold: float = 1e-10) -> int:
    """
    Compute effective rank of activation matrix.
    
    Args:
        Z: (n_samples, dim) activation matrix
        threshold: Singular values below this are considered zero
    
    Returns:
        Effective rank
    """
    _, Sigma, _ = np.linalg.svd(Z, full_matrices=False)
    rank = np.sum(Sigma > threshold)
    return rank


def reconstruction_error(v: np.ndarray, U: np.ndarray, k: int) -> float:
    """
    Compute reconstruction error of concept vector v using top k basis vectors from U.
    
    This is: ||v - sum(beta_i * u_i)||^2
    
    Args:
        v: (dim,) concept vector
        U: (dim, n_basis) basis vectors (columns)
        k: Number of basis vectors to use
    
    Returns:
        Reconstruction error (L2 norm squared)
    """
    if k > U.shape[1]:
        k = U.shape[1]
    
    if k == 0:
        return np.linalg.norm(v) ** 2
    
    U_k = U[:, :k]  # (dim, k)
    
    # Solve least squares: beta = (U_k^T U_k)^{-1} U_k^T v
    # Or equivalently: beta = U_k^T v (since U is orthonormal)
    beta = U_k.T @ v  # (k,)
    
    # Reconstruction: v_hat = U_k @ beta
    v_hat = U_k @ beta  # (dim,)
    
    # Error: ||v - v_hat||^2
    error = np.linalg.norm(v - v_hat) ** 2
    
    return error


def compute_novelty_score(
    v: np.ndarray,
    U_human: np.ndarray,
    U_az: np.ndarray,
    k_values: List[int]
) -> Tuple[List[float], bool]:
    """
    Compute novelty scores for concept vector v across different k values.
    
    Novelty score = error_human - error_az
    Positive score means AZ basis explains v better than human basis.
    
    Args:
        v: (dim,) concept vector
        U_human: (dim, rank_human) human basis vectors
        U_az: (dim, rank_az) AZ basis vectors
        k_values: List of k values to test
    
    Returns:
        novelty_scores: List of novelty scores for each k
        is_novel: True if AZ basis is better for ALL k values
    """
    novelty_scores = []
    
    for k in k_values:
        error_human = reconstruction_error(v, U_human, k)
        error_az = reconstruction_error(v, U_az, k)
        
        novelty_score = error_human - error_az
        novelty_scores.append(novelty_score)
    
    # Accept if AZ basis is consistently better (all scores positive)
    is_novel = all(score > 0 for score in novelty_scores)
    
    return novelty_scores, is_novel


def collect_activations_from_positions(
    net,
    positions: List[dict],
    layer: str,
    max_samples: int = 17184
) -> np.ndarray:
    """
    Extract activations for a set of positions at a given layer.

    Args:
        net: Neural network model
        positions: List of position dicts with 'grid' and 'player'
        layer: Layer name to extract from
        max_samples: Maximum number of samples to use

    Returns:
        Z: (n_samples, dim) activation matrix (flattened if needed)
    """
    from src.hooks.extract import get_single_activation

    # Subsample if needed
    if len(positions) > max_samples:
        idx = np.random.choice(len(positions), max_samples, replace=False)
        positions = [positions[i] for i in idx]

    activations = []

    for pos in positions:
        # get_single_activation returns flattened activations
        act = get_single_activation(
            net,
            pos['grid'],
            pos['player'],
            layer
        )
        activations.append(act)

    Z = np.stack(activations)  # (n_samples, dim)
    return Z


class NoveltyFilter:
    """
    Filter for identifying novel concepts using SVD-based reconstruction.
    """
    
    def __init__(
        self,
        U_human: np.ndarray,
        U_az: np.ndarray,
        k_values: Optional[List[int]] = None,
        save_dir: Optional[Path] = None
    ):
        """
        Args:
            U_human: (dim, rank_human) human game basis vectors
            U_az: (dim, rank_az) AZ game basis vectors
            k_values: List of k values to test (default: geometric progression)
            save_dir: Directory to save basis vectors and stats
        """
        self.U_human = U_human
        self.U_az = U_az
        self.dim = U_human.shape[0]
        self.rank_human = U_human.shape[1]
        self.rank_az = U_az.shape[1]
        
        # Default k values: geometric progression from 1 to min(ranks)
        if k_values is None:
            max_k = min(self.rank_human, self.rank_az)
            # Generate: [1, 2, 4, 8, 16, ..., max_k]
            k_values = []
            k = 1
            while k <= max_k:
                k_values.append(k)
                k = min(k * 2, max_k)
            if k_values[-1] != max_k:
                k_values.append(max_k)
        
        self.k_values = k_values
        self.save_dir = save_dir
        
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            self._save_basis_info()
    
    def _save_basis_info(self):
        """Save basis vector statistics."""
        info = {
            'dim': int(self.dim),
            'rank_human': int(self.rank_human),
            'rank_az': int(self.rank_az),
            'k_values': [int(k) for k in self.k_values],
        }
        
        with open(self.save_dir / 'basis_info.json', 'w') as f:
            json.dump(info, f, indent=2)
    
    def filter_concept(self, v: np.ndarray) -> Tuple[bool, dict]:
        """
        Determine if concept vector v is novel.
        
        Args:
            v: (dim,) concept vector
        
        Returns:
            is_novel: True if concept should be kept
            stats: Dictionary with novelty scores and details
        """
        novelty_scores, is_novel = compute_novelty_score(
            v, self.U_human, self.U_az, self.k_values
        )
        
        stats = {
            'is_novel': bool(is_novel),
            'novelty_scores': [float(s) for s in novelty_scores],
            'k_values': [int(k) for k in self.k_values],
        }
        
        return is_novel, stats


def build_novelty_filter(
    net,
    human_positions: List[dict],
    az_positions: List[dict],
    layer: str,
    n_samples: int = 17184,
    save_dir: Optional[Path] = None,
) -> NoveltyFilter:
    """
    Build novelty filter by computing SVD bases for human and AZ games.
    
    Args:
        net: Neural network model
        human_positions: List of positions from human games
        az_positions: List of positions from AZ games
        layer: Layer name to extract activations from
        n_samples: Number of samples to use for each
        save_dir: Directory to save basis vectors
    
    Returns:
        NoveltyFilter object
    """
    print(f"\nBuilding novelty filter for layer: {layer}")
    
    # Collect activations
    print(f"  Collecting human activations ({len(human_positions)} positions)...")
    Z_human = collect_activations_from_positions(net, human_positions, layer, n_samples)
    
    print(f"  Collecting AZ activations ({len(az_positions)} positions)...")
    Z_az = collect_activations_from_positions(net, az_positions, layer, n_samples)
    
    print(f"  Activation shapes: human={Z_human.shape}, az={Z_az.shape}")
    
    # Compute ranks
    rank_human = compute_rank(Z_human)
    rank_az = compute_rank(Z_az)
    
    print(f"  Ranks: human={rank_human}, az={rank_az}")
    
    # Compute SVD bases
    print("  Computing SVD bases...")
    U_human, Sigma_human = compute_svd_basis(Z_human)
    U_az, Sigma_az = compute_svd_basis(Z_az)
    
    # Transpose to get (dim, rank) format
    U_human = U_human.T  # Now (n_samples, rank) -> (dim, rank)
    U_az = U_az.T
    
    # Wait, that's wrong. Let me fix:
    # SVD gives us: Z = U @ Sigma @ Vt
    # We want the column space basis, which is U
    # U is (n_samples, rank), but we need (dim, rank)
    # Actually, we need the RIGHT singular vectors (rows of Vt)
    
    # Let me recalculate properly:
    print("  Recalculating SVD properly...")
    _, _, Vt_human = np.linalg.svd(Z_human, full_matrices=False)
    _, _, Vt_az = np.linalg.svd(Z_az, full_matrices=False)
    
    # Vt is (rank, dim), we want (dim, rank)
    U_human = Vt_human.T  # (dim, rank)
    U_az = Vt_az.T  # (dim, rank)
    
    print(f"  Basis shapes: human={U_human.shape}, az={U_az.shape}")
    
    # Save basis vectors if requested
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'U_human.npy', U_human)
        np.save(save_dir / 'U_az.npy', U_az)
        np.save(save_dir / 'Sigma_human.npy', Sigma_human)
        np.save(save_dir / 'Sigma_az.npy', Sigma_az)
    
    return NoveltyFilter(U_human, U_az, save_dir=save_dir)