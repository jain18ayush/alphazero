import numpy as np 
from typing import List, Tuple, Optional

def compute_basis(activations: np.ndarray) -> np.ndarray:
    """
    Compute the basis of the activations.
    """
    U, Sigma, Vt = np.linalg.svd(activations, full_matrices=False)
    return Vt.T # returns the basis vectors 

def is_novel(
    v: np.ndarray,
    V_human: np.ndarray,
    V_az: np.ndarray,
    k_values: List[int]
) -> Tuple[bool, List[float]]:
    """
    Check if concept vector v is novel.
    
    Args:
        v: (dim,) concept vector
        V_human: (dim, rank) basis from human games  
        V_az: (dim, rank) basis from AZ games
        k_values: list of k values to test (e.g., [1, 2, 4, 8, 16, 32])
        
    Returns:
        novel: True if ALL scores > 0
        scores: novelty score for each k
    """
    scores = []
    
    for k in k_values:
        err_human = reconstruction_error(v, V_human, k)
        err_az = reconstruction_error(v, V_az, k)
        score = err_human - err_az
        scores.append(score)
    
    # Novel only if ALL scores are positive
    novel = all(s > 0 for s in scores)
    
    return novel, scores

def reconstruction_error(v: np.ndarray, V: np.ndarray, k: int) -> float:
    """
    How well can we reconstruct v using the first k basis vectors?
    
    Args:
        v: (dim,) - the concept vector
        V: (dim, rank) - basis vectors as columns
        k: number of basis vectors to use
        
    Returns:
        error: squared L2 norm of (v - reconstruction)
    """
    # Use only first k columns
    V_k = V[:, :k]  # (dim, k)
    
    # Project: find coefficients
    coeffs = V_k.T @ v  # (k,)
    
    # Reconstruct
    v_proj = V_k @ coeffs  # (dim,)
    
    # Error = how much is "left over"
    error = np.linalg.norm(v - v_proj) ** 2
    
    return error


class NoveltyFilter:
    """
    Filter concept vectors to keep only novel ones.
    
    Usage:
        nf = NoveltyFilter(Z_human, Z_az)
        novel_concepts = nf.filter(concept_vectors)
    """
    
    def __init__(self, V_human: np.ndarray = None, V_az: np.ndarray = None, Z_human: np.ndarray = None, Z_az: np.ndarray = None):
        """
        Construct a NoveltyFilter using either the bases V_human/V_az directly,
        or by providing activation matrices Z_human/Z_az to compute the bases.

        Args:
            V_human: (dim, r1) basis for human activations (optional)
            V_az: (dim, r2) basis for AZ activations (optional)
            Z_human: (n_samples, dim) activations from human games (optional)
            Z_az: (n_samples, dim) activations from AZ games (optional)
        """
        if V_human is not None and V_az is not None:
            self.V_human = V_human
            self.V_az = V_az
        elif Z_human is not None and Z_az is not None:
            self.V_human = compute_basis(Z_human)
            self.V_az = compute_basis(Z_az)
            np.save("novelty_V_human.npy", self.V_human)
            np.save("novelty_V_az.npy", self.V_az)
        else:
            raise ValueError("Must provide either both V_human and V_az, or both Z_human and Z_az.")

        # k values: geometric progression up to min rank
        max_k = min(self.V_human.shape[1], self.V_az.shape[1])
        self.k_values = []
        k = 1
        while k <= max_k:
            self.k_values.append(k)
            k *= 2
        
    def check(self, v: np.ndarray) -> Tuple[bool, List[float]]:
        """Check if single concept is novel."""
        return is_novel(v, self.V_human, self.V_az, self.k_values)
    
    def filter(self, concepts: List[np.ndarray]) -> List[np.ndarray]:
        """Filter list of concepts, keeping only novel ones."""
        return [v for v in concepts if self.check(v)[0]]