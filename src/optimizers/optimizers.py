import cvxpy as cp
import numpy as np
from src.registries import OPTIMIZERS
from typing import List, Dict

@OPTIMIZERS.register("convex")
def find_concept_vector(Z_plus: np.ndarray, Z_minus: np.ndarray, opt_cfg: dict):
    """
    Find sparse concept vector via convex optimization.
    
    Args:
        Z_plus: (n_plus, dim) activations for positive class
        Z_minus: (n_minus, dim) activations for negative class  
        margin: minimum separation (v·z⁺ ≥ v·z⁻ + margin)
    
    Returns:
        v: (dim,) concept vector
    """
    n_plus, dim = Z_plus.shape
    n_minus = Z_minus.shape[0]
    
    # Variable we're solving for
    v = cp.Variable(dim)
    
    # Constraints: every positive beats every negative
    constraints = []
    for i in range(n_plus):
        for j in range(n_minus):
            constraints.append(v @ Z_plus[i] >= v @ Z_minus[j] + opt_cfg['l1_margin'])
    
    # Objective: minimize L1 norm
    objective = cp.Minimize(cp.norm(v, 1))
    
    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    if problem.status == 'optimal':
        return v.value
    else:
        print(f"Optimization failed: {problem.status}")
        return None


@OPTIMIZERS.register("dynamic_convex")
def dynamic_convex_optimizer(
    activation_pairs: List[Dict[str, np.ndarray]],
    cfg: dict,
) -> np.ndarray:
    """
    Find sparse concept vector for dynamic (trajectory) concepts.
    
    Constraint: v · z_chosen[t] >= v · z_rejected[t] + margin
                for all timesteps t, for all pairs
    
    Args:
        activation_pairs: list of {'chosen': (T, dim), 'rejected': (T, dim)}
        cfg: optimization config with 'margin' key
    
    Returns:
        v: (dim,) concept vector
    """
    if len(activation_pairs) == 0:
        raise ValueError("No activation pairs provided")
    
    # Get dimension from first pair
    dim = activation_pairs[0]['chosen'].shape[1]
    margin = cfg.get('margin', 0.0)
    
    v = cp.Variable(dim)
    constraints = []
    
    n_constraints = 0
    for pair in activation_pairs:
        z_c = pair['chosen']   # (T_c, dim)
        z_r = pair['rejected']  # (T_r, dim)
        
        T = min(len(z_c), len(z_r))
        
        for t in range(T):
            constraints.append(v @ z_c[t] >= v @ z_r[t] + margin)
            n_constraints += 1
    
    print(f"  Dynamic convex: {len(activation_pairs)} pairs, {n_constraints} constraints, dim={dim}")
    
    if n_constraints == 0:
        print("  Warning: No constraints generated")
        return np.zeros(dim)
    
    problem = cp.Problem(cp.Minimize(cp.norm(v, 1)), constraints)
    
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except Exception as e:
        print(f"  ECOS failed: {e}, trying SCS...")
        problem.solve(solver=cp.SCS, verbose=False)
    
    if problem.status == 'optimal' or problem.status == 'optimal_inaccurate':
        return v.value
    else:
        print(f"  Optimization status: {problem.status}")
        return v.value if v.value is not None else np.zeros(dim)


@OPTIMIZERS.register("dynamic_convex_single_player")
def dynamic_convex_single_player(
    activation_pairs: List[Dict[str, np.ndarray]],
    cfg: dict,
) -> np.ndarray:
    """
    Dynamic concept optimizer with multiple subpar trajectories (Equation 5).
    Only uses every other timestep for single player perspective.

    From Schut et al.: "for concepts for a single player, use {z_2t}"
    Equation 5: v · z_t^+ >= v · z_t^j + margin for all j (all subpar)
    """
    if len(activation_pairs) == 0:
        raise ValueError("No activation pairs provided")
    
    dim = activation_pairs[0]['chosen'].shape[1]
    margin = cfg.get('margin', 0.0)
    
    v = cp.Variable(dim)
    constraints = []
    
    n_constraints = 0
    total_subpar = 0

    for pair in activation_pairs:
        z_c = pair['chosen']          # (T, dim)
        z_r_list = pair['rejected']   # List of (T, dim) - EQUATION 5 FIX

        # Loop through ALL subpar trajectories (not just one!)
        for z_r in z_r_list:
            T = min(len(z_c), len(z_r))

            # Only even timesteps (player to move at root)
            for t in range(0, T, 2):
                constraints.append(v @ z_c[t] >= v @ z_r[t] + margin)
                n_constraints += 1

        total_subpar += len(z_r_list)

    avg_subpar = total_subpar / len(activation_pairs) if len(activation_pairs) > 0 else 0
    print(f"  Dynamic convex (single player, Equation 5):")
    print(f"    {len(activation_pairs)} pairs × {avg_subpar:.1f} subpar × ~2.5 timesteps (even only)")
    print(f"    = {n_constraints} constraints on dim={dim}")
    
    if n_constraints == 0:
        return np.zeros(dim)
    
    problem = cp.Problem(cp.Minimize(cp.norm(v, 1)), constraints)
    
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except Exception as e:
        print(f"  ECOS failed ({e}), trying SCS...")
        problem.solve(solver=cp.SCS, verbose=False)

    if problem.status == 'optimal' or problem.status == 'optimal_inaccurate':
        print(f"  Solver: {problem.status}, objective: {problem.value:.4f}")
        return v.value
    else:
        print(f"  Optimization status: {problem.status}")
        return v.value if v.value is not None else np.zeros(dim)