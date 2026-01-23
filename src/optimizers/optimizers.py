import cvxpy as cp
import numpy as np
from src.registries import OPTIMIZERS

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