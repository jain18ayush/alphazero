# src/evaluators/evals.py
from typing import Dict, Any
import numpy as np
from src.registries import EVALUATORS

@EVALUATORS.register("acc")
def accuracy_at_0(ctx: dict, threshold: int) -> dict:
    p = ctx["Zp_test"] @ ctx["v"]
    m = ctx["Zm_test"] @ ctx["v"]
    acc = (np.mean(p > threshold) + np.mean(m <= threshold)) / 2.0
    return {"acc": float(acc)}

@EVALUATORS.register("hist")
def hist(ctx: dict, save_as: str = "hist.png") -> dict:
    import matplotlib.pyplot as plt

    p = ctx["Zp_test"] @ ctx["v"]
    m = ctx["Zm_test"] @ ctx["v"]

    out_dir = ctx.get("out_dir")
    if out_dir is None:
        raise ValueError("hist evaluator requires ctx['out_dir']")

    plt.figure()
    plt.hist(p, alpha=0.5, label="plus")
    plt.hist(m, alpha=0.5, label="minus")
    plt.legend()
    plt.title(f"Layer={ctx.get('layer', 'unknown')}")
    plt.savefig(out_dir / save_as)
    plt.close()

    return {"saved": save_as}

import numpy as np
from typing import Dict, Any, List


@EVALUATORS.register("trajectory_constraint_satisfaction")
def trajectory_constraint_satisfaction(ctx: dict, cfg: dict) -> Dict[str, Any]:
    """
    What fraction of (pair, timestep) constraints are satisfied?
    """
    activation_pairs = ctx['activation_pairs_test']
    v = ctx['v']
    margin = cfg.get('margin', 0.0)
    
    if v is None or len(activation_pairs) == 0:
        return {'constraint_satisfaction': 0.0, 'n_satisfied': 0, 'n_total': 0}
    
    satisfied = 0
    total = 0
    margins_list = []
    
    for pair in activation_pairs:
        z_c = pair['chosen']
        z_r = pair['rejected']
        T = min(len(z_c), len(z_r))
        
        for t in range(T):
            score_c = z_c[t] @ v
            score_r = z_r[t] @ v
            actual_margin = score_c - score_r
            
            margins_list.append(actual_margin)
            
            if actual_margin >= margin:
                satisfied += 1
            total += 1
    
    return {
        'constraint_satisfaction': float(satisfied / total) if total > 0 else 0.0,
        'n_satisfied': satisfied,
        'n_total': total,
        'mean_margin': float(np.mean(margins_list)) if margins_list else 0.0,
        'min_margin': float(np.min(margins_list)) if margins_list else 0.0,
    }


@EVALUATORS.register("trajectory_pair_accuracy")
def trajectory_pair_accuracy(ctx: dict, cfg: dict) -> Dict[str, Any]:
    """
    For each pair: is mean(chosen scores) > mean(rejected scores)?
    """
    activation_pairs = ctx['activation_pairs_test']
    v = ctx['v']
    
    if v is None or len(activation_pairs) == 0:
        return {'pair_accuracy': 0.0, 'n_correct': 0, 'n_total': 0}
    
    correct = 0
    total = len(activation_pairs)
    
    for pair in activation_pairs:
        z_c = pair['chosen']
        z_r = pair['rejected']
        
        mean_chosen = np.mean(z_c @ v)
        mean_rejected = np.mean(z_r @ v)
        
        if mean_chosen > mean_rejected:
            correct += 1
    
    return {
        'pair_accuracy': float(correct / total) if total > 0 else 0.0,
        'n_correct': correct,
        'n_total': total,
    }


@EVALUATORS.register("sparsity")
def sparsity(ctx: dict, cfg: dict) -> Dict[str, Any]:
    """Count non-zero dimensions in concept vector."""
    v = ctx['v']
    
    if v is None:
        return {'n_nonzero': 0, 'sparsity': 1.0, 'dim': 0}
    
    threshold = cfg.get('threshold', 1e-6)
    n_nonzero = int(np.sum(np.abs(v) > threshold))
    dim = len(v)
    
    return {
        'n_nonzero': n_nonzero,
        'sparsity': float(1 - n_nonzero / dim) if dim > 0 else 1.0,
        'dim': dim,
    }


def run_evals(ctx: dict, eval_cfg: dict) -> Dict[str, Any]:
    """Run all configured evaluations."""
    results = {}
    
    for run_cfg in eval_cfg.get('runs', []):
        name = run_cfg['name']
        eval_fn = EVALUATORS.get(name)
        
        if eval_fn is None:
            print(f"  Warning: evaluator '{name}' not found, skipping")
            continue
        
        result = eval_fn(ctx, run_cfg)
        results[name] = result
    
    return results