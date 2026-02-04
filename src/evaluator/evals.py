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
def trajectory_constraint_satisfaction(ctx: dict, margin: float = 0.0, **kwargs) -> Dict[str, Any]:
    """
    What fraction of (pair, timestep, subpar) constraints are satisfied?
    Updated for Equation 5: handles multiple subpar trajectories per pair.
    """
    activation_pairs = ctx['activation_pairs_test']
    v = ctx['v']

    if v is None or len(activation_pairs) == 0:
        return {'constraint_satisfaction': 0.0, 'n_satisfied': 0, 'n_total': 0}

    satisfied = 0
    total = 0
    margins_list = []

    for pair in activation_pairs:
        z_c = pair['chosen']
        z_r_list = pair['rejected']  # NOW A LIST (Equation 5)

        # Loop through all subpar trajectories
        for z_r in z_r_list:
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
def trajectory_pair_accuracy(ctx: dict, **kwargs) -> Dict[str, Any]:
    """
    For each (pair, subpar): is mean(chosen scores) > mean(rejected scores)?
    Updated for Equation 5: handles multiple subpar trajectories per pair.
    """
    activation_pairs = ctx['activation_pairs_test']
    v = ctx['v']

    if v is None or len(activation_pairs) == 0:
        return {'pair_accuracy': 0.0, 'n_correct': 0, 'n_total': 0}

    correct = 0
    total_comparisons = 0

    for pair in activation_pairs:
        z_c = pair['chosen']
        z_r_list = pair['rejected']  # NOW A LIST (Equation 5)

        mean_chosen = np.mean(z_c @ v)

        # Each subpar is a separate comparison
        for z_r in z_r_list:
            mean_rejected = np.mean(z_r @ v)

            if mean_chosen > mean_rejected:
                correct += 1
            total_comparisons += 1

    return {
        'pair_accuracy': float(correct / total_comparisons) if total_comparisons > 0 else 0.0,
        'n_correct': correct,
        'n_total': total_comparisons,
    }


@EVALUATORS.register("sparsity")
def sparsity(ctx: dict, threshold: float = 1e-6, **kwargs) -> Dict[str, Any]:
    """Count non-zero dimensions in concept vector."""
    v = ctx['v']

    if v is None:
        return {'n_nonzero': 0, 'sparsity': 1.0, 'dim': 0}
    n_nonzero = int(np.sum(np.abs(v) > threshold))
    dim = len(v)

    return {
        'n_nonzero': n_nonzero,
        'sparsity': float(1 - n_nonzero / dim) if dim > 0 else 1.0,
        'dim': dim,
    }


@EVALUATORS.register("activation_stats")
def activation_stats(ctx: dict, **kwargs) -> Dict[str, Any]:
    """
    Understand activation scale to set appropriate margin (Issue #5 fix).
    Collects all activations from test set and computes statistics.
    """
    activation_pairs = ctx.get('activation_pairs_test', [])

    if len(activation_pairs) == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'recommended_margin': 0.0}

    all_activations = []

    for pair in activation_pairs:
        # Collect chosen activations
        all_activations.append(pair['chosen'].flatten())

        # Collect all rejected activations
        for z_r in pair['rejected']:
            all_activations.append(z_r.flatten())

    all_acts = np.concatenate(all_activations)

    return {
        'mean': float(np.mean(all_acts)),
        'std': float(np.std(all_acts)),
        'min': float(np.min(all_acts)),
        'max': float(np.max(all_acts)),
        'median': float(np.median(all_acts)),
        'recommended_margin': float(0.1 * np.std(all_acts)),  # 10% of std
    }


@EVALUATORS.register("spine_pair_separation")
def spine_pair_separation(ctx: dict, **kwargs) -> Dict[str, Any]:
    """
    Evaluate how well the concept vector separates spine Z+ vs Z- pairs.

    Reads ctx['v'], ctx['Z_plus'], ctx['Z_minus'], ctx['out_dir'].
    Computes projections, constraint satisfaction rate, margin stats.
    Saves histogram plot to out_dir / "pair_separation.png".
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    v = ctx['v']
    Z_plus = ctx['Z_plus']
    Z_minus = ctx['Z_minus']
    out_dir = ctx.get('out_dir')

    if v is None or len(Z_plus) == 0:
        return {'constraint_satisfaction': 0.0, 'n_satisfied': 0, 'n_total': 0,
                'mean_margin': 0.0, 'min_margin': 0.0}

    v = np.asarray(v).reshape(-1)
    Z_plus = np.asarray(Z_plus)
    Z_minus = np.asarray(Z_minus)

    # Normalize v for scale-invariant projections
    v_norm = np.linalg.norm(v)
    v_unit = v / v_norm if v_norm > 0 else v

    p = Z_plus @ v_unit
    m = Z_minus @ v_unit
    d = p - m  # per-pair margin

    n_total = len(d)
    n_satisfied = int(np.sum(d >= 0))
    constraint_satisfaction = float(n_satisfied / n_total) if n_total > 0 else 0.0

    result = {
        'constraint_satisfaction': constraint_satisfaction,
        'n_satisfied': n_satisfied,
        'n_total': n_total,
        'mean_margin': float(np.mean(d)) if n_total > 0 else 0.0,
        'min_margin': float(np.min(d)) if n_total > 0 else 0.0,
        'median_margin': float(np.median(d)) if n_total > 0 else 0.0,
        'mean_plus_proj': float(np.mean(p)) if n_total > 0 else 0.0,
        'mean_minus_proj': float(np.mean(m)) if n_total > 0 else 0.0,
    }

    # Save histogram
    if out_dir is not None:
        from pathlib import Path
        out_dir = Path(out_dir)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Overlaid projections
        axes[0].hist(p, bins=30, alpha=0.6, label="Z+ (main)")
        axes[0].hist(m, bins=30, alpha=0.6, label="Z- (alt)")
        axes[0].set_xlabel("projection onto v")
        axes[0].set_ylabel("count")
        axes[0].set_title("Projections: main vs alternative")
        axes[0].legend()

        # Per-pair margin
        axes[1].hist(d, bins=30, alpha=0.8)
        axes[1].axvline(0, linestyle="--", color="red", linewidth=1, label="margin=0")
        axes[1].set_xlabel("margin (plus - minus)")
        axes[1].set_ylabel("count")
        axes[1].set_title(f"Pair margins ({constraint_satisfaction:.0%} satisfied)")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(out_dir / "pair_separation.png", dpi=150)
        plt.close()
        result['saved'] = "pair_separation.png"

    return result


@EVALUATORS.register("trajectory_dynamics")
def trajectory_dynamics(ctx: dict, threshold: float = 0.01, **kwargs) -> Dict[str, Any]:
    """
    Validate that concept actually changes over trajectory (Issue #2 fix).
    A true dynamic concept should show temporal variance in activations.
    If variance is low, concept might just be detecting static properties.
    """
    activation_pairs = ctx.get('activation_pairs_test', [])
    v = ctx.get('v')

    if v is None or len(activation_pairs) == 0:
        return {
            'temporal_variance': 0.0,
            'is_dynamic': False,
            'chosen_variance': 0.0,
            'rejected_variance': 0.0,
            'avg_chosen_trend': 0.0,
            'avg_rejected_trend': 0.0,
        }

    chosen_variances = []
    rejected_variances = []
    chosen_trends = []
    rejected_trends = []

    for pair in activation_pairs:
        # Analyze chosen trajectory dynamics
        z_c = pair['chosen']
        scores_c = z_c @ v  # (T,) activations over time

        chosen_variances.append(np.var(scores_c))

        # Compute trend (slope) - is it increasing/decreasing?
        if len(scores_c) > 1:
            trend_c = np.polyfit(range(len(scores_c)), scores_c, 1)[0]  # slope
            chosen_trends.append(trend_c)

        # Analyze rejected trajectories dynamics
        for z_r in pair['rejected']:
            scores_r = z_r @ v
            rejected_variances.append(np.var(scores_r))

            if len(scores_r) > 1:
                trend_r = np.polyfit(range(len(scores_r)), scores_r, 1)[0]
                rejected_trends.append(trend_r)

    avg_chosen_var = np.mean(chosen_variances) if chosen_variances else 0.0
    avg_rejected_var = np.mean(rejected_variances) if rejected_variances else 0.0
    avg_variance = (avg_chosen_var + avg_rejected_var) / 2

    return {
        'temporal_variance': float(avg_variance),
        'is_dynamic': bool(avg_variance > threshold),  # True if concept changes over time
        'chosen_variance': float(avg_chosen_var),
        'rejected_variance': float(avg_rejected_var),
        'avg_chosen_trend': float(np.mean(chosen_trends)) if chosen_trends else 0.0,
        'avg_rejected_trend': float(np.mean(rejected_trends)) if rejected_trends else 0.0,
    }


