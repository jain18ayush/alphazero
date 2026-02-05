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

    When ctx contains 'Z_plus_test'/'Z_minus_test', reports both train and
    test metrics (test = held-out pairs not seen during optimization).
    Falls back to ctx['Z_plus']/ctx['Z_minus'] only when test split absent.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    v = ctx['v']
    Z_plus_train = ctx['Z_plus']
    Z_minus_train = ctx['Z_minus']
    Z_plus_test = ctx.get('Z_plus_test')
    Z_minus_test = ctx.get('Z_minus_test')
    out_dir = ctx.get('out_dir')

    has_test = Z_plus_test is not None and Z_minus_test is not None and len(Z_plus_test) > 0

    if v is None or len(Z_plus_train) == 0:
        return {'test_constraint_satisfaction': 0.0, 'test_n_total': 0,
                'test_mean_margin': 0.0, 'test_min_margin': 0.0,
                'train_constraint_satisfaction': 0.0, 'train_n_total': 0,
                'train_mean_margin': 0.0}

    v = np.asarray(v).reshape(-1)
    v_norm = np.linalg.norm(v)
    v_unit = v / v_norm if v_norm > 0 else v

    def _compute_metrics(Z_p, Z_m):
        Z_p = np.asarray(Z_p)
        Z_m = np.asarray(Z_m)
        p = Z_p @ v_unit
        m = Z_m @ v_unit
        d = p - m
        n = len(d)
        n_sat = int(np.sum(d >= 0))
        return {
            'constraint_satisfaction': float(n_sat / n) if n > 0 else 0.0,
            'n_satisfied': n_sat,
            'n_total': n,
            'mean_margin': float(np.mean(d)) if n > 0 else 0.0,
            'min_margin': float(np.min(d)) if n > 0 else 0.0,
            'median_margin': float(np.median(d)) if n > 0 else 0.0,
            'mean_plus_proj': float(np.mean(p)) if n > 0 else 0.0,
            'mean_minus_proj': float(np.mean(m)) if n > 0 else 0.0,
        }, p, m, d

    train_metrics, p_train, m_train, d_train = _compute_metrics(Z_plus_train, Z_minus_train)

    if has_test:
        test_metrics, p_test, m_test, d_test = _compute_metrics(Z_plus_test, Z_minus_test)
    else:
        # Backwards-compatible: no test split, use train as primary
        test_metrics, p_test, m_test, d_test = train_metrics, p_train, m_train, d_train

    result = {}
    for k, val in test_metrics.items():
        result[f'test_{k}'] = val
    for k, val in train_metrics.items():
        result[f'train_{k}'] = val

    # Save histogram of TEST projections (the meaningful generalization measure)
    if out_dir is not None:
        from pathlib import Path
        out_dir = Path(out_dir)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        split_label = "test" if has_test else "all"

        # Overlaid projections
        axes[0].hist(p_test, bins=30, alpha=0.6, label="Z+ (main)")
        axes[0].hist(m_test, bins=30, alpha=0.6, label="Z- (alt)")
        axes[0].set_xlabel("projection onto v")
        axes[0].set_ylabel("count")
        axes[0].set_title(f"Projections ({split_label}): main vs alternative")
        axes[0].legend()

        # Per-pair margin
        test_cs = test_metrics['constraint_satisfaction']
        axes[1].hist(d_test, bins=30, alpha=0.8)
        axes[1].axvline(0, linestyle="--", color="red", linewidth=1, label="margin=0")
        axes[1].set_xlabel("margin (plus - minus)")
        axes[1].set_ylabel("count")
        axes[1].set_title(f"Pair margins ({split_label}, {test_cs:.0%} satisfied)")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(out_dir / "pair_separation.png", dpi=150)
        plt.close()
        result['saved'] = "pair_separation.png"

    return result


@EVALUATORS.register("spine_nontriviality")
def spine_nontriviality(ctx: dict, threshold: float = 1e-8, **kwargs) -> Dict[str, Any]:
    """
    Check that the concept vector is not trivially zero.

    Reports l2 norm, max component magnitude, non-zero count, effective rank,
    and a boolean is_trivial flag.
    """
    v = ctx['v']
    if v is None:
        return {
            'is_trivial': True,
            'l2_norm': 0.0,
            'max_abs': 0.0,
            'n_nonzero': 0,
            'dim': 0,
            'effective_rank': 0,
        }
    v = np.asarray(v).reshape(-1)
    l2 = float(np.linalg.norm(v))
    max_abs = float(np.max(np.abs(v)))
    n_nonzero = int(np.sum(np.abs(v) > threshold))
    is_trivial = l2 < threshold
    # effective rank: dimensions contributing >1% of total norm
    if l2 > 0:
        fracs = np.abs(v) / l2
        effective_rank = int(np.sum(fracs > 0.01))
    else:
        effective_rank = 0
    return {
        'is_trivial': bool(is_trivial),
        'l2_norm': l2,
        'max_abs': max_abs,
        'n_nonzero': n_nonzero,
        'dim': len(v),
        'effective_rank': effective_rank,
    }


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

# need to actually make this a proper ctx accept etc. 
@EVALUATORS.register("diagnose_concept_vector")
def diagnose_concept_vector(v, name="concept"):
    """Visual sanity checks for a concept vector."""
    import matplotlib.pyplot as plt
    
    v = np.asarray(v).reshape(-1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Distribution of components (should be sparse - spike at 0)
    axes[0].hist(v, bins=100, log=True)
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel("Component value")
    axes[0].set_ylabel("Count (log)")
    axes[0].set_title(f"Component distribution\n(should spike at 0 if sparse)")
    
    # 2. Sorted absolute values (should drop quickly for sparse)
    sorted_abs = np.sort(np.abs(v))[::-1]
    axes[1].plot(sorted_abs)
    axes[1].set_xlabel("Component rank")
    axes[1].set_ylabel("|v_i|")
    axes[1].set_title("Sorted component magnitudes\n(should drop fast if sparse)")
    axes[1].set_yscale('log')
    
    # 3. Cumulative norm contribution
    cumsum = np.cumsum(sorted_abs**2) / np.sum(sorted_abs**2)
    axes[2].plot(cumsum)
    axes[2].axhline(0.9, color='red', linestyle='--', label='90% of norm')
    axes[2].set_xlabel("# of components")
    axes[2].set_ylabel("Cumulative norm fraction")
    axes[2].set_title(f"How many dims to capture 90% of norm?\n({np.searchsorted(cumsum, 0.9)} dims)")
    axes[2].legend()
    
    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(f"{name}_diagnosis.png", dpi=150)
    plt.close()
    
    # Print summary
    print(f"\n{name}:")
    print(f"  Dims for 50% norm: {np.searchsorted(cumsum, 0.5)}")
    print(f"  Dims for 90% norm: {np.searchsorted(cumsum, 0.9)}")
    print(f"  Dims for 99% norm: {np.searchsorted(cumsum, 0.99)}")