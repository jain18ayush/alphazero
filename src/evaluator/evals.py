# src/evaluators/evals.py
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
