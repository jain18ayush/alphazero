from src.registries import EVALUATORS
import src.evaluator.evals  # ensure registration

def run_evals(ctx: dict, eval_cfg: dict) -> dict:
    """
    eval_cfg example:
      evaluation:
        runs:
          - { name: "accuracy_at_0" }
          - { name: "hist", bins: 60, save_as: "hist.png" }
    """
    results = {}
    for spec in eval_cfg.get("runs", []):
        name = spec["name"]
        fn = EVALUATORS.get(name)
        kwargs = {k: v for k, v in spec.items() if k != "name"}
        results[name] = fn(ctx, **kwargs) or {}
    return results
