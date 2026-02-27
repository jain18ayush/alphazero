import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml

from src.registries import DATASETS, MODELS
import src.datasets.datasets
import src.models.models

from src.amplification.amplify import run_amplification_experiment


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_run_dir(experiment_name: str, root: str = "runs") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / experiment_name / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_concept_vectors(concepts_cfg: list[dict]) -> list[dict]:
    vectors = []
    for item in concepts_cfg:
        vec = np.load(item["path"])
        vectors.append({
            "name": item.get("name", Path(item["path"]).stem),
            "path": item["path"],
            "layer": item["layer"],
            "vector": vec,
        })
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_dir = make_run_dir(cfg["experiment_name"])

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Run dir: {run_dir}")

    print("Loading model...")
    net = MODELS.get(cfg["model"]["source"])(cfg["model"])

    print("Loading positions...")
    ds_cfg = cfg["positions"].copy()
    ds_cfg["net"] = net
    positions = DATASETS.get(ds_cfg["source"])(ds_cfg)
    print("Loading evaluation dataset...")
    eval_dataset = None
    if "evaluation_dataset" in cfg:
        eval_cfg = cfg["evaluation_dataset"].copy()
        eval_dataset = load_amplification_eval_dataset(
            eval_cfg["path"],
            max_positions=eval_cfg.get("max_positions"),
        )

    if len(positions) == 0:
        raise ValueError("No positions loaded")

    max_positions = cfg.get("max_positions")
    if max_positions is not None:
        positions = positions[: int(max_positions)]

    print(f"Positions: {len(positions)}")

    print("Loading concept vectors...")
    concepts = load_concept_vectors(cfg["concept_vectors"])

    alpha_range = cfg["alpha_range"]
    n_sims = int(cfg.get("n_sims", 100))
    beta = float(cfg.get("beta", 0.01))

    all_results = {}

    for concept in concepts:
        print(f"Running concept: {concept['name']} (layer {concept['layer']})")
        results = run_amplification_experiment(
            net=net,
            concept_vector=concept["vector"],
            layer=concept["layer"],
            positions=positions,
            alpha_range=alpha_range,
            n_sims=n_sims,
            beta=beta,
        )
        if eval_dataset is not None:
            eval_summary = evaluate_concept_on_dataset(
                net,
                concept_vector=concept["vector"],
                layer=concept["layer"],
                eval_dataset=eval_dataset,
                alpha_range=alpha_range,
                n_sims=n_sims,
                beta=beta,
            )
        else:
            eval_summary = None

        concept_out = {
            "name": concept["name"],
            "layer": concept["layer"],
            "path": concept["path"],
            "results": {
                str(alpha): {
                    "move_change_rate": r.move_change_rate,
                    "value_shift_mean": r.value_shift_mean,
                    "value_shift_std": r.value_shift_std,
                    "n_positions": r.n_positions,
                }
                for alpha, r in results.items()
            },
            "eval": eval_summary,
        }

        all_results[concept["name"]] = concept_out
        save_json(concept_out, run_dir / f"{concept['name']}_results.json")

    save_json(all_results, run_dir / "results.json")
    print("Done.")


if __name__ == "__main__":
    main()
