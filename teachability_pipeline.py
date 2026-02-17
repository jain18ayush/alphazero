import json
import argparse
import time
import numpy as np
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

from src.registries import DATASETS, MODELS
from src.teachability.teachability import (
    _measure_top1_agreement_with_cached_teacher,
    dynamic_prototypes_for_concept,
    filter_prototypes_from_cache,
    is_teachable,
    measure_top1_agreement,
    mcts_policy_for_positions,
    precompute_rollout_data,
    run_teachability_benchmark,
    select_student_checkpoint,
)

import src.datasets.datasets
import src.models.models


class PhaseTimer:
    def __init__(self):
        self.totals: dict[str, float] = defaultdict(float)

    @contextmanager
    def __call__(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.totals[name] += time.monotonic() - t0

    def summary(self) -> dict[str, float]:
        return dict(self.totals)


def load_yaml(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def make_run_dir(name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_model(cfg: dict):
    model = MODELS.get(cfg["source"])(cfg)
    model.eval()
    return model


def _subsample_positions(positions, max_positions: int | None, rng: np.random.RandomState):
    if max_positions is None or len(positions) <= max_positions:
        return list(positions), list(range(len(positions)))
    idx = rng.permutation(len(positions))[:max_positions]
    return [positions[i] for i in idx], idx.tolist()


def _split_positions(positions, test_split: float, rng: np.random.RandomState):
    n = len(positions)
    n_test = max(1, int(n * test_split))
    idx = rng.permutation(n)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    X_train = [positions[i] for i in train_idx]
    X_test = [positions[i] for i in test_idx]
    return X_train, X_test


def _sample_random_positions(
    prototypes,
    random_positions,
    n_train: int,
    n_test: int,
    rng: np.random.RandomState,
):
    needed = n_train + n_test
    if len(random_positions) == 0:
        return [], [], True

    # Build a set of unique identifiers for prototype positions for fast checking.
    # If positions are dicts with 'grid' and 'player', use a tuple of those as key.
    def pos_key(pos):
        # This assumes 'grid' is a numpy array, so we ravel/tostring for hashability
        # and 'player' is int or str.
        grid = pos['grid']
        player = pos['player']
        # Use bytes of the grid and the player as a tuple
        return (grid.tobytes(), player)

    prototype_keys = set(pos_key(p) for p in prototypes)
    filtered_random_positions = [pos for pos in random_positions if pos_key(pos) not in prototype_keys]

    if len(filtered_random_positions) == 0:
        return [], [], True

    if len(filtered_random_positions) >= needed:
        idx = rng.permutation(len(filtered_random_positions))[:needed]
        sampled = [filtered_random_positions[i] for i in idx]
        used_replacement = False
    else:
        idx = rng.choice(len(filtered_random_positions), needed, replace=True)
        sampled = [filtered_random_positions[i] for i in idx]
        used_replacement = True

    return sampled[:n_train], sampled[n_train:], used_replacement


def run_teachability(cfg: dict, run_dir: Path):
    seed = cfg.get("seed", 42)
    rng = np.random.RandomState(seed)

    # 1) Load teacher model.
    print("\n=== Loading teacher model ===")
    teacher_cfg = cfg["teacher_model"]
    teacher = _load_model(teacher_cfg)
    print(f"Teacher model source: {teacher_cfg['source']}")

    # 2) Load positions and random controls. 
    print("\n=== Loading position datasets ===")
    positions_cfg = dict(cfg["positions"])
    positions_cfg["net"] = teacher
    positions = DATASETS.get(positions_cfg["source"])(positions_cfg)
    print(f"Prototype pool positions: {len(positions)}")

    # 3) Load concepts via DATASETS registry.
    print("\n=== Loading concepts ===")
    concept_cfg = cfg["concepts"]
    concept_source = concept_cfg.get("source", "concept_vectors_from_runs")
    concepts = DATASETS.get(concept_source)(concept_cfg)
    print(f"Loaded {len(concepts)} concepts")

    # 4) Settings.
    dyn_cfg = cfg["dynamic_prototypes"]
    teach_cfg = cfg["teachability"]
    select_cfg = cfg.get("student_selection", {})

    n_sim = teach_cfg.get("n_sim", 200)
    temp = teach_cfg.get("temp", 1.0)
    epochs = teach_cfg.get("epochs", 5)
    lr = teach_cfg.get("lr", 1e-4)
    batch_size = teach_cfg.get("batch_size", 64)
    test_split = teach_cfg.get("test_split", 0.2)
    teachability_margin = teach_cfg.get("teachability_margin", 0.05)
    min_prototypes = teach_cfg.get("min_prototypes", 20)

    max_depth = dyn_cfg.get("max_depth", 10)
    sort_key = dyn_cfg.get("sort_key", "N")
    min_margin = dyn_cfg.get("min_margin", 0.0)
    min_value_gap = dyn_cfg.get("min_value_gap", 0.20)
    min_visit_gap_ratio = dyn_cfg.get("min_visit_gap_ratio", 0.10)
    t_offset = dyn_cfg.get("t_offset", 5)
    top_percent = dyn_cfg.get("top_percent")
    max_prototypes = dyn_cfg.get("max_prototypes")
    max_positions = dyn_cfg.get("max_positions")

    overlap_threshold = select_cfg.get("overlap_threshold", 0.2)
    student_base_cfg = dict(cfg["student_model"])
    checkpoint_paths = select_cfg.get("checkpoint_paths") or []

    all_results = []
    run_timer = PhaseTimer()

    # Optionally subsample positions before the expensive precompute step.
    if max_positions is not None and len(positions) > max_positions:
        sub_idx = rng.permutation(len(positions))[:max_positions]
        positions = [positions[i] for i in sub_idx]
        print(f"Subsampled to {len(positions)} positions (max_positions={max_positions})")

    # Pre-compute MCTS contrasts + activations for all positions (concept-independent).
    unique_layers = list({c["layer"] for c in concepts})
    print(f"\n=== Pre-computing rollout data for {len(positions)} positions, layers={unique_layers} ===")

    with run_timer("precompute_rollout_data"):
        rollout_cache, precompute_timing = precompute_rollout_data(
            net=teacher,
            positions=positions,
            layers=unique_layers,
            n_sim=n_sim,
            max_depth=max_depth,
            sort_key=sort_key,
            min_value_gap=min_value_gap,
            min_visit_gap_ratio=min_visit_gap_ratio,
            t_offset=t_offset,
        )

    save_json(precompute_timing, run_dir / "precompute_timing.json")

    print(f"\n=== Processing {len(concepts)} concepts ===")
    for i, concept in enumerate(concepts):
        layer = concept["layer"]
        vector = concept["vector"]
        concept_path = concept["path"]
        print(f"\n--- Concept {i:04d} | layer={layer} ---")

        concept_dir = run_dir / f"concept_{i:04d}"
        concept_dir.mkdir(parents=True, exist_ok=True)
        np.save(concept_dir / "concept_vector.npy", vector)

        concept_timer = PhaseTimer()

        with concept_timer("prototype_filtering"), run_timer("prototype_filtering"):
            prototypes, proto_meta, proto_stats = filter_prototypes_from_cache(
                cache=rollout_cache,
                layer=layer,
                v=vector,
                min_margin=min_margin,
                top_percent=top_percent,
                max_prototypes=max_prototypes,
            )

        save_json(proto_stats, concept_dir / "prototype_stats.json")
        save_json({"examples": proto_meta}, concept_dir / "prototype_meta.json")
        save_json({"examples": proto_meta[:50]}, concept_dir / "prototype_meta_sample.json")

        if len(prototypes) < min_prototypes:
            result = {
                "layer": layer,
                "concept_path": concept_path,
                "n_prototypes": len(prototypes),
                "status": "skipped_few_prototypes",
                "min_prototypes": int(min_prototypes),
                "is_teachable": False,
            }
            save_json(result, concept_dir / "result.json")
            save_json(concept_timer.summary(), concept_dir / "timing.json")
            all_results.append(result)
            print(f"  Skipping (only {len(prototypes)} prototypes)")
            continue

        # Pre-compute teacher top-1 on prototypes once for student selection.
        with concept_timer("teacher_mcts_prototypes"), run_timer("teacher_mcts_prototypes"):
            _, teacher_top1_prototypes = mcts_policy_for_positions(
                teacher, prototypes, n_sim=n_sim, temp=temp,
            )

        # Step B: Student selection.
        student_selection_info = {}
        selected_student_cfg = dict(student_base_cfg)

        with concept_timer("student_selection"), run_timer("student_selection"):
            if checkpoint_paths:
                def make_student_cfg(path: str):
                    cfg_i = dict(student_base_cfg)
                    cfg_i["checkpoint_path"] = path
                    return cfg_i

                student_selection_info = select_student_checkpoint(
                    checkpoint_paths=checkpoint_paths,
                    make_model_cfg=make_student_cfg,
                    load_model=_load_model,
                    teacher=teacher,
                    prototypes=prototypes,
                    overlap_threshold=overlap_threshold,
                    n_sim=n_sim,
                    temp=temp,
                    teacher_top1=teacher_top1_prototypes,
                )
                selected_student_cfg = make_student_cfg(student_selection_info["selected_path"])
                student_selection_info["mode"] = "sweep"
            else:
                student = _load_model(selected_student_cfg)
                _, overlap = _measure_top1_agreement_with_cached_teacher(
                    teacher_top1_prototypes, student, prototypes, n_sim=n_sim, temp=temp,
                )
                student_selection_info = {
                    "mode": "single",
                    "selected_path": selected_student_cfg.get("checkpoint_path"),
                    "selected_overlap": float(overlap),
                    "overlap_threshold": float(overlap_threshold),
                    "checked": [],
                }

        save_json(student_selection_info, concept_dir / "student_selection.json")

        # Step C/D/E/F: benchmark.
        X_train_concept, X_test_concept = _split_positions(prototypes, test_split, rng)
        X_train_random, X_test_random, used_replacement = _sample_random_positions(
            prototypes,
            positions,
            len(X_train_concept),
            len(X_test_concept),
            rng,
        )

        def load_selected_student():
            return _load_model(dict(selected_student_cfg))

        with concept_timer("benchmark"), run_timer("benchmark"):
            benchmark = run_teachability_benchmark(
                load_student=load_selected_student,
                teacher=teacher,
                X_train_concept=X_train_concept,
                X_test_concept=X_test_concept,
                X_train_random=X_train_random,
                X_test_random=X_test_random,
                n_sim=n_sim,
                temp=temp,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )

        teachable, gain = is_teachable(benchmark, margin=teachability_margin)

        result = {
            "layer": layer,
            "concept_path": concept_path,
            "n_prototypes": int(len(prototypes)),
            "train_size": int(len(X_train_concept)),
            "test_size": int(len(X_test_concept)),
            "used_random_with_replacement": bool(used_replacement),
            "benchmark": benchmark,
            "concept_specific_gain": float(gain),
            "teachability_margin": float(teachability_margin),
            "is_teachable": bool(teachable),
            "status": "teachable" if teachable else "not_teachable",
        }

        save_json(benchmark, concept_dir / "benchmark.json")
        save_json(result, concept_dir / "result.json")
        save_json(concept_timer.summary(), concept_dir / "timing.json")
        all_results.append(result)

        timing_str = " ".join(f"{k}={v:.1f}s" for k, v in concept_timer.summary().items())
        print(
            f"  baseline={benchmark['baseline_eval_C']:.3f} "
            f"C->C={benchmark['train_C_eval_C']:.3f} "
            f"R->C={benchmark['train_R_eval_C']:.3f} "
            f"gain={gain:.3f} teachable={teachable}"
        )
        print(f"  timing: {timing_str}")

    summary = {
        "n_concepts": len(all_results),
        "n_teachable": sum(1 for r in all_results if r.get("is_teachable")),
        "results": all_results,
    }

    save_json(summary, run_dir / "results.json")
    save_json(run_timer.summary(), run_dir / "timing.json")
    print(f"\n=== Results saved to {run_dir} ===")
    timing_str = " ".join(f"{k}={v:.1f}s" for k, v in run_timer.summary().items())
    print(f"=== Timing: {timing_str} ===")
    return summary


def main():
    import yaml

    parser = argparse.ArgumentParser(description="Teachability pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_dir = make_run_dir(cfg["experiment_name"])

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Run dir: {run_dir}")

    run_teachability(cfg, run_dir)


if __name__ == "__main__":
    main()
