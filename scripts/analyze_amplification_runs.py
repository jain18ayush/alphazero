#!/usr/bin/env python3
"""Analyze amplification runs and plot trends across runs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


@dataclass
class AlphaStats:
    alpha: float
    move_change_mean: float
    move_change_std: float
    value_shift_mean: float
    value_shift_std: float
    n_concepts: int


def _sorted_alpha_keys(result_dict: Dict[str, dict]) -> List[Tuple[float, str]]:
    alphas: List[Tuple[float, str]] = []
    for key in result_dict.keys():
        try:
            alphas.append((float(key), key))
        except ValueError:
            continue
    return sorted(alphas, key=lambda x: x[0])


def _summarize_run(results: Dict[str, dict]) -> List[AlphaStats]:
    sample_concept = next(iter(results.values()))
    alpha_keys = _sorted_alpha_keys(sample_concept["results"])
    stats: List[AlphaStats] = []

    for alpha_float, alpha_key in alpha_keys:
        move_rates: List[float] = []
        value_means: List[float] = []
        for concept in results.values():
            metrics = concept["results"][alpha_key]
            move_rates.append(metrics["move_change_rate"])
            value_means.append(metrics["value_shift_mean"])

        move_mean = sum(move_rates) / len(move_rates)
        value_mean = sum(value_means) / len(value_means)
        move_std = (sum((x - move_mean) ** 2 for x in move_rates) / len(move_rates)) ** 0.5
        value_std = (sum((x - value_mean) ** 2 for x in value_means) / len(value_means)) ** 0.5

        stats.append(
            AlphaStats(
                alpha=alpha_float,
                move_change_mean=move_mean,
                move_change_std=move_std,
                value_shift_mean=value_mean,
                value_shift_std=value_std,
                n_concepts=len(move_rates),
            )
        )

    return stats


def _plot_per_run(run_name: str, results: Dict[str, dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Move-change rate plot
    plt.figure(figsize=(7, 4.5))
    for concept_name, concept in results.items():
        alpha_keys = _sorted_alpha_keys(concept["results"])
        xs = [a for a, _ in alpha_keys]
        ys = [concept["results"][key]["move_change_rate"] for _, key in alpha_keys]
        plt.plot(xs, ys, marker="o", alpha=0.6, label=concept_name)

    stats = _summarize_run(results)
    xs = [s.alpha for s in stats]
    ys = [s.move_change_mean for s in stats]
    plt.plot(xs, ys, marker="o", color="black", linewidth=2, label="avg")

    plt.title(f"Move-change rate vs alpha ({run_name})")
    plt.xlabel("alpha")
    plt.ylabel("move_change_rate")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{run_name}_move_change.png"), dpi=200)
    plt.close()

    # Value-shift mean plot
    plt.figure(figsize=(7, 4.5))
    for concept_name, concept in results.items():
        alpha_keys = _sorted_alpha_keys(concept["results"])
        xs = [a for a, _ in alpha_keys]
        ys = [concept["results"][key]["value_shift_mean"] for _, key in alpha_keys]
        plt.plot(xs, ys, marker="o", alpha=0.6, label=concept_name)

    xs = [s.alpha for s in stats]
    ys = [s.value_shift_mean for s in stats]
    plt.plot(xs, ys, marker="o", color="black", linewidth=2, label="avg")

    plt.title(f"Value-shift mean vs alpha ({run_name})")
    plt.xlabel("alpha")
    plt.ylabel("value_shift_mean")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{run_name}_value_shift.png"), dpi=200)
    plt.close()


def _plot_across_runs(run_summaries: Dict[str, List[AlphaStats]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    for run_name, stats in run_summaries.items():
        xs = [s.alpha for s in stats]
        ys = [s.move_change_mean for s in stats]
        plt.plot(xs, ys, marker="o", alpha=0.7, label=run_name)
    plt.title("Avg move-change rate vs alpha (across runs)")
    plt.xlabel("alpha")
    plt.ylabel("move_change_rate")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "across_runs_move_change.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for run_name, stats in run_summaries.items():
        xs = [s.alpha for s in stats]
        ys = [s.value_shift_mean for s in stats]
        plt.plot(xs, ys, marker="o", alpha=0.7, label=run_name)
    plt.title("Avg value-shift mean vs alpha (across runs)")
    plt.xlabel("alpha")
    plt.ylabel("value_shift_mean")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "across_runs_value_shift.png"), dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze amplification run results.")
    parser.add_argument("--runs-dir", default="runs/amplification_bn2_demo")
    args = parser.parse_args()

    run_dirs = [
        d for d in os.listdir(args.runs_dir)
        if os.path.isdir(os.path.join(args.runs_dir, d))
    ]

    summaries: Dict[str, List[AlphaStats]] = {}
    summary_out: Dict[str, dict] = {}

    for run_name in sorted(run_dirs):
        results_path = os.path.join(args.runs_dir, run_name, "results.json")
        if not os.path.exists(results_path):
            continue

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        stats = _summarize_run(results)
        summaries[run_name] = stats
        summary_out[run_name] = {
            "alphas": [s.alpha for s in stats],
            "move_change_mean": [s.move_change_mean for s in stats],
            "move_change_std": [s.move_change_std for s in stats],
            "value_shift_mean": [s.value_shift_mean for s in stats],
            "value_shift_std": [s.value_shift_std for s in stats],
            "n_concepts": [s.n_concepts for s in stats],
        }

        _plot_per_run(run_name, results, os.path.join(args.runs_dir, "analysis"))

    if summaries:
        _plot_across_runs(summaries, os.path.join(args.runs_dir, "analysis"))

    if summary_out:
        summary_path = os.path.join(args.runs_dir, "analysis", "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_out, f, indent=2)
        print(f"Wrote summary: {summary_path}")
    else:
        print("No results.json files found in runs directory.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
