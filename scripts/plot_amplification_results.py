#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    run_dir = Path("runs/amplification_pos_2812_final/20260303_101459")
    results_path = run_dir / "results.json"

    with open(results_path, "r") as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for concept_name, concept_data in results.items():
        alpha_vals = sorted([float(a) for a in concept_data["results"].keys()])
        move_change = [concept_data["results"][str(a)]["move_change_rate"] for a in alpha_vals]
        value_shift = [concept_data["results"][str(a)]["value_shift_mean"] for a in alpha_vals]

        eval_data = concept_data.get("eval", {})
        baseline_acc = eval_data.get("baseline_accuracy", 0)
        alpha_results = eval_data.get("alpha_results", {})
        eval_acc = [alpha_results.get(str(a), {}).get("accuracy", np.nan) for a in alpha_vals]

        label = concept_name.replace("pos_2812_", "")

        axes[0].plot(alpha_vals, move_change, marker="o", label=label)
        axes[1].plot(alpha_vals, value_shift, marker="o", label=label)
        axes[2].plot(alpha_vals, eval_acc, marker="o", label=label)
        axes[2].axhline(y=baseline_acc, linestyle="--", alpha=0.4)

    axes[0].set_title("Move Change Rate vs Alpha")
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("Move Change Rate")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Value Shift Mean vs Alpha")
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("Value Shift Mean")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title("STS Accuracy vs Alpha")
    axes[2].set_xlabel("Alpha")
    axes[2].set_ylabel("Accuracy")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    out_path = run_dir / "pos_2812_amplification_plots.png"
    plt.savefig(out_path, dpi=160)

    summary_path = run_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("pos_2812 amplification experiment summary\n")
        f.write("=====================================\n\n")
        for concept_name, concept_data in results.items():
            f.write(f"Concept: {concept_name}\n")
            baseline = concept_data.get("eval", {}).get("baseline_accuracy", "N/A")
            f.write(f"  Baseline STS accuracy: {baseline}\n")

            alpha_items = concept_data["results"]
            best_alpha = max(alpha_items.items(), key=lambda kv: kv[1]["move_change_rate"])
            f.write(f"  Max move change rate: alpha={best_alpha[0]}, rate={best_alpha[1]['move_change_rate']:.3f}\n")

            best_value = max(alpha_items.items(), key=lambda kv: kv[1]["value_shift_mean"])
            f.write(f"  Max value shift mean: alpha={best_value[0]}, shift={best_value[1]['value_shift_mean']:.4f}\n")

            f.write("\n")

    print(f"Saved plot to: {out_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
