"""
Generate quiz data from prototype positions.

Loads prototype metadata, runs MCTS on each position, builds spine trees,
and serializes everything to JSON for the Streamlit quiz app.

Usage:
    python generate_quiz_data.py --config configs/quiz.yaml
"""

import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.registries import DATASETS, MODELS
from alphazero.games.othello import OthelloBoard
from alphazero.players import AlphaZeroPlayer

# Import to register
import src.datasets.datasets
import src.models.models


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def make_run_dir(name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_spine_tree_no_activations(mct, board, max_depth, sort_key="N"):
    """
    Build a spine decision tree from an MCTS tree, without extracting activations.

    Same heap-indexed structure as src/spine/spine.py:build_spine_tree but skips
    the get_single_activation call.

    Returns levels[depth][idx] = node_dict or None.
    """
    levels = [[] for _ in range(max_depth + 1)]
    gid_counter = 0

    def walk(node, board_state, depth, label, move, idx, is_main_path):
        nonlocal gid_counter

        if depth > max_depth:
            return

        while len(levels[depth]) <= idx:
            levels[depth].append(None)

        levels[depth][idx] = {
            "grid": board_state.grid.copy(),
            "player": int(board_state.player),
            "move": move,
            "label": label,
            "n_children": len(node.children),
            "N": getattr(node, "N", None),
            "Q": getattr(node, "Q", None),
            "depth": depth,
            "main": is_main_path,
            "id": gid_counter,
        }
        gid_counter += 1

        if not node.children:
            return

        items = sorted(
            node.children.items(),
            key=lambda x: getattr(x[1], sort_key),
            reverse=True,
        )

        # Best child - always follow
        best_move, best_node = items[0]
        b_best = board_state.clone()
        b_best.play_move(best_move)
        walk(best_node, b_best, depth + 1, "best", best_move, 2 * idx, is_main_path)

        # Second-best child - only if on main path
        if len(items) >= 2 and is_main_path:
            second_move, second_node = items[1]
            b_second = board_state.clone()
            b_second.play_move(second_move)
            walk(
                second_node, b_second, depth + 1, "2nd", second_move, 2 * idx + 1, False
            )

    walk(mct.root, board.clone(), 0, "root", None, 0, True)
    return levels


def serialize_spine_levels(levels):
    """Convert spine levels to JSON-safe format (numpy -> Python types)."""
    out = []
    for level in levels:
        serialized_level = []
        for node in level:
            if node is None:
                serialized_level.append(None)
                continue
            s = {}
            for k, v in node.items():
                if isinstance(v, np.ndarray):
                    s[k] = v.tolist()
                elif isinstance(v, (np.integer,)):
                    s[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    s[k] = float(v)
                elif isinstance(v, tuple):
                    s[k] = list(v)
                else:
                    s[k] = v
            serialized_level.append(s)
        out.append(serialized_level)
    return out


def generate_quiz_item(pos, meta, net, n_sims, spine_depth, board_size, sort_key):
    """
    Generate a single quiz item for a prototype position.

    Returns a dict with position info, AZ's best move, move stats, and spine tree,
    or None if the position is invalid (game over, only pass move).
    """
    board = OthelloBoard(n=board_size)
    board.grid = pos["grid"].copy()
    board.player = pos["player"]

    legal_moves = board.get_moves(board.player)
    if not legal_moves or legal_moves == [board.pass_move]:
        return None

    # Run MCTS
    player = AlphaZeroPlayer(nn=net, n_sim=n_sims)
    player.reset()
    best_move, _, _, _ = player.get_move(board, temp=0)

    # Extract child stats
    move_stats = []
    for move, child in player.mct.root.children.items():
        move_stats.append(
            {
                "move": list(move) if isinstance(move, tuple) else move,
                "N": int(child.N),
                "Q": float(child.Q),
            }
        )
    move_stats.sort(key=lambda x: x["N"], reverse=True)

    # Build spine tree (no activations)
    board_snapshot = board.clone()
    board_snapshot.grid = pos["grid"].copy()
    board_snapshot.player = pos["player"]
    levels = build_spine_tree_no_activations(
        player.mct, board_snapshot, spine_depth, sort_key
    )

    return {
        "source_idx": int(meta["source_idx"]),
        "prototype_meta": {
            k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
            for k, v in meta.items()
            if k != "source_idx"
        },
        "position": {
            "grid": pos["grid"].tolist()
            if isinstance(pos["grid"], np.ndarray)
            else pos["grid"],
            "player": int(pos["player"]),
            "move_number": int(pos.get("move_number", -1)),
        },
        "legal_moves": [
            list(m) if isinstance(m, tuple) else m
            for m in legal_moves
            if m != board.pass_move
        ],
        "az_move": list(best_move) if isinstance(best_move, tuple) else best_move,
        "move_stats": move_stats,
        "spine_tree": {
            "max_depth": spine_depth,
            "levels": serialize_spine_levels(levels),
        },
    }


def run_quiz_generation(cfg, run_dir):
    """Main orchestrator: load data, run MCTS on each prototype, write JSON."""
    # 1. Load positions
    print("\n=== Loading positions ===")
    pos_cfg = cfg["positions"]
    positions = DATASETS.get(pos_cfg["source"])(pos_cfg)
    print(f"Loaded {len(positions)} positions")

    # 2. Load model
    print("\n=== Loading model ===")
    model_cfg = cfg["model"]
    net = MODELS.get(model_cfg["source"])(model_cfg)
    net.eval()
    print(f"Model: {model_cfg['name']}")

    # 3. Load prototype metadata
    print("\n=== Loading prototype metadata ===")
    meta_path = cfg["prototypes"]["meta_path"]
    with open(meta_path) as f:
        proto_data = json.load(f)
    prototypes = proto_data["examples"]
    print(f"Loaded {len(prototypes)} prototypes")

    # 4. Generate quiz items
    spine_cfg = cfg["spine"]
    n_sims = spine_cfg["n_sims"]
    spine_depth = spine_cfg["max_depth"]
    sort_key = spine_cfg.get("sort_key", "N")
    board_size = pos_cfg["board_size"]

    items = []
    for meta in tqdm(prototypes, desc="Generating quiz items"):
        src_idx = meta["source_idx"]
        if src_idx >= len(positions):
            print(f"  Skipping source_idx={src_idx} (out of range)")
            continue

        pos = positions[src_idx]
        item = generate_quiz_item(
            pos, meta, net, n_sims, spine_depth, board_size, sort_key
        )
        if item is not None:
            items.append(item)
            print(f"  [{len(items)}] source_idx={src_idx}, az_move={item['az_move']}")
        else:
            print(f"  Skipped source_idx={src_idx} (no legal moves)")

    # 5. Build output
    output = {
        "config": {
            "n_sims": n_sims,
            "spine_depth": spine_depth,
            "model_name": model_cfg["name"],
            "sort_key": sort_key,
            "board_size": board_size,
        },
        "items": items,
    }

    # Write main output
    out_path = Path(cfg["output"]["path"])
    save_json(output, out_path)
    print(f"\nWrote {len(items)} quiz items to {out_path}")

    # Also save copy + config in run dir
    save_json(output, run_dir / "quiz_data.json")
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"Run artifacts saved to {run_dir}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate quiz data from prototypes")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument(
        "--n_sims", type=int, default=None, help="Override n_sims from config"
    )
    parser.add_argument(
        "--spine_depth", type=int, default=None, help="Override spine max_depth"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Override output path"
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # Apply CLI overrides
    if args.n_sims is not None:
        cfg["spine"]["n_sims"] = args.n_sims
    if args.spine_depth is not None:
        cfg["spine"]["max_depth"] = args.spine_depth
    if args.output is not None:
        cfg["output"]["path"] = args.output

    run_dir = make_run_dir(cfg["experiment_name"])

    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Run dir: {run_dir}")

    run_quiz_generation(cfg, run_dir)


if __name__ == "__main__":
    main()
