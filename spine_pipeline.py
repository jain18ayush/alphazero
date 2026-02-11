import json
import yaml
import torch
import argparse
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.registries import DATASETS, MODELS, OPTIMIZERS, EVALUATORS
from src.evaluator.registry import run_evals
from src.spine.spine import (
    find_disagreement_positions,
    build_spine_tree,
    extract_pairs_from_spine,
)
from alphazero.games.othello import OthelloBoard, OthelloNet
from alphazero.players import AlphaZeroPlayer

# Import to register
import src.datasets.datasets
import src.models.models
import src.optimizers.optimizers
import src.evaluator.evals


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj, path: Path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)


def make_run_dir(name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def visualize_spine_decision_tree(levels, n, board_scale=1.0, path=None):
    """
    Spine + branches visualization that scales to larger max_depth.

    Layout idea:
    - Lane 0: the main (all-left / best-only) spine.
    - Lane k (k>=1): nodes whose FIRST right-turn (second-best) occurs at depth k.
      (Computed from idx's binary path bits for that depth.)
    - This keeps width O(max_depth) rather than O(2^max_depth).

    Assumes heap-style indexing:
      parent (d, idx) -> children (d+1, 2*idx) [best/left], (d+1, 2*idx+1) [second/right]
    """

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from alphazero.games.othello import OthelloBoard

    max_depth = len(levels) - 1

    # ---- Collect nodes ----
    node_positions = {}
    for d, level in enumerate(levels):
        for idx, info in enumerate(level):
            if info is not None:
                node_positions[(d, idx)] = info

    if not node_positions:
        print("No nodes to visualize.")
        return

    # ---- Lane assignment ----
    # lane = 0 if path bits are all 0; else lane = position (1..d) of first '1' in MSB->LSB path bits.
    def lane_for(d, idx):
        if d == 0:
            return 0
        # idx in [0, 2^d). Represent with d bits.
        bits = f"{idx:0{d}b}"  # MSB first
        first_one = bits.find("1")
        return 0 if first_one == -1 else (first_one + 1)  # +1 => lanes 1..d

    lane = {(d, idx): lane_for(d, idx) for (d, idx) in node_positions.keys()}
    max_lane = max(lane.values())  # <= max_depth

    # ---- Layout parameters (tuned to scale) ----
    # Board "card" dimensions in *data coords*
    board_size = 1.0 * board_scale
    meta_w = 1.25 * board_scale          # width of metadata panel
    pad = 0.18 * board_scale             # padding around board/meta inside card
    card_w = board_size + meta_w + 2 * pad
    card_h = board_size + 2 * pad + 0.25 * board_scale   # extra top space for title line

    lane_gap = 0.55 * board_scale        # gap between lanes
    depth_gap = 0.65 * board_scale       # gap between rows
    x_step = card_w + lane_gap
    y_step = card_h + depth_gap

    # ---- Compute node centers (x,y) ----
    # Put depth 0 at top => y increases downward, then invert axis at end (or compute reversed).
    xy = {}
    for (d, idx) in node_positions.keys():
        L = lane[(d, idx)]
        cx = L * x_step
        cy = d * y_step
        xy[(d, idx)] = (cx, cy)

    # ---- Figure sizing ----
    # Convert data coords -> inches roughly: we pick figsize based on number of lanes / depth.
    fig_w = max(8, (max_lane + 1) * (2.2 * board_scale))
    fig_h = max(8, (max_depth + 1) * (1.75 * board_scale))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-x_step * 0.6, (max_lane + 1) * x_step - x_step * 0.4)
    ax.set_ylim(-y_step * 0.6, (max_depth + 1) * y_step - y_step * 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()  # depth 0 at top

    # ---- Drawing helpers ----
    def draw_card(cx, cy, info, is_root=False):
        grid = info["grid"]
        move = info["move"]
        label = info["label"]
        n_ch = info["n_children"]
        N = info.get("N")
        Q = info.get("Q")

        # Colors
        if is_root:
            bg = "#e3f2fd"
            accent = "#1565c0"
        elif label == "best":
            bg = "#e8f5e9"
            accent = "#2e7d32"
        else:
            bg = "#fff3e0"
            accent = "#ef6c00"

        # Card origin
        left = cx - card_w / 2
        top = cy - card_h / 2

        # Card box
        card = patches.FancyBboxPatch(
            (left, top),
            card_w, card_h,
            boxstyle="round,pad=0.03",
            facecolor=bg, edgecolor="#777777", linewidth=0.9
        )
        ax.add_patch(card)

        # Title line (top-left inside card)
        if is_root:
            title = f"ROOT  |  children={n_ch}"
        else:
            title = f"{label.upper():>4}  move={move}  |  children={n_ch}"
        ax.text(
            left + pad, top + 0.18 * board_scale,
            title,
            ha="left", va="center",
            fontsize=8 * board_scale, fontweight="bold", color=accent
        )

        # Board region
        board_left = left + pad
        board_top = top + 0.32 * board_scale + pad
        board_bottom = board_top
        # board rectangle (we draw from top-left using bottom coords)
        board_y = board_top
        board_x = board_left

        # Board background
        ax.add_patch(
            patches.Rectangle(
                (board_x, board_y),
                board_size, board_size,
                facecolor=OthelloBoard.COLORS[0], edgecolor="black", linewidth=0.8
            )
        )

        # Grid lines
        cell = board_size / n
        for i in range(1, n):
            ax.plot([board_x + i * cell, board_x + i * cell], [board_y, board_y + board_size], "k-", lw=0.35)
            ax.plot([board_x, board_x + board_size], [board_y + i * cell, board_y + i * cell], "k-", lw=0.35)

        # Pieces + highlight
        r = cell * 0.38
        for row in range(n):
            for col in range(n):
                v = grid[row, col]
                if v != 0:
                    px = board_x + col * cell + cell / 2
                    py = board_y + (n - 1 - row) * cell + cell / 2
                    ax.add_patch(
                        patches.Circle((px, py), r,
                                       facecolor=OthelloBoard.COLORS[v],
                                       edgecolor="black", linewidth=0.35)
                    )
                if move == (row, col):
                    hx = board_x + col * cell
                    hy = board_y + (n - 1 - row) * cell
                    ax.add_patch(
                        patches.Rectangle((hx, hy), cell, cell,
                                          fill=False, edgecolor="red", linewidth=1.8)
                    )

        # Metadata panel (right side)
        meta_left = board_x + board_size + pad
        meta_top = board_y + 0.02 * board_scale

        # Nicely formatted stats (monospace so it lines up)
        lines = []
        if N is not None:
            lines.append(f"N: {N:.0f}")
        if Q is not None:
            lines.append(f"Q: {Q:+.3f}")
        # Add more fields if present
        if "P" in info:
            lines.append(f"P: {info['P']:.3f}")
        if "U" in info:
            lines.append(f"U: {info['U']:.3f}")

        meta_text = "\n".join(lines) if lines else "(no stats)"
        ax.text(
            meta_left, meta_top,
            meta_text,
            ha="left", va="top",
            fontsize=8 * board_scale,
            family="monospace",
            color="#222222"
        )

        idx = info.get("id")
        if idx is not None:
            ax.text(
                left + card_w - pad * 0.6,
                top + card_h - pad * 0.6,
                f"idx={idx}",
                ha="right",
                va="bottom",
                fontsize=7 * board_scale,
                family="monospace",
                color="#444444"
            )


    def draw_arrow(p_xy, c_xy, is_best):
        # Slightly offset arrow to reduce overlap with cards
        (px, py) = p_xy
        (cx, cy) = c_xy

        # Arrow from bottom center of parent card to top center of child card
        start = (px, py + card_h * 0.45)
        end = (cx, cy - card_h * 0.45)

        color = "#2e7d32" if is_best else "#ef6c00"
        lw = 2.0 if is_best else 1.6

        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0)
        )

    # ---- Draw edges first (so cards sit on top) ----
    for (d, idx), info in node_positions.items():
        if d >= max_depth:
            continue
        parent = (d, idx)
        for k, child_idx in enumerate([2 * idx, 2 * idx + 1]):
            child = (d + 1, child_idx)
            if child not in node_positions:
                continue
            # k==0 => best/left, k==1 => second/right
            draw_arrow(xy[parent], xy[child], is_best=(k == 0))

    # ---- Draw nodes ----
    for (d, idx), info in node_positions.items():
        cx, cy = xy[(d, idx)]
        draw_card(cx, cy, info, is_root=(d == 0))


    # ---- Lane labels (helpful when depth gets big) ----
    ax.text(0, -0.35 * y_step, "Lane 0: spine (best-only)", ha="left", va="center",
            fontsize=9 * board_scale, color="#2e7d32", fontweight="bold")
    for L in range(1, max_lane + 1):
        ax.text(L * x_step, -0.35 * y_step, f"Lane {L}: first 2nd-best at depth {L}",
                ha="center", va="center", fontsize=8 * board_scale, color="#ef6c00")

    plt.title("Spine + branches (2nd-best only once per branch, then best-only)", fontsize=12, pad=18)
    plt.tight_layout()
    plt.savefig(path)    


def run_spine(cfg: dict, run_dir: Path):
    seed = cfg['seed']
    np.random.seed(seed)

    # 1. Load positions
    print("\n=== Loading positions ===")
    pos_cfg = cfg['positions']
    positions = DATASETS.get(pos_cfg['source'])(pos_cfg)
    print(f"Loaded {len(positions)} positions")

    # 2. Load strong model
    print("\n=== Loading models ===")
    strong_cfg = cfg['strong_model']
    strong_net = MODELS.get("file")(strong_cfg)
    strong_net.eval()
    print(f"Strong model: {strong_cfg['name']}")

    # 3. Load weak model (from checkpoint)
    weak_cfg = cfg['weak_model']
    model_config_path = Path(strong_cfg['checkpoint_path']) / strong_cfg['name'] / "config.json"
    with open(model_config_path, "r") as f:
        json_config = json.load(f)
    model_config = OthelloNet.CONFIG(**json_config)
    weak_net = OthelloNet(config=model_config)
    weak_net.load_state_dict(torch.load(weak_cfg['checkpoint_path'], weights_only=True))
    weak_net.eval()
    print(f"Weak model: {weak_cfg['checkpoint_path']}")

    # 4. Disagreement filter (two-pass)
    print("\n=== Disagreement filter ===")
    dis_cfg = cfg['disagreement']

    if 'exists' in dis_cfg:
        interesting = np.load(dis_cfg['exists'], allow_pickle=True)
    else: 
        interesting = find_disagreement_positions(
            positions,
            strong_net,
            weak_net,
            board_size=pos_cfg['board_size'],
            n_sims_pass1=dis_cfg['n_sims_pass1'],
            n_sims_pass2=dis_cfg['n_sims_pass2'],
        )
        print(f"Interesting positions: {len(interesting)}")

        if len(interesting) == 0:
            print("No disagreement positions found. Exiting.")
            save_json({"n_positions": 0}, run_dir / "results.json")
            return {}
        else: 
            # INSERT_YOUR_CODE
            save_path = run_dir / "interesting"
            np.save(save_path, np.array(interesting, dtype=object), allow_pickle=True)

    # 5. Process each interesting position
    layer_names = [l['name'] for l in cfg['hooks']['layers']]
    spine_cfg = cfg['spine']
    opt_cfg = cfg['optimization']
    eval_cfg = cfg['evaluation']

    optimizer = OPTIMIZERS.get(opt_cfg['method'])

    # Accumulators for batch concept vectors per layer
    batch_vectors = {layer: [] for layer in layer_names}
    batch_metadata = {layer: [] for layer in layer_names}

    all_results = {}

    print(f"\n=== Processing {len(interesting)} positions, {len(layer_names)} layers ===")

    for pos_idx, pos in enumerate(tqdm(interesting[3400:], desc="Processing positions", initial=3400, total=len(interesting))):
        pos_idx += 3400 # rough way to move it up 
        try: 
            pos_label = f"pos_{pos_idx:04d}"
            print(f"\n--- {pos_label} (move {pos.get('move_number', '?')}) ---")

            pos_dir = run_dir / pos_label
            pos_dir.mkdir(parents=True, exist_ok=True)

            # Save position info
            pos_info = {
                'grid': pos['grid'].tolist(),
                'player': int(pos['player']),
                'move_number': pos.get('move_number'),
            }
            save_json(pos_info, pos_dir / "position.json")

            # Set up board and run MCTS once
            board = OthelloBoard(n=pos_cfg['board_size'])
            board.grid = pos['grid'].copy()
            board.player = pos['player']
            board_snapshot = board.clone()

            player = AlphaZeroPlayer(nn=strong_net, n_sim=spine_cfg['n_sims'])
            player.reset()
            move, _, _, _ = player.get_move(board)
            mct = player.mct
            print(f"  MCTS done. Best move: {move}, root children: {len(mct.root.children)}")

            pos_results = {}

            # Process each layer (reuse the same MCTS tree)
            for layer in layer_names:
                print(f"  Layer: {layer}")

                # Build spine tree --> need to change this so it can take multiple layers at once and return multiple levels for each layer
                levels = build_spine_tree(
                    mct, board_snapshot, strong_net,
                    max_depth=spine_cfg['max_depth'],
                    layer_name=layer,
                    sort_key=spine_cfg.get('sort_key', 'N'),
                )

                if pos_dir / f"spine_tree.png" not in os.listdir(pos_dir):
                # save the spine tree visualization 
                    visualize_spine_decision_tree(levels, board_snapshot.n, path=pos_dir / f"spine_tree.png")
                
                # Extract pairs
                Z_plus, Z_minus, pairs_meta = extract_pairs_from_spine(levels)
                n_pairs = len(pairs_meta)
                print(f"    Extracted {n_pairs} pairs")

                if n_pairs == 0:
                    print(f"    No pairs extracted, skipping")
                    continue

                dim = Z_plus.shape[1]
                print(f"    Activation dim: {dim}")

                # Train/test split of pairs
                test_frac = eval_cfg.get('test_split', 0.3)
                n_test = max(1, int(n_pairs * test_frac))
                indices = np.random.permutation(n_pairs)
                test_idx = indices[:n_test]
                train_idx = indices[n_test:]

                Z_plus_train, Z_minus_train = Z_plus[train_idx], Z_minus[train_idx]
                Z_plus_test, Z_minus_test = Z_plus[test_idx], Z_minus[test_idx]
                n_train = len(train_idx)
                print(f"    Split: {n_train} train, {n_test} test")

                # Optimize on train pairs only
                print(f"    Optimizing ({opt_cfg['method']})...")
                v = optimizer(Z_plus_train, Z_minus_train, opt_cfg)

                # Create layer output directory
                layer_safe = layer.replace('/', '_')
                layer_dir = pos_dir / f"layer={layer_safe}"
                layer_dir.mkdir(parents=True, exist_ok=True)

                # Evaluate
                ctx = {
                    'layer': layer,
                    'v': v,
                    'Z_plus': Z_plus_train,
                    'Z_minus': Z_minus_train,
                    'Z_plus_test': Z_plus_test,
                    'Z_minus_test': Z_minus_test,
                    'out_dir': layer_dir,
                }

                print(f"    Evaluating...")
                layer_results = run_evals(ctx, eval_cfg)
                pos_results[layer] = layer_results

                # Save concept vector and results
                if v is not None:
                    np.save(layer_dir / "concept_vector.npy", v)
                    batch_vectors[layer].append(v)
                    batch_metadata[layer].append({
                        'pos_index': pos_idx,
                        'move_number': pos.get('move_number'),
                        'n_pairs': n_pairs,
                        'n_train': n_train,
                        'n_test': n_test,
                    })

                save_json(layer_results, layer_dir / "results.json")

                # Print summary
                if 'spine_pair_separation' in layer_results:
                    sep = layer_results['spine_pair_separation']
                    test_cs = sep.get('test_constraint_satisfaction', sep.get('constraint_satisfaction', 0))
                    train_cs = sep.get('train_constraint_satisfaction', 0)
                    test_margin = sep.get('test_mean_margin', sep.get('mean_margin', 0))
                    print(f"    Constraint satisfaction: train={train_cs:.3f}, test={test_cs:.3f}")
                    print(f"    Test mean margin: {test_margin:.4f}")
                if 'sparsity' in layer_results:
                    sp = layer_results['sparsity']
                    print(f"    Sparsity: {sp['n_nonzero']}/{sp['dim']} non-zero ({sp['sparsity']:.3f})")

            save_json(pos_results, pos_dir / "results.json")
            all_results[pos_label] = pos_results
        
        except Exception as e:
            print(f"Exception occurred in processing {pos_label}: {e}")
            import traceback
            traceback.print_exc()
            all_results[pos_label] = {"error": str(e)}
            save_json({"error": str(e)}, pos_dir / "results.json")


    # 6. Aggregate: stack concept vectors per layer into batch files
    print(f"\n=== Saving batch concept vectors ===")
    summary = {
        'n_positions_total': len(positions),
        'n_positions_interesting': len(interesting),
        'layers': {},
    }

    for layer in layer_names:
        layer_safe = layer.replace('/', '_')
        batch_dir = run_dir / f"batch_layer={layer_safe}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        vectors = batch_vectors[layer]
        metadata = batch_metadata[layer]

        if len(vectors) > 0:
            batch = np.stack(vectors)
            np.save(batch_dir / "concept_vectors.npy", batch)
            save_json(metadata, batch_dir / "metadata.json")
            print(f"  {layer}: saved {batch.shape} batch")
            summary['layers'][layer] = {
                'n_vectors': len(vectors),
                'shape': list(batch.shape),
            }
        else:
            print(f"  {layer}: no vectors to save")
            summary['layers'][layer] = {'n_vectors': 0}

    save_json(summary, run_dir / "results.json")
    save_json(all_results, run_dir / "all_position_results.json")
    print(f"\n=== Results saved to {run_dir} ===")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Spine concept vector pipeline")
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_dir = make_run_dir(cfg['experiment_name'])

    # Save config
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.safe_dump(cfg, f)

    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Run dir: {run_dir}")

    run_spine(cfg, run_dir)


if __name__ == "__main__":
    main()
