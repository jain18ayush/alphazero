import json
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

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

    for pos_idx, pos in enumerate(interesting):
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

            # Build spine tree
            levels = build_spine_tree(
                mct, board_snapshot, strong_net,
                max_depth=spine_cfg['max_depth'],
                layer_name=layer,
                sort_key=spine_cfg.get('sort_key', 'N'),
            )

            # Extract pairs
            Z_plus, Z_minus, pairs_meta = extract_pairs_from_spine(levels)
            n_pairs = len(pairs_meta)
            print(f"    Extracted {n_pairs} pairs")

            if n_pairs == 0:
                print(f"    No pairs extracted, skipping")
                continue

            dim = Z_plus.shape[1]
            print(f"    Activation dim: {dim}")

            # Optimize
            print(f"    Optimizing ({opt_cfg['method']})...")
            v = optimizer(Z_plus, Z_minus, opt_cfg)

            # Create layer output directory
            layer_safe = layer.replace('/', '_')
            layer_dir = pos_dir / f"layer={layer_safe}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            # Evaluate
            ctx = {
                'layer': layer,
                'v': v,
                'Z_plus': Z_plus,
                'Z_minus': Z_minus,
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
                })

            save_json(layer_results, layer_dir / "results.json")

            # Print summary
            if 'spine_pair_separation' in layer_results:
                sep = layer_results['spine_pair_separation']
                print(f"    Constraint satisfaction: {sep['constraint_satisfaction']:.3f}")
                print(f"    Mean margin: {sep['mean_margin']:.4f}")
            if 'sparsity' in layer_results:
                sp = layer_results['sparsity']
                print(f"    Sparsity: {sp['n_nonzero']}/{sp['dim']} non-zero ({sp['sparsity']:.3f})")

        save_json(pos_results, pos_dir / "results.json")
        all_results[pos_label] = pos_results

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
