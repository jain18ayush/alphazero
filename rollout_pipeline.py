import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from src.registries import DATASETS, MODELS, OPTIMIZERS, EVALUATORS
from src.hooks.extract import extract_paired_trajectory_activations
from src.evaluator.registry import run_evals

# Import to register
import src.datasets.datasets
import src.models.models
import src.optimizers.optimizers


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj, path: Path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def make_run_dir(name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_dynamic(cfg: dict, run_dir: Path):
    seed = cfg['seed']
    rng = np.random.RandomState(seed)
    
    # 1. Load model first so it can be passed to datasets
    print("\n=== Loading model ===")
    net = MODELS.get(cfg['model']['source'])(cfg['model'])
    
    # 2. Collect trajectory pairs (pass model to dataset config)
    print("\n=== Collecting trajectory pairs ===")
    dataset_cfg = cfg['dataset'].copy()
    dataset_cfg['seed'] = seed
    dataset_cfg['net'] = net  # Pass pre-loaded model
    
    pairs = DATASETS.get(cfg['dataset']['source'])(dataset_cfg)
    print(f"Collected {len(pairs)} trajectory pairs")
    
    if len(pairs) < 3:
        raise ValueError(f"Only {len(pairs)} pairs collected. Need at least 3.")
    
    # 3. Train/test split (by pairs)
    n = len(pairs)
    test_frac = cfg['evaluation']['test_split']
    n_test = max(1, int(n * test_frac))
    
    idx = rng.permutation(n)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    
    pairs_train = [pairs[i] for i in train_idx]
    pairs_test = [pairs[i] for i in test_idx]
    
    print(f"Split: {len(pairs_train)} train, {len(pairs_test)} test")
    
    # 4. Process each layer
    layer_names = [l['name'] for l in cfg['hooks']['layers']]
    opt_cfg = cfg['optimization']
    eval_cfg = cfg['evaluation']
    
    optimizer = OPTIMIZERS.get(opt_cfg['method'])
    
    results = {}
    
    print(f"\n=== Processing {len(layer_names)} layers ===")
    
    for layer in layer_names:
        print(f"\nLayer: {layer}")
        
        # Extract activations
        print("  Extracting activations...")
        try:
            act_train = extract_paired_trajectory_activations(net, pairs_train, layer)
            act_test = extract_paired_trajectory_activations(net, pairs_test, layer)
        except KeyError as e:
            print(f"  Skipping layer: {e}")
            continue
        
        # Check dimensions
        dim = act_train[0]['chosen'].shape[1]
        print(f"  Activation dim: {dim}")
        
        # Subsample for optimization if needed
        batch_size = opt_cfg.get('batch_size', len(act_train))
        act_train_batch = act_train[:batch_size]
        
        # Optimize
        print(f"  Optimizing ({opt_cfg['method']}, {len(act_train_batch)} pairs)...")
        v = optimizer(act_train_batch, opt_cfg)
        
        # Create output directory for this layer
        layer_safe = layer.replace('/', '_')
        layer_dir = run_dir / f"layer={layer_safe}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate
        ctx = {
            'layer': layer,
            'v': v,
            'activation_pairs_test': act_test,
            'activation_pairs_train': act_train,
            'out_dir': layer_dir,
        }
        
        print("  Evaluating...")
        layer_results = run_evals(ctx, eval_cfg)
        results[layer] = layer_results
        
        # Save concept vector
        if v is not None:
            np.save(layer_dir / "concept_vector.npy", v)
        
        save_json(layer_results, layer_dir / "results.json")
        
        # Print summary
        if 'trajectory_constraint_satisfaction' in layer_results:
            sat = layer_results['trajectory_constraint_satisfaction']['constraint_satisfaction']
            print(f"  Constraint satisfaction: {sat:.3f}")
        if 'trajectory_pair_accuracy' in layer_results:
            acc = layer_results['trajectory_pair_accuracy']['pair_accuracy']
            print(f"  Pair accuracy: {acc:.3f}")
        if 'sparsity' in layer_results:
            sp = layer_results['sparsity']
            print(f"  Sparsity: {sp['n_nonzero']}/{sp['dim']} non-zero ({sp['sparsity']:.3f})")
    
    # Save all results
    save_json(results, run_dir / "results.json")
    print(f"\n=== Results saved to {run_dir} ===")
    
    return results


# Add to your main pipeline (after run_dynamic in your main file)

def run_dynamic_with_novelty(cfg: dict, run_dir: Path):
    """Extended pipeline with novelty filtering."""
    seed = cfg['seed']
    rng = np.random.RandomState(seed)
    
    # 1. Load model first so it can be passed to datasets
    print("\n=== Loading model ===")
    net = MODELS.get(cfg['model']['source'])(cfg['model'])
    
    # 2. Collect trajectory pairs (pass model to dataset config)
    print("\n=== Collecting trajectory pairs ===")
    dataset_cfg = cfg['dataset'].copy()
    dataset_cfg['seed'] = seed
    dataset_cfg['net'] = net  # Pass pre-loaded model
    
    pairs = DATASETS.get(cfg['dataset']['source'])(dataset_cfg)
    print(f"Collected {len(pairs)} trajectory pairs")
    
    if len(pairs) < 3:
        raise ValueError(f"Only {len(pairs)} pairs collected. Need at least 3.")
    
    # 3. NEW: Collect human and AZ game positions for novelty filtering
    if cfg.get('novelty_filtering', {}).get('enabled', False):
        print("\n=== Collecting games for novelty filtering ===")
        
        novelty_cfg = cfg['novelty_filtering']
        
        # Collect human game positions
        human_cfg = {
            'board_size': cfg['model']['board_size'],
            'n_positions': novelty_cfg.get('n_samples', 17184),
        }
        print("  Loading human games...")
        human_positions = DATASETS.get(novelty_cfg['human_source'])(human_cfg)
        
        # Collect AZ game positions
        az_cfg = {
            'board_size': cfg['model']['board_size'],
            'n_games': novelty_cfg.get('n_games', 100),
        }
        print("  Generating AZ games...")
        az_positions = DATASETS.get(novelty_cfg['az_source'])(az_cfg)
        
        print(f"  Human positions: {len(human_positions)}")
        print(f"  AZ positions: {len(az_positions)}")
        
        # Build novelty filters for each layer
        novelty_filters = {}
        layer_names = [l['name'] for l in cfg['hooks']['layers']]
        
        for layer in layer_names:
            try:
                from src.novelty.novelty_filter import build_novelty_filter
                filter_dir = run_dir / "novelty" / layer.replace('/', '_')
                
                novelty_filters[layer] = build_novelty_filter(
                    net=net,
                    human_positions=human_positions,
                    az_positions=az_positions,
                    layer=layer,
                    n_samples=novelty_cfg.get('n_samples', 17184),
                    save_dir=filter_dir,
                )
            except Exception as e:
                print(f"  Failed to build filter for {layer}: {e}")
                novelty_filters[layer] = None
    else:
        novelty_filters = {}
    
    # 4. Train/test split (as before)
    n = len(pairs)
    test_frac = cfg['evaluation']['test_split']
    n_test = max(1, int(n * test_frac))
    
    idx = rng.permutation(n)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    
    pairs_train = [pairs[i] for i in train_idx]
    pairs_test = [pairs[i] for i in test_idx]
    
    print(f"\nSplit: {len(pairs_train)} train, {len(pairs_test)} test")
    
    # 5. Process each layer WITH novelty filtering
    layer_names = [l['name'] for l in cfg['hooks']['layers']]
    opt_cfg = cfg['optimization']
    eval_cfg = cfg['evaluation']
    
    optimizer = OPTIMIZERS.get(opt_cfg['method'])
    
    results = {}
    novelty_stats = {
        'total': 0,
        'novel': 0,
        'not_novel': 0,
        'by_layer': {}
    }
    
    print(f"\n=== Processing {len(layer_names)} layers ===")
    
    for layer in layer_names:
        print(f"\nLayer: {layer}")
        
        # Extract activations
        print("  Extracting activations...")
        try:
            act_train = extract_paired_trajectory_activations(net, pairs_train, layer)
            act_test = extract_paired_trajectory_activations(net, pairs_test, layer)
        except KeyError as e:
            print(f"  Skipping layer: {e}")
            continue
        
        dim = act_train[0]['chosen'].shape[1]
        print(f"  Activation dim: {dim}")
        
        # Optimize
        batch_size = opt_cfg.get('batch_size', len(act_train))
        act_train_batch = act_train[:batch_size]
        
        print(f"  Optimizing ({opt_cfg['method']}, {len(act_train_batch)} pairs)...")
        v = optimizer(act_train_batch, opt_cfg)
        
        # NEW: Apply novelty filter
        is_novel = True
        novelty_info = None
        
        if layer in novelty_filters and novelty_filters[layer] is not None:
            print("  Applying novelty filter...")
            is_novel, novelty_info = novelty_filters[layer].filter_concept(v)
            print(f"    Novel: {is_novel}")
            
            novelty_stats['total'] += 1
            if is_novel:
                novelty_stats['novel'] += 1
            else:
                novelty_stats['not_novel'] += 1
            
            novelty_stats['by_layer'][layer] = {
                'is_novel': is_novel,
                **novelty_info
            }
        
        # Create output directory
        layer_safe = layer.replace('/', '_')
        layer_dir = run_dir / f"layer={layer_safe}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate
        ctx = {
            'layer': layer,
            'v': v,
            'activation_pairs_test': act_test,
            'activation_pairs_train': act_train,
            'out_dir': layer_dir,
            'is_novel': is_novel,  # Add to context
            'novelty_info': novelty_info,
        }
        
        print("  Evaluating...")
        layer_results = run_evals(ctx, eval_cfg)
        
        # Add novelty info to results
        if novelty_info is not None:
            layer_results['novelty'] = novelty_info
        
        results[layer] = layer_results
        
        # Save concept vector
        if v is not None:
            np.save(layer_dir / "concept_vector.npy", v)
        
        save_json(layer_results, layer_dir / "results.json")
        
        # Print summary
        if 'trajectory_constraint_satisfaction' in layer_results:
            sat = layer_results['trajectory_constraint_satisfaction']['constraint_satisfaction']
            print(f"  Constraint satisfaction: {sat:.3f}")
        if 'trajectory_pair_accuracy' in layer_results:
            acc = layer_results['trajectory_pair_accuracy']['pair_accuracy']
            print(f"  Pair accuracy: {acc:.3f}")
        if 'sparsity' in layer_results:
            sp = layer_results['sparsity']
            print(f"  Sparsity: {sp['n_nonzero']}/{sp['dim']} non-zero ({sp['sparsity']:.3f})")
    
    # Save all results with novelty stats
    save_json(results, run_dir / "results.json")
    save_json(novelty_stats, run_dir / "novelty_stats.json")
    
    print(f"\n=== Novelty filtering summary ===")
    print(f"Total concepts: {novelty_stats['total']}")
    print(f"Novel: {novelty_stats['novel']}")
    print(f"Not novel: {novelty_stats['not_novel']}")
    if novelty_stats['total'] > 0:
        pct = 100 * novelty_stats['novel'] / novelty_stats['total']
        print(f"Novelty rate: {pct:.1f}%")
    
    print(f"\n=== Results saved to {run_dir} ===")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_dir = make_run_dir(cfg['experiment_name'])

    # Save config
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.safe_dump(cfg, f)

    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Run dir: {run_dir}")

    # Check if novelty filtering is enabled
    if cfg.get('novelty_filtering', {}).get('enabled', False):
        print("Novelty filtering ENABLED")
        run_dynamic_with_novelty(cfg, run_dir)
    else:
        print("Novelty filtering DISABLED")
        run_dynamic(cfg, run_dir)


if __name__ == "__main__":
    main()