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
    
    # 1. Collect trajectory pairs
    print("\n=== Collecting trajectory pairs ===")
    dataset_cfg = cfg['dataset'].copy()
    dataset_cfg['seed'] = seed
    dataset_cfg['checkpoint_path'] = cfg['model']['checkpoint_path']
    dataset_cfg['model_name'] = cfg['model']['name']
    
    pairs = DATASETS.get(cfg['dataset']['source'])(dataset_cfg)
    print(f"Collected {len(pairs)} trajectory pairs")
    
    if len(pairs) < 3:
        raise ValueError(f"Only {len(pairs)} pairs collected. Need at least 3.")
    
    # 2. Load model for activation extraction
    print("\n=== Loading model ===")
    net = MODELS.get(cfg['model']['source'])(cfg['model'])
    
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
    
    run_dynamic(cfg, run_dir)


if __name__ == "__main__":
    main()