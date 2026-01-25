import json 
import yaml
import argparse
import logging
from datetime import datetime 
from typing import Any, Dict
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.registries import DATASETS, MODELS, HOOKS, OPTIMIZERS, EVALUATORS
import src.datasets.datasets
import src.concepts.concepts
import src.models.models 
import src.optimizers.optimizers

from src.hooks.extract import extract_features_by_layer
from src.concepts.registry import split_by_concept, ConceptConfig
from src.evaluator.registry import run_evals


def setup_logging(run_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure logging to both file and terminal."""
    logger = logging.getLogger("concept_probe")
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def load_yaml(path: str): 
    with open(path, "r") as f: 
        return yaml.safe_load(f)

def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def make_run_dir(experiment_name: str, root: str = "runs") -> Path: 
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / experiment_name / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir 

def run_single_layer(cfg, run_dir, logger): 
    seed = cfg['seed']
    
    # Dataset
    logger.info("Loading dataset...")
    ds_cfg = cfg['dataset']
    ds_builder = DATASETS.get(ds_cfg['source'])
    dataset = ds_builder(ds_cfg)
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Concept
    logger.info("Splitting by concept...")
    concept_cfg = ConceptConfig.from_dict(cfg['concept'])
    X_plus, X_minus = split_by_concept(dataset, concept_cfg)
    logger.info(f"Concept split: {len(X_plus)} positive, {len(X_minus)} negative")

    # Model
    logger.info("Loading model...")
    model_cfg = cfg['model']
    model_builder = MODELS.get(model_cfg['source'])
    net = model_builder(model_cfg)
    logger.info(f"Model loaded: {model_cfg['source']}")

    # Hooks / feature extraction    
    layers = cfg["hooks"]["layers"]
    layer_names = [l["name"] for l in layers]
    logger.info(f"Extracting features for layers: {layer_names}")
    
    Zp_by_layer = extract_features_by_layer(net, X_plus, layers)
    Zm_by_layer = extract_features_by_layer(net, X_minus, layers)
    logger.info("Feature extraction complete")

    # Optimization setup
    opt_cfg = cfg['optimization']
    eval_cfg = cfg['evaluation']
    optimizer = OPTIMIZERS.get(opt_cfg['method'])
    logger.info(f"Optimizer: {opt_cfg['method']}")
    
    v_by_layer = {}
    results = {}

    for layer in layer_names:
        logger.info(f"Processing layer: {layer}")
        
        Zp = Zp_by_layer[layer]
        Zm = Zm_by_layer[layer]

        Zp_train, Zp_test = train_test_split(
            Zp,
            test_size=eval_cfg['test_split'],
            random_state=seed,
            shuffle=False,
        )
        Zm_train, Zm_test = train_test_split(
            Zm,
            test_size=eval_cfg['test_split'],
            random_state=seed,
            shuffle=False,
        )
        logger.debug(f"  Train: {len(Zp_train)} pos, {len(Zm_train)} neg")
        logger.debug(f"  Test: {len(Zp_test)} pos, {len(Zm_test)} neg")

        logger.info(f"  Optimizing concept vector...")
        v_by_layer[layer] = optimizer(
            Zp_train[:opt_cfg['batch_size']], 
            Zm_train[:opt_cfg['batch_size']], 
            opt_cfg
        )

        layer_str = f"layer={layer}"
        layer_dir = run_dir / layer_str
        layer_dir.mkdir(parents=True, exist_ok=True)

        ctx = {
            "layer": layer,
            "Zp_test": Zp_test,
            "Zm_test": Zm_test,
            "v": v_by_layer[layer],
            "out_dir": layer_dir,
        }

        logger.info(f"  Running evaluations...")
        results[layer] = run_evals(ctx, eval_cfg)
        
        with open(layer_dir / "eval_results.json", "w") as f:
            json.dump(results[layer], f, indent=2)
        logger.info(f"  Layer {layer} complete. Results saved.")

    # Save aggregated results
    big_results_path = run_dir / "eval_results.json"
    with open(big_results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"All results saved to {big_results_path}")


def main(): 
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    run_dir = make_run_dir(cfg['experiment_name'])
    save_yaml(cfg, run_dir / 'config.yaml')
    
    logger = setup_logging(run_dir)
    logger.info(f"Experiment: {cfg['experiment_name']}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config saved to {run_dir / 'config.yaml'}")
    
    import time
    start_time = time.time()
    
    try:
        run_single_layer(cfg, run_dir, logger)
    except Exception as e:
        logger.exception(f"Run failed with error: {e}")
        raise
    
    elapsed_time = time.time() - start_time
    logger.info(f"Run complete. Total time: {elapsed_time:.2f}s")


if __name__ == "__main__":
    main()