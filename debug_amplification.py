#!/usr/bin/env python3
"""
Debug script for amplification pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Step 1: Loading YAML config")
import yaml
with open("configs/amplify_pos2812.yaml") as f:
    cfg = yaml.safe_load(f)
print(f"✓ Config loaded: {cfg['experiment_name']}")

print("\nStep 2: Importing registries...")
from src.registries import DATASETS, MODELS
import src.datasets.datasets
import src.models.models
print("✓ Registries imported")

print("\nStep 3: Loading model...")
net = MODELS.get(cfg["model"]["source"])(cfg["model"])
print("✓ Model loaded")

print("\nStep 4: Loading positions...")
ds_cfg = cfg["positions"].copy()
ds_cfg["net"] = net
print(f"  Dataset config: {ds_cfg}")
print(f"  Creating dataset of type: {ds_cfg['source']}")

try:
    dataset_fn = DATASETS.get(ds_cfg["source"])
    print(f"  ✓ Got dataset function: {dataset_fn}")
    positions = dataset_fn(ds_cfg)
    print(f"✓ Loaded {len(positions)} positions")
except Exception as e:
    print(f"✗ Error loading positions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: Checking evaluation dataset...")
if "evaluation_dataset" in cfg:
    from src.amplification.amplify import load_amplification_eval_dataset
    eval_cfg = cfg["evaluation_dataset"]
    print(f"  Loading from: {eval_cfg['path']}")
    try:
        eval_dataset = load_amplification_eval_dataset(
            eval_cfg["path"],
            max_positions=eval_cfg.get("max_positions"),
        )
        print(f"✓ Loaded {len(eval_dataset)} evaluation positions")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

print("\nStep 6: Loading concept vectors...")
import numpy as np
for item in cfg["concept_vectors"]:
    try:
        vec = np.load(item["path"])
        print(f"✓ {item['name']}: shape {vec.shape}, layer {item['layer']}")
    except Exception as e:
        print(f"✗ Failed to load {item['name']}: {e}")

print("\n✓ All components loaded successfully!")
print(f"\nReady to run amplification with:")
print(f"  - Model: {cfg['model']['name']}")
print(f"  - Positions: {len(positions)}")
print(f"  - Concepts: {len(cfg['concept_vectors'])}")
print(f"  - Alpha range: {cfg['alpha_range']}")
print(f"  - n_sims: {cfg['n_sims']}")
