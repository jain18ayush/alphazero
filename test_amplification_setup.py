#!/usr/bin/env python3
"""
Quick test script to verify the amplification pipeline setup
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("1. Testing imports...")
try:
    import numpy as np
    print("   ✓ numpy imported")
    
    import torch
    print("   ✓ torch imported")
    
    import yaml
    print("   ✓ yaml imported")
    
    from alphazero.base import Board
    print("   ✓ alphazero imported")
    
    from src.registries import DATASETS, MODELS
    import src.datasets.datasets
    import src.models.models
    print("   ✓ src registries imported")
    
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

print("\n2. Loading config and model...")
try:
    with open("configs/amplify_pos2812.yaml") as f:
        cfg = yaml.safe_load(f)
    print(f"   ✓ Config loaded: {cfg['experiment_name']}")
    
    net = MODELS.get(cfg["model"]["source"])(cfg["model"])
    print("   ✓ Model loaded")
    
except Exception as e:
    print(f"   ✗ Config/Model error: {e}")
    sys.exit(1)

print("\n3. Loading concept vectors...")
try:
    for item in cfg["concept_vectors"]:
        vec = np.load(item["path"])
        print(f"   ✓ {item['name']}: shape {vec.shape}")
        
except Exception as e:
    print(f"   ✗ Concept vector error: {e}")
    sys.exit(1)

print("\n4. Loading positions...")
try:
    ds_cfg = cfg["positions"].copy()
    ds_cfg["net"] = net
    positions = DATASETS.get(ds_cfg["source"])(ds_cfg)
    print(f"   ✓ Loaded {len(positions)} positions")
    
except Exception as e:
    print(f"   ✗ Position loading error: {e}")
    sys.exit(1)

print("\n5. Ready to run amplification experiment")
print(f"   Model: {cfg['model']['name']}")
print(f"   Concepts: {[c['name'] for c in cfg['concept_vectors']]}")
print(f"   Positions: {len(positions)}")
print(f"   Alpha range: {cfg['alpha_range']}")
print(f"   n_sims: {cfg['n_sims']}")

print("\n✓ All checks passed! Ready for amplification experiment")
