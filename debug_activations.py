import yaml
import numpy as np
from src.registries import DATASETS, MODELS

# Import to register datasets
import src.datasets.datasets
import src.models.models

from src.hooks.extract import extract_paired_trajectory_activations

# Load config  
cfg = yaml.safe_load(open('az.yaml'))

# Collect one pair
cfg['dataset']['n_pairs'] = 1  # Just get 1 pair for debugging
pairs = DATASETS.get(cfg['dataset']['source'])(cfg['dataset'])
print(f"Collected {len(pairs)} pairs")

# Load model
net = MODELS.get(cfg['model']['source'])(cfg['model'])

# Extract activations for first pair
pair = pairs[0]
print(f"\nPair 0:")
print(f"  Chosen trajectory: {len(pair['chosen'])} states")
print(f"  Rejected trajectories: {len(pair['rejected'])} trajectories")

# Check if states are actually different
print("\nChosen trajectory grids (first 3 states):")
for i, state in enumerate(pair['chosen'][:3]):
    print(f"  State {i}: player={state['player']}, grid hash={hash(state['grid'].tobytes())}")

# Extract activations
act_pairs = extract_paired_trajectory_activations(net, [pair], 'fc1')
act = act_pairs[0]

print(f"\nActivations shape:")
print(f"  Chosen: {act['chosen'].shape}")

print(f"\nChosen activations (first 5 dims, first 3 timesteps):")
for t in range(min(3, len(act['chosen']))):
    print(f"  t={t}: {act['chosen'][t][:5]}")

print(f"\nAre chosen activations identical across timesteps?")
for t in range(1, len(act['chosen'])):
    if np.allclose(act['chosen'][0], act['chosen'][t]):
        print(f"  t=0 vs t={t}: IDENTICAL!")
    else:
        diff = np.mean(np.abs(act['chosen'][0] - act['chosen'][t]))
        print(f"  t=0 vs t={t}: different (mean_abs_diff={diff:.4f})")
