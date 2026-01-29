# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains two major components:

1. **AlphaZero Implementation** (`alphazero/`): A complete implementation of the AlphaZero reinforcement learning algorithm for board games (Othello, TicTacToe, Connect4)
2. **Concept Probing Pipeline** (`src/`, `pipeline.py`, `rollout_pipeline.py`): Research code for extracting and analyzing learned concepts from neural networks using linear probes

The project combines game-playing AI with interpretability research to understand what strategies and concepts AlphaZero networks learn.

## Development Commands

### Setup
```bash
# Install the package in editable mode
pip install -e .

# Download pre-trained AlphaZero models from HuggingFace
alphazero --download
# OR
python alphazero/download.py
```

### Testing
```bash
# Run package tests
alphazero --test

# Get help/documentation
alphazero --help
```

### AlphaZero Training
```bash
# Train AlphaZero (uses config in run.sh)
bash alphazero/run.sh

# Train with custom config
python alphazero/trainer.py -g othello -e experiment_name -c path/to/config.json

# Estimate training time
python alphazero/trainer.py -g othello -x

# Freeze default configurations to JSON
python alphazero/trainers.py -f
```

### Play Against AlphaZero
```bash
# Play via CLI (board saved as PNG in outputs/)
python alphazero/game_cli.py --othello --mcts
python alphazero/game_cli.py --othello --net alphazero-othello --infos --bot-starts
python alphazero/game_cli.py --tictactoe --net alphazero-tictactoe --display pixel
```

### Run Concept Probing Experiments
```bash
# Static concept probing (e.g., "has corner")
python pipeline.py --config base.yaml

# Dynamic concept probing from MCTS trajectories
python rollout_pipeline.py --config az.yaml

# Dynamic concept probing with novelty filtering
python rollout_pipeline.py --config az_novelty.yaml
```

### Compare Players
```bash
python alphazero/contests.py
```

## Architecture

### AlphaZero Component (`alphazero/`)

**Core Architecture:**
- `base.py`: Abstract base classes (`Board`, `Player`, `PolicyValueNetwork`, `Config`)
- `players.py`: Player implementations (Random, Greedy, MCTS, AlphaZero, Human)
- `mcts.py`: Monte Carlo Tree Search (supports both rollout and neural evaluation)
- `trainer.py`: Full AlphaZero training loop (self-play → optimize → evaluate)

**Game Implementations:**
- `games/registers.py`: Central registry mapping game names to boards/networks/configs
- `games/othello.py`: Othello game logic, board, and neural network architecture
- `games/tictactoe.py`: TicTacToe implementation
- `games/connect4.py`: Connect4 implementation

**Training Infrastructure:**
- `schedulers.py`: Temperature schedulers for MCTS exploration during training
- `timers.py`: Time estimation utilities for training duration
- `arena.py`: Tournament system for evaluating players (supports parallel games)
- `visualization.py`: Training graphs (loss curves, eval results)

**Key Design Patterns:**
- Games are registered in `games/registers.py` using dictionaries (`BOARDS_REGISTER`, `NETWORKS_REGISTER`, `CONFIGS_REGISTER`)
- All boards inherit from `Board` base class with standardized interface
- Neural networks inherit from `PolicyValueNetwork` and output (policy, value) pairs
- Training uses self-play with data augmentation (reflections, rotations)

### Concept Probing Component (`src/`)

**Registry System (`src/registries.py`):**
Central registry objects for pluggable components:
- `DATASETS`: Data sources (self-play games, MCTS trajectories, Kaggle dataset)
- `CONCEPTS`: Static concept definitions (e.g., "has_corner")
- `MODELS`: Model loaders (file-based, base models)
- `OPTIMIZERS`: Optimization methods (convex, dynamic_convex)
- `EVALUATORS`: Evaluation metrics (accuracy, sparsity, constraint satisfaction)
- `HOOKS`: Activation extraction hooks

**Pipelines:**
- `pipeline.py`: Static concept probing
  - Collects game positions → splits by concept → extracts features → optimizes sparse linear probe → evaluates
  - Uses `extract_features_by_layer()` to capture activations from multiple layers in one forward pass

- `rollout_pipeline.py`: Dynamic concept probing
  - Collects MCTS trajectory pairs (chosen vs rejected) → extracts paired activations → optimizes concept vector → evaluates
  - Supports novelty filtering to detect whether discovered concepts are novel vs already known
  - Function `run_dynamic()` is the basic version
  - Function `run_dynamic_with_novelty()` adds novelty filtering against human/AZ baseline games

**Data Collection (`src/datasets/datasets.py`):**
- `self_play`: Generate random game positions
- `mcts_trajectories`: Collect (chosen, rejected) trajectory pairs from MCTS search
  - Uses `follow_trajectory()` to trace most-visited path through MCTS tree
  - Filters out ambiguous scenarios where value gap and visit ratio are both low
- `kaggle_othello_games`: Load real human games from Kaggle dataset for novelty baseline
  - Parses standard Othello algebraic notation (e.g., "f5d6c3...")

**Activation Extraction (`src/hooks/extract.py`):**
- `extract_features_by_layer()`: Batch extraction from multiple layers (static concepts)
- `extract_paired_trajectory_activations()`: Extract activations for trajectory pairs (dynamic concepts)
- `get_single_activation()`: Helper for single board state
- Uses PyTorch forward hooks to capture intermediate layer outputs

**Optimization (`src/optimizers/optimizers.py`):**
- `convex`: L1-minimization with separating hyperplane constraints (static concepts)
  - Constraint: v·z⁺ ≥ v·z⁻ + margin for all positive/negative pairs
- `dynamic_convex`: Extends to trajectory data (dynamic concepts)
  - Constraint: v·z_chosen[t] ≥ v·z_rejected[t] + margin for all timesteps t
- `dynamic_convex_single_player`: Filters to every-other timestep (single player concepts)
- All use CVXPY with ECOS/SCS solvers

**Evaluation (`src/evaluator/evals.py`):**
- `acc`: Classification accuracy at threshold
- `hist`: Histogram visualization of projections
- `trajectory_constraint_satisfaction`: Fraction of timestep constraints satisfied
- `trajectory_pair_accuracy`: Trajectory-level accuracy (mean chosen > mean rejected)
- `sparsity`: Count of non-zero dimensions

**Configuration:**
- Experiments configured via YAML files (see `base.yaml`, `az.yaml`, `az_novelty.yaml`)
- Structure: experiment_name, seed, dataset config, model config, hooks (layers), optimization params, evaluation params

## Important Implementation Details

### Board Representation
- AlphaZero networks expect **player-relative input**: multiply grid by current player (1 or -1)
- This is handled in `_to_model_input()` in `src/hooks/extract.py`
- Convention: `input = player * grid` so network always sees from perspective of player-to-move

### OthelloBoard Non-Standard Starting Position
- **Critical**: OthelloBoard uses a **non-standard starting position** with colors swapped:
  - OthelloBoard: d4=Black(1), e4=White(-1), d5=White(-1), e5=Black(1)
  - Standard Othello: d4=White(-1), e4=Black(1), d5=Black(1), e5=White(-1)
- The Kaggle Othello dataset uses standard notation
- `parse_othello_game()` in `src/datasets/datasets.py` handles this by:
  - Simulating games with standard Othello rules
  - Flipping all colors (multiply grid by -1, negate player)
  - Implementing proper move validation with `_play_standard_othello_move()`
- This ensures Kaggle game positions are compatible with OthelloBoard-trained networks

### MCTS Trajectory Collection
- Scenario 3 filtering is critical: must have meaningful gap in either value OR visit count
- Default thresholds: `min_value_gap=0.15`, `min_visit_ratio=1.10`
- Trajectories follow most-visited child at each step
- Both chosen and rejected trajectories are extracted to the same `depth` parameter
- **Node attributes**: Use `N` for visit count and `Q` for value (not `visit_count` or `value`)
- The `Q` value is already normalized (average win rate), no need to divide by `N`

### Model Checkpoints
- Pre-trained models downloaded to `models/` directory via HuggingFace Hub
- Checkpoint path structure: `{checkpoint_path}/{model_name}.pt`
- Board size must match between config and model (critical for Othello: use size 8 for Kaggle dataset)

### Registry Pattern
- All registries use decorator pattern: `@REGISTRY.register("name")`
- Retrieve with `REGISTRY.get("name")`
- To add new components: import the module containing `@register` decorators before first use
- Example: `import src.datasets.datasets` registers all dataset builders

### Novelty Filtering
- Requires collecting baseline activations from both human games and AZ self-play
- Builds filters per-layer to determine if concept is novel or already known
- Implementation in `src/novelty/novelty_filter.py` (built by `run_dynamic_with_novelty()`)

### Training Sample Format
- Samples normalized to player=1 perspective via `sample.normalize()`
- Data augmentation creates 8 symmetries: original + reflection + 3 rotations of each
- Early moves (first 2) skip augmentation to avoid duplicates

## File Organization

### Core AlphaZero Files
- `alphazero/base.py` - Base classes for boards, players, networks
- `alphazero/trainer.py` - Training loop and Sample class
- `alphazero/mcts.py` - MCTS implementation
- `alphazero/players.py` - All player types
- `alphazero/arena.py` - Game tournament system

### Concept Probing Files
- `src/registries.py` - Registry definitions
- `src/registry.py` - Registry base class
- `src/datasets/datasets.py` - Dataset collection
- `src/optimizers/optimizers.py` - Concept vector optimization
- `src/evaluator/evals.py` - Evaluation metrics
- `src/hooks/extract.py` - Activation extraction
- `src/concepts/concepts.py` - Static concept definitions
- `src/novelty/` - Novelty filtering (optional)

### Pipeline Scripts
- `pipeline.py` - Static concept probing pipeline
- `rollout_pipeline.py` - Dynamic concept probing pipeline

### Configuration Examples
- `base.yaml` - Static concept probing config (e.g., "has_corner")
- `az.yaml` - Dynamic concept probing from MCTS
- `az_novelty.yaml` - Dynamic with novelty filtering

## Models Directory Structure
```
models/
├── alphazero-othello.pt          # Main trained model
├── alphazero-othello/
│   ├── config.json               # Training configuration
│   ├── loss.json                 # Training loss history
│   ├── eval.json                 # Evaluation results
│   └── checkpoints/              # Intermediate checkpoints
```

## Experiment Results Structure
```
runs/
└── {experiment_name}/
    └── {timestamp}/
        ├── config.yaml           # Experiment config
        ├── run.log               # Execution log (pipeline.py)
        ├── results.json          # Aggregated results
        ├── novelty_stats.json    # Novelty filtering stats (if enabled)
        └── layer={layer_name}/
            ├── results.json      # Layer-specific results
            ├── concept_vector.npy # Learned concept vector
            └── proj_hist.png     # Projection histogram
```
