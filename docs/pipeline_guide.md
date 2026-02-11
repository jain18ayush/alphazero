# Pipeline Guide: From Paper to Code

## Table of Contents

1. [Overview](#overview)
2. [Paper-to-Code Mapping](#paper-to-code-mapping)
3. [Architecture: The Registry System](#architecture-the-registry-system)
4. [Pipeline 1: Rollout Pipeline (Dynamic Concepts)](#pipeline-1-rollout-pipeline)
5. [Pipeline 2: Spine Pipeline (Disagreement-Based Concepts)](#pipeline-2-spine-pipeline)
6. [Pipeline 3: Teachability Pipeline](#pipeline-3-teachability-pipeline)
7. [The Novelty Filter](#the-novelty-filter)
8. [How the Pipelines Fit Together](#how-the-pipelines-fit-together)
9. [Source Modules Reference](#source-modules-reference)
10. [Configuration Reference](#configuration-reference)
11. [Experiment Results Structure](#experiment-results-structure)

---

## Overview

This codebase implements the methodology from *Bridging the Human-AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero* (Schut et al., 2023). The goal is to extract novel, teachable concepts from AlphaZero's internal representations and present them as puzzles humans can learn from.

The implementation focuses on **Othello** rather than chess. The core idea is the same: AlphaZero learns concepts during self-play that humans don't know. We can find these concepts as directions (vectors) in the network's latent space, filter them for teachability and novelty, and present the best ones to humans.

### End-to-End Flow

```
Positions         Concept Discovery      Filtering           Human Teaching
---------         -----------------      ---------           --------------
AZ self-play  --> Rollout pipeline   --> Teachability    --> Prototypes
  or              or Spine pipeline      filter              + Puzzles
Human games                          --> Novelty filter
```

There are **four pipelines** (three implemented as scripts, one as a notebook workflow):

| Pipeline | Script | Config | Purpose |
|----------|--------|--------|---------|
| Rollout | `rollout_pipeline.py` | `configs/az.yaml` | Find concepts from MCTS trajectory pairs |
| Spine | `spine_pipeline.py` | `configs/spine.yaml` | Find concepts from MCTS decision trees at disagreement positions |
| Teachability | `teachability_pipeline.py` | `configs/teachability.yaml` | Filter concepts by whether a student network can learn them |
| Novelty | `novelty_prototype.ipynb` + `novelty_filter.ipynb` | `configs/novelty.yaml` | Filter concepts by whether they exist only in AZ's game space |

---

## Paper-to-Code Mapping

This section maps the key sections of the paper to their code implementations.

### Section 3: Concept Discovery

**Paper**: Find a concept vector `v` such that `v . z_chosen[t] >= v . z_rejected[t]` for all timesteps `t` in trajectory pairs.

**Code**:

| Paper Element | Code Location | Description |
|--------------|---------------|-------------|
| Trajectory pair collection | `src/datasets/datasets.py` :: `collect_trajectory_pairs()` | Plays random positions, runs MCTS, extracts best vs second-best trajectories |
| Scenario 3 filtering | `src/datasets/datasets.py` :: `get_trajectory_pair()` | Requires `value_gap >= min_value_gap` OR `visit_ratio >= min_visit_ratio` |
| Follow most-visited path | `src/datasets/datasets.py` :: `follow_trajectory()` | Traces most-visited children down the MCTS tree |
| Activation extraction | `src/hooks/extract.py` :: `extract_paired_trajectory_activations()` | PyTorch forward hooks capture layer outputs |
| Convex optimization (Eq. 4) | `src/optimizers/optimizers.py` :: `dynamic_convex()` | CVXPY: minimize `\|v\|_1` subject to per-timestep margin constraints |
| Single-player variant (Eq. 5) | `src/optimizers/optimizers.py` :: `dynamic_convex_single_player()` | Same but only even timesteps (player-to-move at root) |
| Spine-based discovery | `src/spine/spine.py` :: `build_spine_tree()` | Builds a decision tree from MCTS: best move spine + second-best branches |
| Disagreement filter | `src/spine/spine.py` :: `find_disagreement_positions()` | Two-pass filter finding positions where strong and weak models disagree |

### Section 4.1: Teachability Filter

**Paper**: Train a "student" network on concept prototypes. If it learns the teacher's policy better on concept positions than on random positions, the concept is teachable.

**Code**:

| Paper Element | Code Location | Description |
|--------------|---------------|-------------|
| Select prototypes (top 2.5%) | `src/teachability/teachability.py` :: `dynamic_prototypes_for_concept()` | Selects positions where the concept vector's constraints are satisfied across all rollout contrasts |
| Find student checkpoint | `src/teachability/teachability.py` :: `select_student_checkpoint()` | Sweeps checkpoints from latest to earliest, picks the first with overlap < threshold |
| KL-divergence distillation | `src/teachability/teachability.py` :: `distill_student()` | `F.kl_div(student_log_probs, teacher_probs)` for N epochs |
| 4-way benchmark | `src/teachability/teachability.py` :: `run_teachability_benchmark()` | Train on concept vs random, evaluate on concept vs random = 4 combinations |
| Teachability criterion | `src/teachability/teachability.py` :: `is_teachable()` | `gain = train_C_eval_C - train_R_eval_C > margin` |
| Top-1 agreement metric | `src/teachability/teachability.py` :: `measure_top1_agreement()` | Fraction of positions where student and teacher pick the same move |

### Section 4.2: Novelty Filter

**Paper**: Compute SVD basis for human and AZ game activations. A concept is novel if it is harder to reconstruct from the human basis than from the AZ basis, for all values of `k`.

**Code**:

| Paper Element | Code Location | Description |
|--------------|---------------|-------------|
| SVD basis computation | `src/novelty/novelty.py` :: `compute_basis()` | `U, Sigma, Vt = np.linalg.svd(activations)` returns `Vt.T` |
| Reconstruction error | `src/novelty/novelty.py` :: `reconstruction_error()` | Project `v` onto first `k` columns of basis, compute `\|v - v_proj\|^2` |
| Novelty score | `src/novelty/novelty.py` :: `is_novel()` | `score = err_human - err_az` for each `k`; novel if ALL scores > 0 |
| NoveltyFilter class | `src/novelty/novelty.py` :: `NoveltyFilter` | Wraps basis computation + filtering with geometric `k` progression |
| Build baselines (notebook) | `novelty_prototype.ipynb` | Loads Kaggle human games + AZ self-play, extracts activations, builds filter |
| Apply filter (notebook) | `novelty_filter.ipynb` | Loads existing concept vectors, runs `filter.check()` on each |

### Section 5: Evaluation

**Paper**: Accuracy, constraint satisfaction, sparsity, concept amplification.

**Code**:

| Paper Element | Code Location | Description |
|--------------|---------------|-------------|
| Test accuracy | `src/evaluator/evals.py` :: `accuracy_at_0()` | `(mean(plus > 0) + mean(minus <= 0)) / 2` |
| Constraint satisfaction | `src/evaluator/evals.py` :: `trajectory_constraint_satisfaction()` | Fraction of `(pair, timestep, subpar)` where `v . z_c >= v . z_r` |
| Pair accuracy | `src/evaluator/evals.py` :: `trajectory_pair_accuracy()` | Fraction of pairs where `mean(chosen scores) > mean(rejected scores)` |
| Sparsity | `src/evaluator/evals.py` :: `sparsity()` | Count of dimensions where `\|v_i\| > threshold` |
| Spine pair separation | `src/evaluator/evals.py` :: `spine_pair_separation()` | Train/test constraint satisfaction with histogram visualization |
| Nontriviality check | `src/evaluator/evals.py` :: `spine_nontriviality()` | Verifies concept vector is not all zeros |
| Temporal dynamics | `src/evaluator/evals.py` :: `trajectory_dynamics()` | Validates concept changes over trajectory (not just static) |
| Concept amplification | **NOT YET IMPLEMENTED** | The formula `z' = (1-a)z + a*b*(||z||/||v||)*v` has no code |

---

## Architecture: The Registry System

All pluggable components use a central registry pattern defined in `src/registry.py` and `src/registries.py`. This is the backbone that lets configs reference components by name.

```python
# src/registry.py - 22 lines
class Registry:
    def register(self, key: str):   # decorator: @REGISTRY.register("name")
    def get(self, key: str):         # lookup:   REGISTRY.get("name")
```

```python
# src/registries.py - 7 registries
DATASETS   = Registry("dataset")     # Data sources
CONCEPTS   = Registry("concept")     # Concept definitions
MODELS     = Registry("model")       # Model loaders
OPTIMIZERS = Registry("optimizer")   # Concept vector optimizers
EVALUATORS = Registry("evaluator")   # Evaluation metrics
HOOKS      = Registry("hook")        # Activation hooks
COUNTERFACTUALS = Registry("counterfactuals")
```

Components register themselves via decorators at import time. Pipeline scripts import the modules to trigger registration:

```python
# At the top of every pipeline script:
import src.datasets.datasets    # registers: self_play, model_play, mcts_trajectories, kaggle_othello_games, ...
import src.models.models        # registers: base, file, checkpoint_file
import src.optimizers.optimizers # registers: convex, dynamic_convex, dynamic_convex_single_player, spine_soft_margin, spine_hard_margin
import src.evaluator.evals      # registers: acc, hist, trajectory_constraint_satisfaction, sparsity, ...
```

A config file references registered names:

```yaml
# Example: configs/spine.yaml
optimization:
  method: "spine_soft_margin"    # -> OPTIMIZERS.get("spine_soft_margin")
positions:
  source: "batched_load_npy"     # -> DATASETS.get("batched_load_npy")
```

---

## Pipeline 1: Rollout Pipeline

**Script**: `rollout_pipeline.py`
**Configs**: `configs/az.yaml` (basic), `configs/az_novelty.yaml` (with novelty filtering)
**Entry point**: `run_dynamic()` or `run_dynamic_with_novelty()`

### What It Does

This pipeline discovers dynamic concept vectors from MCTS trajectory pairs. It generates random board positions, runs MCTS to get the best and second-best continuations, and finds a vector `v` in the network's latent space that separates them.

### Step-by-Step Walkthrough

```
run_dynamic(cfg, run_dir)
```

**Step 1: Load Model**
```python
net = MODELS.get(cfg['model']['source'])(cfg['model'])
```
Loads the AlphaZero model. The `"file"` source loads from a `.pt` checkpoint (`src/models/models.py:13`).

**Step 2: Collect Trajectory Pairs**
```python
pairs = DATASETS.get("mcts_trajectories")(dataset_cfg)
```
Calls `collect_trajectory_pairs()` in `src/datasets/datasets.py:298`. For each attempt:
1. Creates a random mid-game position (4-20 random moves)
2. Runs MCTS with `n_sims` simulations
3. Calls `get_trajectory_pair()` which:
   - Finds the best and second-best children by visit count
   - Applies Scenario 3 filter: requires `|Q_best - Q_second| >= min_value_gap` OR `N_best/N_second >= min_visit_ratio`
   - Follows the most-visited path down each child for `depth` steps via `follow_trajectory()`
4. Returns `{'chosen': [states...], 'rejected': [states...], 'value_gap': ..., ...}`

**Step 3: Train/Test Split**
```python
n_test = max(1, int(n * test_frac))
```
Randomly splits trajectory pairs (not individual positions) into train and test sets.

**Step 4: Per-Layer Processing**
For each layer in `cfg['hooks']['layers']`:

4a. **Extract Activations**: `extract_paired_trajectory_activations()` (`src/hooks/extract.py:158`) runs each board state through the network, captures the target layer's output via a PyTorch forward hook, and flattens it to a 1D vector. Returns `{'chosen': (T, dim), 'rejected': List[(T, dim)]}` per pair.

4b. **Optimize**: Calls the registered optimizer (e.g., `dynamic_convex_single_player`). CVXPY solves:
```
minimize ||v||_1
subject to: v . z_chosen[t] >= v . z_rejected[t] + margin
            for all pairs, for all even timesteps t
```

4c. **Evaluate**: Calls `run_evals()` (`src/evaluator/registry.py:4`) which iterates over evaluation specs in the config and dispatches to registered evaluators.

4d. **Save**: Writes `concept_vector.npy` and `results.json` per layer.

### The Novelty Variant

`run_dynamic_with_novelty()` extends the above by:
1. Loading human game positions (Kaggle dataset) and AZ self-play positions
2. Building a `NoveltyFilter` per layer from their activations
3. After optimization, calling `filter.check(v)` to tag each concept as novel or not

### Config Reference: `configs/az.yaml`

```yaml
dataset:
  source: mcts_trajectories
  board_size: 6                  # 6x6 Othello
  n_pairs: 50                   # Target trajectory pairs
  n_sims: 200                   # MCTS simulations per position
  depth: 5                      # Trajectory length
  min_value_gap: 0.30           # Scenario 3: Q-value difference threshold
  min_visit_ratio: 2.0          # Scenario 3: visit count ratio threshold

model:
  source: "file"
  board_size: 6
  name: "alphazero-othello"
  checkpoint_path: "models/"

hooks:
  layers:                        # Which layers to extract concepts from
    - {name: "conv1"}
    - {name: "conv2"}
    # ... conv3, conv4, fc1, fc2

optimization:
  method: dynamic_convex_single_player   # Equation 5 optimizer
  margin: 0.0                            # Constraint margin

evaluation:
  test_split: 0.2
  runs:
    - {name: activation_stats}                           # Activation scale diagnostics
    - {name: trajectory_dynamics, threshold: 0.01}       # Is it actually dynamic?
    - {name: trajectory_constraint_satisfaction, margin: 0.0}
    - {name: trajectory_pair_accuracy}
    - {name: sparsity, threshold: 1.0e-6}
```

---

## Pipeline 2: Spine Pipeline

**Script**: `spine_pipeline.py`
**Config**: `configs/spine.yaml`
**Entry point**: `run_spine()`

### What It Does

This is the primary concept discovery pipeline. Instead of random positions, it focuses on **disagreement positions** where a strong model and a weak model (earlier checkpoint) choose different moves. At each position, it builds a "spine" decision tree from MCTS and extracts (best-path, alternative-path) activation pairs for optimization.

This produces far more concept vectors than the rollout pipeline (one per position per layer, vs one globally), giving ~3,400+ vectors.

### Step-by-Step Walkthrough

```
run_spine(cfg, run_dir)
```

**Step 1: Load Positions**
```python
positions = DATASETS.get("batched_load_npy")(pos_cfg)
```
Loads pre-generated positions from `.npy` batch files in `batches/`. These are positions from AZ self-play games.

**Step 2: Load Strong and Weak Models**
- Strong: The fully trained AlphaZero model (`alphazero-othello-8`)
- Weak: An earlier training checkpoint (`alphazero-othello-8-chkpt-5.pt`)

**Step 3: Disagreement Filter**
```python
interesting = find_disagreement_positions(positions, strong_net, weak_net, ...)
```
`find_disagreement_positions()` (`src/spine/spine.py:10`) runs a two-pass filter:
- **Pass 1** (100 MCTS sims): Quick filter. Keep positions where strong and weak models pick different moves.
- **Pass 2** (1000 MCTS sims): Confirm disagreements with deeper search.

This selects "interesting" positions where the strong model knows something the weak model doesn't. The `interesting.npy` file caches these results.

**Step 4: Per-Position Processing**

For each interesting position (the main loop at `spine_pipeline.py:368`):

4a. **Run MCTS**: 10,000 simulations with the strong model. This builds a deep, well-explored search tree.

4b. **Build Spine Tree**: `build_spine_tree()` (`src/spine/spine.py:71`) recursively walks the MCTS tree:
- The **main path** (spine) follows the best move at every depth
- At each spine node, the **second-best** child spawns a branch
- Second-best branches follow only their own best moves (no further branching)
- Uses heap-style indexing: parent `(d, idx)` -> children `(d+1, 2*idx)` best, `(d+1, 2*idx+1)` second
- At each node, calls `get_single_activation()` to capture the network's layer output

4c. **Extract Pairs**: `extract_pairs_from_spine()` (`src/spine/spine.py:288`) creates `(Z_plus, Z_minus)` pairs:
- At each depth, the main-path node's activation = Z+
- Each alternative node's activation at the same depth = Z-
- This gives multiple pairs per position (one per branching depth)

4d. **Train/Test Split + Optimize**: Splits pairs, then runs the registered optimizer (default: `spine_soft_margin`):
```
minimize C * sum(xi) + l2 * ||v||^2
subject to: Z_plus @ v >= Z_minus @ v + margin - xi
            xi >= 0
```
This is a soft-margin SVM-like formulation (L2 regularization, slack variables). Compared to the rollout pipeline's L1-minimization, this produces denser vectors.

4e. **Evaluate + Visualize**: Runs evaluation metrics (`spine_pair_separation`, `sparsity`, `spine_nontriviality`) and generates:
- `pair_separation.png`: Histogram of Z+ vs Z- projections onto `v`
- `spine_tree.png`: Visual decision tree showing board states, moves, N/Q values

4f. **Save + Accumulate**: Saves per-position results and accumulates concept vectors into a batch file per layer.

### Spine Tree Visualization

`visualize_spine_decision_tree()` (`spine_pipeline.py:44-295`) renders the MCTS decision tree as a visual diagram:

- **Lane 0**: The spine (all best moves)
- **Lane k**: Nodes where the first second-best choice occurs at depth k
- Green arrows = best moves, orange arrows = second-best
- Each node card shows: board state (with pieces), move highlight (red square), N (visits), Q (value)

This keeps the layout O(max_depth) wide rather than O(2^max_depth).

### Config Reference: `configs/spine.yaml`

```yaml
positions:
  source: "batched_load_npy"       # Load from pre-generated batches
  folder: "batches/"
  board_size: 8                    # 8x8 standard Othello

disagreement:
  exists: interesting.npy          # Cached disagreement positions (skip recomputation)
  n_sims_pass1: 100                # Quick filter MCTS budget
  n_sims_pass2: 1000               # Confirmation MCTS budget

strong_model:
  name: "alphazero-othello-8"      # Fully trained model
  checkpoint_path: "models/"

weak_model:
  checkpoint_path: "models/alphazero-othello-8/checkpoints/alphazero-othello-8-chkpt-5.pt"

spine:
  n_sims: 10000                    # Deep MCTS for spine building (expensive!)
  max_depth: 5                     # Tree depth
  sort_key: "N"                    # Sort children by visit count

hooks:
  layers:
    - {name: "bn2"}                # Batch norm after conv2 (2048-dim)

optimization:
  method: "spine_soft_margin"      # Soft-margin SVM with L2 regularization
  margin: 1.0                      # Target separation margin
  C: 1.0                           # Slack penalty
  l2: 1.0e-3                       # L2 regularization strength

evaluation:
  test_split: 0.3                  # 30% held-out pairs for testing
  runs:
    - {name: "sparsity"}
    - {name: "spine_pair_separation"}
    - {name: "spine_nontriviality"}
    - {name: "diagnose_concept_vector"}
```

---

## Pipeline 3: Teachability Pipeline

**Script**: `teachability_pipeline.py`
**Config**: `configs/teachability.yaml`
**Entry point**: `run_teachability()`

### What It Does

This pipeline takes previously discovered concept vectors and tests whether they are "teachable" -- can a weaker student model learn the teacher's policy faster by training on concept-specific prototypes vs random positions?

### Step-by-Step Walkthrough

```
run_teachability(cfg, run_dir)
```

**Step 1: Load Teacher Model + Position Pool + Concept Vectors**
```python
teacher = _load_model(teacher_cfg)
positions = DATASETS.get(positions_cfg["source"])(positions_cfg)
concepts = DATASETS.get("concept_vectors_from_runs")(concept_cfg)
```
- Teacher: The fully trained AZ model
- Positions: A large pool of game positions (from `batches/`)
- Concepts: Previously discovered vectors loaded from `runs/concept_vectors/BasicVectors/`

**Step 2: For Each Concept** (the main loop at `teachability_pipeline.py:156`):

2a. **Find Dynamic Prototypes**: `dynamic_prototypes_for_concept()` (`src/teachability/teachability.py:45`):
- For each candidate position, runs MCTS and calls `extract_rollout_contrasts()` to get the optimal rollout and multiple subpar rollouts
- Scores each rollout state: `score = v . z` using `get_single_activation()`
- Accepts the position only if `optimal_score[t] >= subpar_score[t]` for ALL subpar rollouts and ALL timesteps
- These are positions where the concept is clearly active

2b. **Select Student Checkpoint**: `select_student_checkpoint()` (`src/teachability/teachability.py:322`):
- Sweeps available checkpoints from latest to earliest
- For each, measures top-1 agreement with the teacher on prototype positions
- Selects the latest checkpoint where agreement < `overlap_threshold` (default 0.2)
- This ensures the student doesn't already know the concept

2c. **4-Way Benchmark**: `run_teachability_benchmark()` (`src/teachability/teachability.py:254`):
- Splits prototypes into 80/20 train/test
- Samples random positions (same count, excluding prototypes)
- Runs four experiments:

| Train Set | Eval Set | Metric Name |
|-----------|----------|-------------|
| Concept prototypes | Concept test | `train_C_eval_C` |
| Concept prototypes | Random test | `train_C_eval_R` |
| Random positions | Concept test | `train_R_eval_C` |
| Random positions | Random test | `train_R_eval_R` |

Each experiment: load fresh student -> distill with KL divergence -> measure top-1 agreement with teacher.

2d. **Teachability Decision**: `is_teachable()` (`src/teachability/teachability.py:310`):
```python
gain = train_C_eval_C - train_R_eval_C
return gain > margin  # default margin = 0.05
```
A concept is teachable if training on its prototypes helps the student learn the teacher's moves *on concept positions* more than training on random positions does.

### Config Reference: `configs/teachability.yaml`

```yaml
concepts:
  source: "concept_vectors_from_runs"
  run_root: "runs/concept_vectors/BasicVectors"  # Where to find concept vectors
  layers: ["bn2"]
  max_concepts: 5                                 # Process at most N concepts

dynamic_prototypes:
  max_depth: 10          # Rollout contrast depth
  min_value_gap: 0.20    # Filter for meaningful contrasts
  max_prototypes: 200    # Cap on prototypes per concept
  max_positions: 10000   # Sample from position pool

teachability:
  n_sim: 200             # MCTS simulations for policy comparison
  epochs: 5              # Distillation training epochs
  lr: 1.0e-4             # Learning rate
  batch_size: 64         # Distillation batch size
  test_split: 0.2        # Prototype train/test split
  min_prototypes: 20     # Skip concept if fewer prototypes found
  teachability_margin: 0.05  # Minimum concept-specific gain

student_selection:
  overlap_threshold: 0.2     # Max teacher-student agreement
  checkpoint_paths: []       # List of checkpoint paths to sweep

teacher_model:
  source: "file"
  name: "alphazero-othello-8"
  checkpoint_path: "models/"

student_model:
  source: "checkpoint_file"
  checkpoint_path: "models/alphazero-othello-8/checkpoints/alphazero-othello-8-chkpt-5.pt"
  config_path: "models/alphazero-othello-8/config.json"
```

---

## The Novelty Filter

**Module**: `src/novelty/novelty.py`
**Notebooks**: `novelty_prototype.ipynb` (build baselines), `novelty_filter.ipynb` (apply filter)
**Pre-computed baselines**: `configs/novelty_V_az.npy`, `configs/novelty_V_human.npy`

### What It Does

The novelty filter determines whether a concept exists only in AZ's "knowledge space" (novel) or is already present in human games (not novel). It uses SVD to build basis vectors for each game space and measures how well each concept vector can be reconstructed from each basis.

### Building the Baselines (`novelty_prototype.ipynb`)

This notebook builds the SVD baselines from scratch:

**Cell 1-2**: Load config and model
```python
cfg = load_yaml('configs/novelty.yaml')
net = MODELS.get(cfg['model']['source'])(cfg['model'])
```

**Cell 3**: Load human game positions from Kaggle
```python
human_games = DATASETS.get(pos_cfg['source'])(pos_cfg)  # kaggle_othello_games
```
Loads ~25,000 positions from the Kaggle Othello dataset. The `parse_othello_game()` function handles the color-swap between standard Othello notation and the non-standard OthelloBoard starting position (see CLAUDE.md for details).

**Cell 4**: Generate AZ self-play positions
```python
az_games = DATASETS.get(ds_cfg['source'])(ds_cfg)  # model_play
```

**Cell 5**: Extract activations for both sets
```python
human_activations = extract_features_by_layer(net, human_games, cfg['hooks']['layers'])
az_activations = extract_features_by_layer(net, az_games, cfg['hooks']['layers'])
```
Single forward pass per set captures activations at all requested layers.

**Cell 6**: Build the filter
```python
filter = NoveltyFilter(human_activations['bn2'], az_activations['bn2'])
```
Constructor calls `compute_basis()` which runs `np.linalg.svd()` on each activation matrix. This produces:
- `V_human`: (dim, rank) basis vectors for the human game activation subspace
- `V_az`: (dim, rank) basis vectors for the AZ game activation subspace

These are saved to `configs/novelty_V_human.npy` and `configs/novelty_V_az.npy`.

### Applying the Filter (`novelty_filter.ipynb`)

This notebook loads pre-computed baselines and checks each discovered concept vector:

**Cell 1**: Load pre-computed baselines
```python
V_az = np.load('configs/novelty_V_az.npy')
V_human = np.load('configs/novelty_V_human.npy')
filter = NoveltyFilter(V_human=V_human, V_az=V_az)
```

**Cell 2**: Iterate over all concept vectors in a run directory
```python
concept_vectors = np.load(concept_vector_path)
filter_result = filter.check(concept_vectors)
```

### How `NoveltyFilter.check()` Works

For a concept vector `v`:
1. For each `k` in a geometric progression `[1, 2, 4, 8, ..., max_k]`:
   - Compute reconstruction error using first `k` human basis vectors: `err_human = ||v - V_human[:, :k] @ (V_human[:, :k].T @ v)||^2`
   - Compute reconstruction error using first `k` AZ basis vectors: `err_az = ||v - V_az[:, :k] @ (V_az[:, :k].T @ v)||^2`
   - Score = `err_human - err_az`
2. Novel if ALL scores > 0 (concept is harder to reconstruct from human games at every scale)

### Current Results

From `novelty_filter.ipynb`, out of ~3,400 concept vectors:
- **11 are marked as novel** (positions: 1449, 2549, 0038, 2751, 0001, 2165, 2361, 2812, 2027, 3287, 3134)
- Novelty scores are very small (1e-4 to 1e-3 range), indicating most concepts live in the shared subspace of both human and AZ games
- This is expected: most of what AZ knows, humans also know. The filter catches the rare exceptions.

### Config Reference: `configs/novelty.yaml`

```yaml
positions:
  source: "kaggle_othello_games"
  board_size: 8
  n_positions: 25000

dataset:
  source: "model_play"
  n_games: 400
  board_size: 8
  n_sims: 1000

model:
  source: "file"
  board_size: 8
  name: "alphazero-othello-8"
  checkpoint_path: "models/"

hooks:
  layers:
    - {name: "bn2"}
```

---

## How the Pipelines Fit Together

### The Full Discovery-to-Teaching Pipeline

```
                   DATA GENERATION
                   ==============
  AZ Self-Play ─────────────────┐
  (model_play dataset)          │
                                ▼
  Kaggle Human Games ──► novelty_prototype.ipynb ──► V_human.npy, V_az.npy
  (kaggle_othello_games)        (SVD baselines)

                   CONCEPT DISCOVERY
                   =================
  Batch Positions ──► spine_pipeline.py ──► runs/concept_vectors/BasicVectors/
  (batches/*.npy)     │                     pos_XXXX/layer=bn2/concept_vector.npy
                      │                     (~3,400 concept vectors)
                      │
                      ├─ Disagreement filter (strong vs weak model)
                      ├─ MCTS spine tree (10K sims)
                      ├─ Extract (Z+, Z-) pairs
                      └─ Optimize v (spine_soft_margin)

  Random Positions ──► rollout_pipeline.py ──► runs/concept_vectors_az/
  (generated on-fly)   │                       layer=*/concept_vector.npy
                       │                       (1 vector per layer)
                       ├─ MCTS trajectory pairs
                       ├─ Extract paired activations
                       └─ Optimize v (dynamic_convex_single_player)

                   FILTERING
                   =========
  concept_vectors ──► novelty_filter.ipynb ──► novelty_results.json
  + V_human, V_az     (NoveltyFilter.check)    (11 novel concepts identified)

  concept_vectors ──► teachability_pipeline.py ──► runs/teachability_filter/
  + position pool      │                           concept_XXXX/result.json
                       ├─ Find dynamic prototypes
                       ├─ Select student checkpoint
                       ├─ 4-way benchmark (distill + evaluate)
                       └─ is_teachable? (gain > 0.05)

                   TEACHING (MVP - not yet built)
                   ========
  Best novel          ──► Prototype selection  ──► Puzzle set
  teachable concept       + Board rendering        (5-10 puzzles)
```

### Data Flow Between Pipelines

1. **Position Generation -> Concept Discovery**:
   - `model_play` dataset generates AZ games saved to `batches/*.npy`
   - `batched_load_npy` dataset loads them into the spine pipeline
   - The rollout pipeline generates its own positions on-the-fly

2. **Concept Discovery -> Novelty Filtering**:
   - Spine pipeline saves `concept_vector.npy` per position per layer under `runs/`
   - Novelty filter notebook loads these vectors and the pre-computed SVD baselines
   - Output: a `novelty_results.json` mapping position IDs to novel/not-novel

3. **Concept Discovery -> Teachability Filtering**:
   - `concept_vectors_from_runs` dataset loader (`src/datasets/datasets.py:750`) walks a `runs/` directory tree, collecting all `concept_vector.npy` files with their layer metadata
   - Teachability pipeline loads these and the position pool, then runs the full benchmark

4. **Novelty + Teachability -> MVP**:
   - Concepts must pass BOTH filters to be candidates for human teaching
   - The best candidate's prototypes become the puzzle set

### Key Shared Components

All pipelines share these core modules:

| Module | Used By | Function |
|--------|---------|----------|
| `src/hooks/extract.py` | All pipelines | Activation extraction via forward hooks |
| `src/registries.py` | All pipelines | Component lookup by config name |
| `src/evaluator/registry.py` | Rollout + Spine | Dispatches evaluation runs from config |
| `src/optimizers/optimizers.py` | Rollout + Spine | CVXPY concept vector optimization |
| `alphazero/players.py` | All pipelines | `AlphaZeroPlayer` wraps MCTS + neural eval |
| `alphazero/mcts.py` | All pipelines | MCTS search tree (provides `.root`, `.children`, `.N`, `.Q`) |

---

## Source Modules Reference

### `src/hooks/extract.py` -- Activation Extraction

The interface between PyTorch and the concept probing math. All activation extraction goes through this module.

| Function | Signature | Description |
|----------|-----------|-------------|
| `extract_features_by_layer()` | `(net, positions, layers) -> {name: (N, D)}` | Batch extraction from ALL layers in a single forward pass. Used by novelty baseline construction and the static pipeline. |
| `get_single_activation()` | `(net, grid, player, layer_name) -> (D,)` | Single board state, single layer. Used by spine tree building (one call per node). |
| `extract_trajectory_activations()` | `(net, states, layer_name) -> (T, D)` | Sequential extraction for a trajectory. Calls `get_single_activation()` per state. |
| `extract_paired_trajectory_activations()` | `(net, pairs, layer_name) -> [{'chosen': (T,D), 'rejected': [(T,D),...]}]` | Handles Equation 5 format with multiple subpar trajectories per pair. |

All functions use `_to_model_input()` which multiplies the grid by the player to get player-relative input (critical convention).

### `src/optimizers/optimizers.py` -- Concept Vector Solvers

Six registered optimizers, each solving a variant of the concept vector optimization:

| Name | Norm | Constraints | Use Case |
|------|------|-------------|----------|
| `convex` | L1 | All positive vs all negative pairs | Static concepts |
| `dynamic_pairs` | L1 | Aligned pairs (row-wise) | Simple dynamic |
| `dynamic_convex` | L1 | Per-timestep constraints | Dynamic concepts |
| `dynamic_convex_single_player` | L1 | Even timesteps only, multiple subpar (Eq. 5) | Single-player dynamic |
| `spine_soft_margin` | L2 + slack | `Z+ @ v >= Z- @ v + margin - xi` | **Spine pipeline default** |
| `spine_hard_margin` | L1 | `Z+ @ v >= Z- @ v + margin` (no slack) | Sparse spine concepts |

The spine optimizers receive raw `(Z_plus, Z_minus)` numpy arrays. The dynamic optimizers receive a list of `{'chosen': ..., 'rejected': ...}` dicts.

### `src/evaluator/evals.py` -- Evaluation Metrics

Nine registered evaluators:

| Name | Input | Output | Purpose |
|------|-------|--------|---------|
| `acc` | `Zp_test`, `Zm_test`, `v` | `{acc: float}` | Binary accuracy at threshold |
| `hist` | Same + `out_dir` | Saves histogram PNG | Visual: plus vs minus projections |
| `trajectory_constraint_satisfaction` | `activation_pairs_test`, `v` | `{constraint_satisfaction, mean_margin, ...}` | Fraction of per-timestep constraints satisfied |
| `trajectory_pair_accuracy` | Same | `{pair_accuracy, ...}` | Fraction of pairs where chosen > rejected |
| `sparsity` | `v` | `{n_nonzero, sparsity, dim}` | Concept vector sparsity |
| `activation_stats` | `activation_pairs_test` | `{mean, std, recommended_margin}` | Activation scale for setting margin |
| `spine_pair_separation` | `Z_plus`, `Z_minus`, `v` | Train + test metrics + histogram | Primary spine evaluation with visualization |
| `spine_nontriviality` | `v` | `{is_trivial, l2_norm, effective_rank}` | Checks vector is not all zeros |
| `trajectory_dynamics` | `activation_pairs_test`, `v` | `{temporal_variance, is_dynamic}` | Validates concept changes over time |
| `diagnose_concept_vector` | `v` | Saves diagnostic PNG | Component distribution, sorted magnitudes, cumulative norm |

### `src/datasets/datasets.py` -- Data Sources

Nine registered dataset builders:

| Name | Config Keys | Returns | Description |
|------|-------------|---------|-------------|
| `batched_load_npy` | `folder` | `List[dict]` | Concatenates `.npy` files from a folder |
| `model_play` | `board_size, n_games, net, n_sims, ...` | `List[dict]` | Generates positions from AZ self-play |
| `self_play` | `board_size, n_games` | `List[dict]` | Random player games (fast, low quality) |
| `mcts_trajectories` | `board_size, n_pairs, n_sims, depth, ...` | `List[dict]` | Trajectory pairs from MCTS |
| `kaggle_othello_games` | `board_size, n_positions, ...` | `List[dict]` | Positions from Kaggle human games |
| `human_games` | `board_size, ...` | `List[dict]` | Auto-detect: Kaggle or fallback to random |
| `human_games_npy` | `path` | `List[dict]` | Load human positions from `.npy` file |
| `concept_vectors_from_runs` | `run_root, layers, max_concepts` | `List[dict]` | Load concept vectors from runs directory tree |

Each position dict has: `{'grid': np.ndarray, 'player': int, 'move_number': int}`.

### `src/spine/spine.py` -- MCTS Decision Tree Analysis

| Function | Description |
|----------|-------------|
| `find_disagreement_positions()` | Two-pass filter: quick MCTS + deep MCTS to find positions where strong and weak models disagree |
| `build_spine_tree()` | Recursive walk building heap-indexed tree. Main path spawns second-best branches. Each node gets activation extracted. |
| `extract_rollout_contrasts()` | Builds optimal rollout + multiple subpar rollouts with gap/ratio filters. Used by teachability. |
| `extract_pairs_from_spine()` | Converts spine tree levels into `(Z_plus, Z_minus)` pair arrays for optimization. |

### `src/novelty/novelty.py` -- Novelty Scoring

| Function/Class | Description |
|----------------|-------------|
| `compute_basis()` | SVD on activation matrix, returns right singular vectors as columns |
| `reconstruction_error()` | Projects `v` onto first `k` basis vectors, returns squared residual |
| `is_novel()` | Checks `err_human - err_az > 0` for all `k` in a provided list |
| `NoveltyFilter` | Class wrapping bases + geometric `k` progression. Constructor accepts either raw activations (computes SVD) or pre-computed bases. |

### `src/teachability/teachability.py` -- Teachability Testing

| Function | Description |
|----------|-------------|
| `dynamic_prototypes_for_concept()` | Finds positions where concept constraints hold across all rollout contrasts |
| `mcts_policy_for_positions()` | Gets MCTS policy distributions for a set of positions |
| `measure_top1_agreement()` | Fraction of positions where two models pick the same move |
| `distill_student()` | KL-divergence training of student on teacher's policies |
| `run_teachability_benchmark()` | Full 4-way experiment: train concept/random x eval concept/random |
| `is_teachable()` | `train_C_eval_C - train_R_eval_C > margin` |
| `select_student_checkpoint()` | Sweeps checkpoints to find one with low teacher agreement |

### `src/models/models.py` -- Model Loaders

| Name | Description |
|------|-------------|
| `base` | Fresh OthelloNet with random weights |
| `file` | Load from HuggingFace-style checkpoint (`OthelloNet.from_pretrained`) |
| `checkpoint_file` | Load from raw `.pt` weights + `config.json` |

---

## Configuration Reference

### All Config Files

| File | Pipeline | Board Size | Key Settings |
|------|----------|------------|--------------|
| `configs/base.yaml` | Static pipeline | 6 | `has_corner` concept, L1 convex optimizer, all layers |
| `configs/az.yaml` | Rollout pipeline | 6 | 50 trajectory pairs, 200 MCTS sims, `dynamic_convex_single_player` |
| `configs/az_novelty.yaml` | Rollout + novelty | 8 | 30 pairs, 200 sims, `dynamic_convex`, Kaggle human games |
| `configs/novelty.yaml` | Novelty baselines | 8 | 25K human positions, 400 AZ games, 1000 sims |
| `configs/spine.yaml` | Spine pipeline | 8 | 10K MCTS sims, `spine_soft_margin`, disagreement filter |
| `configs/teachability.yaml` | Teachability filter | 8 | 200 MCTS sims, 5 epochs, 4-way benchmark |

### Common Config Patterns

**Model loading** (appears in every config):
```yaml
model:
  source: "file"          # -> MODELS.get("file")
  name: "alphazero-othello-8"
  checkpoint_path: "models/"
  board_size: 8
```

**Layer selection** (determines activation dimensionality):
```yaml
hooks:
  layers:
    - {name: "bn2"}       # 32 channels x 8 x 8 = 2048 dims (for board_size=8)
```

Available layers for OthelloNet (8x8): `conv1`/`bn1` (2048), `conv2`/`bn2` (2048), `conv3`/`bn3` (1152), `conv4`/`bn4` (512), `fc1`/`fc_bn1` (256), `fc2`/`fc_bn2` (256), `fc_probs` (65).

**Evaluation** (list of registered evaluator names):
```yaml
evaluation:
  test_split: 0.2
  runs:
    - {name: "sparsity", threshold: 1.0e-6}
    - {name: "spine_pair_separation"}
```

---

## Experiment Results Structure

```
runs/
├── concept_vectors/
│   └── BasicVectors/                    # Spine pipeline output
│       ├── interesting.npy              # Cached disagreement positions
│       ├── results.json                 # Summary stats
│       ├── all_position_results.json    # All per-position results
│       ├── batch_layer=bn2/
│       │   ├── concept_vectors.npy      # (N, 2048) stacked vectors
│       │   └── metadata.json
│       └── pos_XXXX/
│           ├── position.json            # {grid, player, move_number}
│           ├── results.json             # Per-position eval results
│           ├── spine_tree.png           # Decision tree visualization
│           └── layer=bn2/
│               ├── concept_vector.npy   # (2048,) concept vector
│               ├── results.json         # Layer-specific eval results
│               └── pair_separation.png  # Projection histogram
│
├── concept_vectors_az/                  # Rollout pipeline output
│   └── YYYYMMDD_HHMMSS/
│       ├── config.yaml
│       ├── results.json
│       └── layer=conv2/
│           ├── concept_vector.npy
│           └── results.json
│
├── teachability_filter/                 # Teachability pipeline output
│   └── YYYYMMDD_HHMMSS/
│       ├── config.yaml
│       ├── results.json                 # Summary: n_concepts, n_teachable
│       └── concept_XXXX/
│           ├── concept_vector.npy
│           ├── prototype_stats.json
│           ├── student_selection.json
│           ├── benchmark.json           # 4-way results
│           └── result.json              # {is_teachable, gain, ...}
│
└── novelty_results.json                 # Novelty filter output (from notebook)
```
