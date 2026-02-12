# Teachability Pipeline (Dynamic Concepts)

This pipeline implements the teachability filter from Section 4.2.1 using shared library components.

## Entrypoints
- Pipeline: `/Users/ayushjain/Development/Research/alphazero/teachability_pipeline.py`
- Teachability core: `/Users/ayushjain/Development/Research/alphazero/src/teachability/teachability.py`

## Shared Library Reuse
- Datasets (`DATASETS`):
  - `human_games_npy` / `model_play` for position pools
  - `concept_vectors_from_runs` for concept artifact loading
- Models (`MODELS`):
  - `file` for pretrained teacher
  - `checkpoint_file` for direct student checkpoint loading
- Spine:
  - `extract_rollout_contrasts` for optimal/subpar rollout extraction from MCTS
- Hooks:
  - `get_single_activation` for per-state layer activations

## Pipeline Steps
1. Load teacher, prototype pool, random-control pool, and concept vectors from registries.
2. For each concept vector:
   - Build dynamic prototypes from rollout constraints `v·z_plus[t] >= v·z_minus_j[t]`.
   - Select student (checkpoint sweep if configured, otherwise single checkpoint).
   - Run 4-way benchmark:
     - `train_C_eval_C`
     - `train_C_eval_R`
     - `train_R_eval_C`
     - `train_R_eval_R`
   - Compute `concept_specific_gain = train_C_eval_C - train_R_eval_C`.
   - Mark teachable if gain exceeds configured margin.

## Config
Use `/Users/ayushjain/Development/Research/alphazero/configs/teachability.yaml`.

Required top-level keys:
- `positions`
- `random_positions`
- `concepts`
- `teacher_model`
- `student_model`
- `dynamic_prototypes`
- `teachability`
- `student_selection`

## Outputs
For each concept under the run directory:
- `concept_vector.npy`
- `prototype_stats.json`
- `prototype_meta_sample.json`
- `student_selection.json`
- `benchmark.json`
- `result.json`

Run-level:
- `results.json`
- `config.yaml`
