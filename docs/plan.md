# Task Breakdown: AlphaZero Concept Discovery MVP

## Context

The project has a working pipeline for discovering dynamic concept vectors from AlphaZero's MCTS decisions (3,400+ vectors already extracted at layer `bn2`), a novelty filter, and a teachability filter. The goal now is to **validate** that these concepts are real and useful, **speed up** the pipeline to get more concepts, and **build an MVP**: one novel teachable concept + a set of puzzles to teach it to humans.

The key risk is that the discovered concepts may not be meaningful. Validation (amplification, manual inspection) is the highest priority.

---

## Task List

### Task 1: Implement Concept Amplification Experiment
**Priority:** CRITICAL -- validates whether concepts are real
**Complexity:** Medium (2-3 days)
**Dependencies:** None
**Assignable independently:** Yes

**Goal:** Modify AlphaZero's activations at inference time to test whether amplifying a concept vector changes its move selection.

**What to build:**
- New module `src/amplification/amplify.py` with:
  - `amplify_activation(z, v, alpha, beta=0.01)` -- implements z' = (1-alpha)*z + alpha*beta*(||z||/||v||)*v
  - A PyTorch forward hook that intercepts the `bn2` layer output and applies amplification during MCTS evaluation
  - `run_amplification_experiment(net, concept_vector, layer, positions, alpha_range, n_sims)` -- for each position and alpha, run MCTS with/without amplification, compare top-1 moves and value estimates
- Script `amplification_pipeline.py` that runs this on N concept vectors across M positions

**Key integration points:**
- Hook pattern: `src/hooks/extract.py:46-49` (read hooks exist; need a write hook that returns modified tensor)
- MCTS evaluation: `alphazero/mcts.py:182` (`nn_evaluation` calls `self._nn.evaluate(board)`)
- Player: `alphazero/players.py:198` (AlphaZeroPlayer)

**Acceptance criteria:**
- For at least some concept vectors, amplification (alpha > 0) measurably changes moves
- Negative amplification (alpha < 0) causes opposite shift
- Report: move change rate vs alpha, value shift vs alpha, across 50+ positions and 5+ concepts

---

### Task 2: Fix Teachability Pipeline Bug + Smoke Test
**Priority:** HIGH
**Complexity:** Low (0.5-1 day)
**Dependencies:** None
**Assignable independently:** Yes

**What to fix:**
- `src/teachability/teachability.py:80` has `enumerate[dict](positions)` -- should be `enumerate(positions)`. This crashes `dynamic_prototypes_for_concept()` at runtime.
- `spine_pipeline.py:411` has a Path/string comparison bug (compares Path object to `os.listdir` strings -- always True)
- `spine_pipeline.py:368-369` has hardcoded offset `[3400:]` -- make configurable via config

**Then validate:**
1. Run existing tests: `python -m pytest tests/ -v`
2. Run teachability pipeline on 2-3 concept vectors from `runs/concept_vectors/BasicVectors/` using `configs/teachability.yaml`
3. Verify it completes end-to-end

**Key files:** `src/teachability/teachability.py`, `teachability_pipeline.py`, `spine_pipeline.py`, `configs/teachability.yaml`

---

### Task 3: Parallelize Spine Pipeline
**Priority:** HIGH -- 38hr sequential runtime is a bottleneck
**Complexity:** Medium-High (3-4 days)
**Dependencies:** None
**Assignable independently:** Yes

**Three levels of speedup:**

**Level 1 (highest impact): Multiprocess across positions**
- Each position is independent (separate MCTS tree, board state)
- Create `process_single_position()` encapsulating lines 371-497 of `spine_pipeline.py`
- Use `multiprocessing.Pool` (pattern exists in `alphazero/arena.py:187-234`)
- Challenge: PyTorch models need loading per worker (use `torch.multiprocessing` with `spawn`)
- Add `n_workers` config field

**Level 2 (medium impact): Batch activation extraction**
- Currently `build_spine_tree()` (`src/spine/spine.py:71`) calls `get_single_activation()` per node per layer (30+ forward passes per tree)
- Collect all board states first, then one call to `extract_features_by_layer()` for all positions and all layers in a single forward pass
- The TODO comment already exists at `spine_pipeline.py:403`

**Level 3 (optional): Reduce MCTS budget**
- Two-pass approach: fewer sims to triage, full sims only on promising positions

**Target:** 4x+ speedup on 8-core machine

**Key files:** `spine_pipeline.py`, `src/spine/spine.py`, `alphazero/arena.py` (parallel pattern), `src/hooks/extract.py` (`extract_features_by_layer`)

---

### Task 4: Rank All Existing Concept Vectors by Novelty + Quality
**Priority:** HIGH -- identifies which concepts to invest in
**Complexity:** Low-Medium (1-2 days)
**Dependencies:** None
**Assignable independently:** Yes

**What to build:**
A script `novelty_ranking.py` that:
1. Loads all 3,400 concept vectors from `runs/concept_vectors/BasicVectors/pos_*/layer=bn2/concept_vector.npy`
2. Loads novelty baselines from `configs/novelty_V_az.npy` and `configs/novelty_V_human.npy`
3. Runs `NoveltyFilter.check(v)` on each (`src/novelty/novelty.py:109`)
4. Also loads quality metrics from each position's `results.json` (constraint satisfaction, mean margin, sparsity, nontriviality)
5. Produces a ranked CSV/JSON combining: novelty scores, quality metrics, position metadata

**Filter criteria for "interesting" concepts:**
- `test_constraint_satisfaction > 0.6`
- `is_trivial == false`
- Novelty score positive at multiple k values

**Investigation needed:** Existing novelty scores are very small (1e-4 range). Determine if this means concepts aren't novel, or if normalization is needed.

**Output:** Top 20-50 most promising concepts ranked by composite score

**Key files:** `src/novelty/novelty.py`, `runs/concept_vectors/BasicVectors/`, `configs/novelty_V_*.npy`

---

### Task 5: Build Prototype Selection + Visual Inspection Tool
**Priority:** HIGH
**Complexity:** Medium (2-3 days)
**Dependencies:** Task 4 (ranked list), Task 2 (bug fix for `dynamic_prototypes_for_concept`)
**Assignable independently:** Yes (can start with simpler scoring approach)

**What to build:**
A notebook or script that for each top-N concept:
1. Loads concept vector + original position + spine tree visualization from `runs/`
2. Scores a pool of diverse positions by projecting activations onto the concept: `score = v . z` using `get_single_activation()` from `src/hooks/extract.py:77`
3. Selects top-K (prototypes) and bottom-K (anti-prototypes) positions
4. Creates a visual panel per concept:
   - Original position + spine tree
   - Top 5 prototype boards (highest concept activation) with scores + best moves
   - Bottom 5 boards for contrast

**Simpler fallback** (if Task 2 not done yet): Skip `dynamic_prototypes_for_concept()` and just score positions directly with `v . z`.

**Key files:** `viz_board.py`, `src/hooks/extract.py` (`get_single_activation`), `src/teachability/teachability.py` (`dynamic_prototypes_for_concept`)

---

### Task 6: Validate Teachability Pipeline on Top Concepts
**Priority:** MEDIUM-HIGH
**Complexity:** Medium (2-3 days, mostly compute)
**Dependencies:** Task 2 (bug fix), Task 4 (ranked concepts)
**Assignable independently:** Yes

**What to do:**
1. Populate `configs/teachability.yaml` with checkpoint paths (10 checkpoints available at `models/alphazero-othello-8/checkpoints/`)
2. Run `teachability_pipeline.py` on top 10-20 concepts from Task 4's ranking
3. Analyze: which concepts pass `is_teachable` (gain > 0.05)? Distribution of gains?
4. Cross-reference: are novel concepts more or less teachable?

**Budget:** ~30 min/concept (200 MCTS sims x ~100 positions per concept)

**Key files:** `teachability_pipeline.py`, `configs/teachability.yaml`, `src/teachability/teachability.py`

---

### Task 7: Multi-Layer Concept Extraction
**Priority:** MEDIUM
**Complexity:** Low-Medium (1-2 days)
**Dependencies:** Task 3 (speed, so re-running is feasible)
**Assignable independently:** Yes

**What to do:**
Current spine pipeline only extracts at `bn2`. The network has `bn1`, `bn2`, `bn3`, `bn4`, `fc_bn1`, `fc_bn2` with different dimensions (2048, 2048, 1152, 512, etc.).

1. Update `configs/spine.yaml` hooks to include multiple layers
2. The pipeline already loops over `layer_names` (spine_pipeline.py:400) -- just needs config change + Task 3's batching
3. Run on existing "interesting" positions
4. Feed new vectors through novelty ranking (Task 4)

**Key files:** `configs/spine.yaml`, `alphazero/games/othello.py` (network architecture)

---

### Task 8: Hand-Check Top Concepts for Human Teachability
**Priority:** MEDIUM-HIGH -- gates the MVP
**Complexity:** Low (1-2 days, mostly manual)
**Dependencies:** Task 4 (ranking), Task 5 (visual tool)
**Assignable independently:** Yes

**What to do:** Manual evaluation of top 10-20 concepts using the visual tool from Task 5.

**Assessment criteria per concept:**
1. **Consistency** -- Do prototypes share a visible pattern?
2. **Contrast** -- Do anti-prototypes visibly lack it?
3. **Strategic relevance** -- Does it relate to known Othello strategy?
4. **Novelty** -- Would a club player already know this?
5. **Teachability** -- Can you write a 1-sentence instruction?
6. **Amplification effect** -- Does amplifying produce consistent move changes? (from Task 1)

**Output:** Markdown/spreadsheet with concept ID, description attempt, teachability rating (1-5), MVP candidate flag

---

### Task 9: Build Puzzle Board Renderer with Annotations
**Priority:** MEDIUM
**Complexity:** Medium (2-3 days)
**Dependencies:** Task 8 (selected concepts)
**Assignable independently:** Yes (can start designing API immediately)

**What to build:** `src/visualization/puzzle_board.py` with:
- `render_puzzle(grid, player, concept_name, hint_text, arrows, highlights, move_annotations)` -- publication-quality board image
- Highlighted squares (colored overlays), move arrows, text annotations
- Before/after paired views (concept-aligned move vs common mistake)
- `PuzzleSet` class: concept metadata + ordered list of puzzle positions

**Reference code:**
- `spine_pipeline.py:44-295` -- sophisticated board rendering with matplotlib (pieces, grid, annotations, metadata panels)
- `viz_board.py` -- simpler baseline renderer

**Output format:** PNG per puzzle, multi-page PDF or HTML for the set

---

### Task 10: Assemble MVP -- One Concept + Teaching Puzzle Set
**Priority:** MEDIUM (final deliverable)
**Complexity:** Medium (2-3 days)
**Dependencies:** Tasks 1, 8, 9
**Assignable independently:** No (integration task)

**What to do:**
1. Pick the best concept from Task 8's evaluation
2. Generate 10-15 candidate puzzles (high prototype score from Task 5)
3. For each puzzle:
   - Normal MCTS -> "correct" move
   - Amplified MCTS (Task 1) -> verify concept pushes toward correct move
   - Dampened MCTS (negative alpha) -> the "common mistake"
4. Curate best 5-10 by clarity
5. Write concept explanation (1-2 sentences)
6. Render with Task 9's puzzle renderer

**Deliverable structure:**
```
Concept: [Name]
Description: [1-2 sentences]

Puzzle 1:
  [Board with annotations]
  "In this position, [concept]. Best move is X because [reason]."
  Common mistake: Y -- ignores [concept].
```

---

## Dependency Graph

```
Task 1 (Amplification) --------\
Task 2 (Bug fix) -> Task 6 -----> Task 8 (Hand-check) -> Task 10 (MVP)
Task 3 (Parallelize) -> Task 7    Task 9 (Renderer) ----/
Task 4 (Ranking) -> Task 5 ------/
```

## Suggested Team Assignments (6 people)

| Person | Immediate Task | Then |
|--------|---------------|------|
| A | Task 1 (Amplification) | Task 10 (MVP assembly) |
| B | Task 2 (Bug fix) | Task 6 (Teachability validation) |
| C | Task 3 (Parallelization) | Task 7 (Multi-layer) |
| D | Task 4 (Novelty ranking) | Task 5 (Prototypes + visual tool) |
| E | Task 9 (Puzzle renderer) | Task 10 (MVP assembly) |
| F | Task 8 (Hand-check -- starts day 3-4 when Tasks 4+5 deliver) | Task 10 (curation) |

**Critical path:** Task 1 + Task 4 -> Task 5 -> Task 8 -> Task 10

## Known Bugs to Fix (part of Task 2)

1. `src/teachability/teachability.py:80` -- `enumerate[dict]` should be `enumerate`
2. `spine_pipeline.py:411` -- Path vs string comparison (always True)
3. `spine_pipeline.py:368-369` -- Hardcoded `[3400:]` offset

## Verification

- **Task 1:** Run amplification on 5 known concept vectors, confirm move changes at alpha=0.5
- **Tasks 2+6:** Run `python -m pytest tests/ -v`, then `python teachability_pipeline.py --config configs/teachability.yaml`
- **Task 3:** Compare sequential vs parallel output on 10 positions, verify identical results
- **Task 4:** Spot-check top 5 and bottom 5 ranked concepts visually
- **Task 10:** Show puzzle set to someone unfamiliar with the concept, see if they can learn it
