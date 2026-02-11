# Implementation Guide: Bridging the Human–AI Knowledge Gap

## Paper: Concept Discovery and Transfer in AlphaZero

This guide breaks down the methodology for discovering and transferring AI concepts to humans, optimized for team collaboration.

---

## Table of Contents

1. [Methodology Overview](#methodology-overview)
2. [Team Task Breakdown](#team-task-breakdown)
3. [Phase 1: Concept Discovery](#phase-1-concept-discovery)
4. [Phase 2: Concept Filtering](#phase-2-concept-filtering)
5. [Phase 3: Method Evaluation](#phase-3-method-evaluation)
6. [Phase 4: Human Evaluation](#phase-4-human-evaluation)
7. [Key Parameters & Datasets](#key-parameters--datasets)

---

## Methodology Overview

### Core Concept
Extract novel chess concepts from AlphaZero (AZ) that exist in machine knowledge space (M) but not in human knowledge space (H). Prove these concepts are teachable to human experts.

### Pipeline
```
AlphaZero Games → Discover Concepts → Filter (Teachability) → Filter (Novelty) → Validate → Human Study
```

### Key Results
- From ~120 candidate concepts → 2-3 final concepts per layer
- 97.6% filtered by teachability, 27.1% by novelty
- All 4 grandmasters improved 6-42% after learning concepts

---

## Team Task Breakdown

### Team Structure (6 People)

**Person 1: Data Pipeline Lead**
- Extract and manage datasets
- Prepare latent representations from AlphaZero
- Handle data storage and access

**Person 2: Concept Discovery - Static**
- Implement convex optimization for static concepts
- Generate concept candidates from position features
- Focus on layers 19, 23

**Person 3: Concept Discovery - Dynamic**
- Implement MCTS rollout analysis
- Generate concept candidates from move sequences
- Handle temporal concept constraints

**Person 4: Teachability Filter**
- Implement student network training
- Run teachability experiments
- Compare concept vs. random baselines

**Person 5: Novelty Filter & Evaluation**
- Implement spectral analysis (SVD)
- Compute novelty scores
- Run algorithmic evaluations (accuracy, amplification)

**Person 6: Human Study & Prototype Selection**
- Filter prototypes for human evaluation
- Design and conduct grandmaster study
- Analyze qualitative feedback

### Parallel vs Sequential Tasks

**Parallel (Can Start Immediately After Data Ready):**
- Person 2 & 3: Concept discovery (different types)
- Person 5: Basis vector computation for novelty filter

**Sequential:**
1. Data Pipeline (Person 1)
2. Concept Discovery (Person 2 & 3)
3. Teachability Filter (Person 4)
4. Novelty Filter (Person 5)
5. Evaluation & Human Study (Person 5 & 6)

### Critical Path
Data → Concepts → Teachability → Novelty → Human Study (estimated: 6-8 weeks)

---

## Phase 1: Concept Discovery

**Goal:** Generate ~120 candidate concept vectors per layer using convex optimization

**Inputs:**
- AlphaZero model with access to layers 18, 19, 20, 21, 23
- Complex chess positions (where two AZ versions disagree)
- MCTS capability for dynamic concepts

### Task 1.1: Prepare Complex Positions
**Owner:** Person 1

**What to do:**
- Run two AlphaZero versions (differing by 75 Elo) on all games
- Select positions where they disagree on best move
- This ensures positions are complex enough to contain interesting concepts

**Output:** Dataset of complex positions (~1,000-5,000 positions)

### Task 1.2: Extract Latent Representations
**Owner:** Person 1

**What to do:**
- For each position, run forward pass through AZ network
- Extract and save latent representations at layers: 18, 19, 20, 21, 23
- Each representation is a vector (e.g., 256 dimensions for layer 19)

**Output:** Matrices Z_l where rows = positions, columns = latent dimensions

### Task 1.3: Static Concept Discovery
**Owner:** Person 2

**What this means:**
- Static concepts are properties of individual positions (e.g., "white has bishop pair", "pawn structure is weak")

**What to do:**
- For each concept hypothesis, create positive (Z⁺) and negative (Z⁻) example sets
- Formulate convex optimization: find vector v such that v·z⁺ ≥ 1 and v·z⁻ ≤ -1
- Solve using CVXPY with constraints: ||v|| ≤ 1
- Generate multiple candidates by:
  - Different random subsets of data
  - Different initialization seeds
  - Different layers

**Key equation:** Maximize margin between v·Z⁺ and v·Z⁻

**Output:** ~50-60 static concept vectors per layer

### Task 1.4: Dynamic Concept Discovery
**Owner:** Person 3

**What this means:**
- Dynamic concepts are about sequences of moves (plans, strategies)

**What to do:**
- For each position, run MCTS (800 simulations)
- Extract two rollouts:
  - X⁺: Best continuation (principal variation)
  - X⁻: Subpar continuation (lower-ranked path)
- Get latent representations for all positions in both rollouts: Z⁺ = [z₀⁺, z₁⁺, ..., z_T⁺]
- Formulate convex optimization: find v such that v·z_t⁺ ≥ v·z_t⁻ for all timesteps t
- This ensures the concept "prefers" the better plan throughout

**Key equation:** For all positions i and timesteps t: v·z_{i,t}⁺ ≥ v·z_{i,t}⁻

**Output:** ~50-60 dynamic concept vectors per layer

### Deliverables After Phase 1
- **Person 1:** Latent representation matrices for all layers
- **Person 2:** ~50-60 static concept vectors per layer
- **Person 3:** ~50-60 dynamic concept vectors per layer
- **Total:** ~120 concept candidates per layer to filter

---

## Phase 2: Concept Filtering

**Goal:** Filter ~120 candidates down to 2-3 high-quality, novel concepts

### Filter 2.1: Teachability (Person 4)

**What this means:**
- A concept is "teachable" if showing examples helps an AI agent (or human) learn it
- Test by training a "student" network on concept prototypes

**The Process:**

**Step 1: Select Prototypes**
- For each concept vector v, compute score for all positions: score = v·z
- Select top 2.5% positions (highest scores) as "prototypes"
- These are positions that best exemplify the concept

**Step 2: Find Student Network**
- Use early AlphaZero checkpoint as "student"
- Requirement: student's policy agreement with teacher (full AZ) < 20%
- This ensures student doesn't already know the concept

**Step 3: Train Student**
- Split prototypes: 80% train, 20% test
- Train student by minimizing KL divergence between student and teacher policies
- Only on training prototypes, for 50 epochs

**Step 4: Evaluate Learning**
- Measure: How often does student pick same move as teacher on test set?
- **Key metric:** T = (# agreements on test set) / (# test positions)

**Step 5: Compare to Baseline**
- Also train student on random AZ positions (same number)
- Accept concept if: (concept-trained accuracy) - (random-trained accuracy) > threshold

**Key insight:** If concept is real and teachable, training on concept prototypes should help the student learn faster than training on random positions

**Outcome:** Filters out 97.6% of concepts (keeps ~3 concepts)

### Filter 2.2: Novelty (Person 5)

**Goal:** Ensure concepts are unique to AZ, not already in human games

**The Process:**

**Step 1: Compute Basis Vectors**
- Create matrices:
  - Z_human: 17,184 positions from human games × layer dimensions
  - Z_az: 17,184 positions from AZ games × layer dimensions
- Compute SVD (Singular Value Decomposition):
  - Z_human = U_h Σ_h V_h^T
  - Z_az = U_a Σ_a V_a^T
- U_h and U_a are the "basis vectors" - think of them as the fundamental building blocks of each game space

**Step 2: Check if AZ Space is Larger**
- Count non-zero singular values (the "rank")
- Expected results (from paper):
  - Layer 19: Human rank = 7,857, AZ rank = 8,269
  - Layer 23: Human rank = 6,544, AZ rank = 6,771
- Higher rank for AZ suggests it has more concepts

**Step 3: Compute Novelty Score for Each Concept**
For concept vector v:
- Reconstruct v using human basis: how well can we express v using U_h?
- Reconstruct v using AZ basis: how well can we express v using U_a?
- Novelty score = (error with human basis) - (error with AZ basis)

**Mathematical form:**
- novelty_score = min ||v - Σβᵢuᵢʰ||² - min ||v - Σγᵢuᵢᵃ||²
- If novelty_score > 0: concept aligns better with AZ than humans → Novel!

**Step 4: Filter Concepts**
- Test multiple values of k (number of basis vectors: 10, 20, 50, 100, ...)
- Accept concept only if novelty_score > 0 for ALL k values tested
- This is a strict criterion

**Outcome:** Filters out additional 27.1% of concepts

### Deliverables After Phase 2
- **Person 4:** Teachability scores for all concepts, filtered list
- **Person 5:** Novelty scores for all concepts, final filtered list
- **Combined:** 1-2 final concepts per layer ready for evaluation

---

## Phase 3: Method Evaluation

**Goal:** Validate that concept vectors actually capture meaningful concepts

**Owner:** Person 5 (can be done in parallel with Phase 2)

### Evaluation 3.1: Test Set Accuracy

**What to do:**
- Split each dataset into 80% train, 20% test
- Measure: On test set, how often does v·z⁺ ≥ v·z⁻?
- Expected results:
  - Pieces dataset: 99% accuracy
  - Stockfish: 76%
  - STS (Strategic Test Suite): 92%
  - Openings: 99-100%

**Why this matters:** Shows the concept vector generalizes beyond training data

### Evaluation 3.2: Sample Efficiency

**What to do:**
- Learn concepts with varying training set sizes: 10, 20, 50, 100, 200 examples
- Repeat 10 times with different random seeds
- Plot: training size vs. test accuracy

**Expected result:** Should reach near-full accuracy with just 10-20 examples

**Why this matters:** Shows concepts can be learned data-efficiently

### Evaluation 3.3: Concept Amplification

**What to do:**
- For puzzles from STS dataset, artificially "amplify" the concept in latent space
- Formula: z̃ = (1-α)z + αβ(||z||/||v||)v where α ∈ [0,1], β=0.01
- Measure: Does amplifying concept increase AZ's performance on concept-related puzzles?

**Expected result:** 
- Higher α → better puzzle-solving performance
- Higher quality concepts → larger improvement

**Why this matters:** Shows the concept vector actually influences AZ's decisions

### Deliverables After Phase 3
- Test accuracy table for all datasets and layers
- Sample efficiency curves
- Amplification performance curves
- Validation that final concepts are high-quality

---

## Phase 4: Human Evaluation

**Goal:** Prove concepts are teachable to human grandmasters

**Owner:** Person 6

### Study Design: 3 Phases

**Participants:** 4 grandmasters (Elo 2600-2800)

**Phase 1: Baseline (Pre-Learning)**
- Show GM 4 puzzles per concept, 3-4 concepts total (12-16 puzzles)
- Ask: "What's the best move?"
- Record their answer
- Measure: % correct according to AZ's solution

**Phase 2: Learning**
- Show same puzzles again, but now with AZ's solution (top moves + variation)
- GM studies AZ's reasoning
- No measurement, just learning

**Phase 3: Test (Post-Learning)**
- Show 4 NEW puzzles from same concepts (unseen positions)
- Ask: "What's the best move?"
- Record their answer
- Measure: % correct

**Key metric:** Improvement = (Phase 3 accuracy) - (Phase 1 accuracy)

### Task 4.1: Select Prototypes for Humans

**Additional filtering criteria:**
- Not trivial tactics (avoid "just win the queen")
- Not endgame tablebase positions (avoid memorization)
- Complex enough to require understanding concept
- AZ's line has depth ≥ 3 moves
- Position demonstrates concept clearly

**Why this matters:** Humans need high-quality, instructive puzzles

### Task 4.2: Collect Qualitative Feedback

**What to collect:**
- GM's thought process (free-form comments)
- Which moves they considered
- How they describe the concept
- Whether they found it novel/interesting

**Why this matters:** 
- Understand what makes concepts teachable
- Identify differences between human and AI thinking
- Validate concepts are actually novel to humans

### Expected Results

From paper:
- **GM 1:** 0% → 42% (+42% improvement)
- **GM 2:** 33% → 58% (+25%)
- **GM 3:** 25% → 42% (+16%)
- **GM 4:** 38% → 44% (+6%)

**All participants improved**, validating teachability

### Deliverables After Phase 4
- Baseline and final accuracy for each GM
- Improvement statistics
- Qualitative analysis of GM feedback
- Examples of successful and unsuccessful learning
- Insights into human-AI differences

---

## Key Parameters & Datasets

### Critical Hyperparameters

| Parameter | Value | Where Used |
|-----------|-------|------------|
| MCTS simulations | 800 | Dynamic concept discovery |
| Prototype percentile | Top 2.5% | Selecting concept examples |
| Student overlap threshold | < 0.2 | Finding suitable student network |
| Training epochs | 50 | Teaching student network |
| Train/test split | 80/20 | All evaluations |
| Amplification β | 0.01 | Concept amplification tests |
| Elo difference | 75 points | Selecting complex positions |

### Layers to Analyze

Focus on these layers (dimensions in brackets):
- **Layer 18-19**: Before policy/value split [8×8×256] → Most concepts found here
- **Layer 20**: Value head [64]
- **Layer 21**: Value head [256]
- **Layer 23**: Policy head [8×8×256] → Direct impact on moves

### Datasets

#### 1. AlphaZero Self-Play Games
- **Size:** 17,184 positions
- **Use:** Source for discovering novel concepts
- **Characteristics:** Super-human play, complex positions

#### 2. Human Games (ChessBase Mega Database)
- **Size:** 17,184 positions  
- **Use:** Novelty filtering (compare AZ vs human space)
- **Characteristics:** Top-level human games

#### 3. Evaluation Datasets

**Pieces Dataset**
- Concept: Piece presence/absence
- Use: Validate method on known concepts
- Expected accuracy: 99%

**Stockfish Dataset**
- Concepts: Tactical motifs
- Use: Validate on engine-known concepts
- Expected accuracy: 76%

**Strategic Test Suite (STS)**
- Concepts: 100 puzzles per strategic theme
- Themes: Square vacancy, knight outposts, bishop pairs, etc.
- Expected accuracy: 92%
- Use: Concept amplification tests

**Opening Dataset**
- Concepts: Opening theory positions
- Two types: General openings, specific lines
- Expected accuracy: 99-100%

---

## Timeline & Dependencies

### Week 1-2: Data Preparation (Person 1)
- Extract all datasets
- Generate latent representations for all layers
- Create complexity-filtered position set
- **Blocker for:** All other tasks

### Week 2-4: Concept Discovery (Persons 2 & 3, parallel)
- Person 2: Static concepts (~60 per layer)
- Person 3: Dynamic concepts (~60 per layer)
- **Depends on:** Data preparation complete
- **Blocker for:** Filtering phases

### Week 3-4: Basis Computation (Person 5, parallel)
- Compute SVD for human and AZ game spaces
- **Depends on:** Data preparation
- **Blocker for:** Novelty filtering

### Week 4-6: Teachability Filter (Person 4)
- Select prototypes for all concepts
- Train and evaluate student networks
- **Depends on:** Concept discovery complete
- **Blocker for:** Novelty filtering

### Week 6-7: Novelty Filter (Person 5)
- Compute novelty scores
- Filter final concepts
- **Depends on:** Teachability filter + basis computation
- **Blocker for:** Human evaluation

### Week 7-8: Evaluation (Person 5 & 6, parallel)
- Person 5: Algorithmic validation
- Person 6: Prepare human study materials
- **Depends on:** Final concepts selected

### Week 8-12: Human Study (Person 6)
- Recruit grandmasters
- Conduct 3-phase study (can overlap participants)
- Analyze results
- **Depends on:** Evaluation complete

**Total estimated time:** 10-12 weeks

---

## Success Criteria

### Algorithmic Validation
✓ Concept vectors achieve >75% test accuracy  
✓ Sample efficiency: <20 examples needed  
✓ Amplification improves performance  
✓ Teachability: concept learning > random baseline

### Novelty Validation
✓ AZ game space rank > human game space rank  
✓ Novelty score > 0 for all k values tested  
✓ Concepts align better with AZ basis than human basis

### Human Validation
✓ All GMs show improvement (Phase 3 > Phase 1)  
✓ Average improvement >10%  
✓ GMs describe concepts as novel/interesting  
✓ Concepts are applicable to new positions

---

## Key Insights from Paper

### What Makes Concepts Teachable
- Clear prototypes that exemplify the concept
- Concepts that generalize across positions
- Not too complex (within human capability)
- Observable improvement from examples

### Human-AI Differences Found
1. **Computational capacity**: AZ can calculate deeper, takes more risks
2. **Motivation**: AZ optimizes for expected value; humans play practically
3. **Flexibility**: AZ more willing to change plans
4. **Style**: AZ prefers slower positional play over forcing tactics

### Common Concept Types Discovered
- Prophylactic moves (restricting opponent options)
- Strategic queen sacrifices
- Unconventional piece maneuvers
- Positional tactics combining strategy and tactics
- Controlling key squares through indirect means

---

## References

**Primary Paper:**  
Schut, L., Tomašev, N., McGrath, T., Hassabis, D., Paquet, U., & Kim, B. (2023). *Bridging the Human–AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero.* arXiv:2310.16410

**Key Dependencies:**
- CVXPY: Convex optimization
- AlphaZero (Silver et al., 2017, 2018)
- ChessBase Mega Database
- Strategic Test Suite (STS)
- python-chess

---

## Quick Start Checklist

**Before Starting:**
- [ ] AlphaZero model accessible with layer outputs
- [ ] Two AZ versions (75 Elo apart) available
- [ ] ChessBase database or equivalent human games
- [ ] CVXPY installed and tested
- [ ] MCTS implementation ready

**Phase 1 Readiness:**
- [ ] Can extract latent representations efficiently
- [ ] Storage for ~17K positions × 5 layers
- [ ] Batch processing pipeline working

**Phase 2-3 Readiness:**
- [ ] Early AZ checkpoints available for student network
- [ ] GPU access for network training
- [ ] Evaluation datasets loaded

**Phase 4 Readiness:**
- [ ] 4 grandmasters recruited
- [ ] Puzzle interface prepared
- [ ] Data collection system ready