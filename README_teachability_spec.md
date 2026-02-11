# Teachability Pipeline — Implementation Specification (§4.2.1)

> **Purpose**: This document is a complete specification for implementing the teachability filtering step from "Bridging the Human–AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero." It is written so that an agent (Claude Code, Copilot, etc.) with access to the repo but **not** the PDF can implement the full pipeline. All equations, hyperparameters, edge cases, and validation criteria are included.

---

## 1. Context & Goal

You have already extracted **dynamic concept vectors** `v_c,l` (at layer `l`) using the convex optimisation framework (§4.1.2). Many of these vectors may represent noise, non-generalisable artefacts, or already-known concepts. **Teachability** is a filter: you keep only concept vectors that can be successfully taught to a student model, proving they encode transferable knowledge.

### What You Have (Inputs)

| Ingredient | Description |
|---|---|
| `teacher_model` | The strong/fully-trained model (e.g. AlphaZero). You can call `teacher_model.policy(x)` to get the full output distribution and `teacher_model.forward(x, layer=l)` to get intermediate activations `z_l`. |
| `student_model` | A weaker model — either an earlier training checkpoint of the teacher, or a separate smaller model. Must understand the domain (e.g. can play chess) but ideally does NOT already know the concept. |
| `concept_vectors` | A list/dict of concept vectors `v_c,l`, each a 1-D numpy/torch tensor with shape `(d_l,)` where `d_l` is the hidden dimension at layer `l`. |
| `dataset_X` | A large pool of input states (e.g. 30,000 game positions sampled from teacher self-play games). |

### What You Produce (Outputs)

| Output | Description |
|---|---|
| `teachable_concepts` | The subset of `concept_vectors` that pass the teachability filter. |
| `prototypes_per_concept` | For each teachable concept, the set of prototype inputs (with train/test split). |
| `teachability_scores` | Per-concept numeric scores measuring how much the student improved. |

---

## 2. Pipeline Overview

```
For each concept vector v_c,l:
    Step A: Find prototypes (inputs exemplifying the concept)
    Step B: Select a suitable student (doesn't know the concept yet)
    Step C: Measure baseline (Phase 1)
    Step D: Teach the student on prototype train set (Phase 2)
    Step E: Evaluate on prototype test set (Phase 3)
    Step F: Benchmark against random-trained control
    Step G: Decide keep/discard based on teachability score
```

---

## 3. Step A — Finding Prototypes

### 3.1 Purpose

Prototypes are input examples from `dataset_X` where the concept is most clearly and consistently active. They serve as the **curriculum** for teaching the student.

### 3.2 Static vs Dynamic — IMPORTANT DISTINCTION

The paper describes two methods. Since you have **dynamic** concept vectors, use the dynamic method. But understand both:

**Static concepts** (for reference only — NOT your case):
```python
# Simple dot product scoring
for x_i in dataset_X:
    z_i = teacher_model.forward(x_i, layer=l)  # activation at layer l
    score_i = np.dot(v_c, z_i)

# Keep top 2.5% by score
threshold = np.percentile(all_scores, 97.5)
prototypes = [x_i for x_i, s in zip(dataset_X, all_scores) if s >= threshold]
```

### 3.3 Dynamic Concept Prototype Selection (YOUR CASE)

For dynamic concepts, a prototype must demonstrate the concept **across an entire planned trajectory**, not just at a single state. The criterion is stricter.

#### Algorithm

```python
def find_dynamic_prototypes(v_c, layer_l, dataset_X, teacher_model):
    """
    Find prototype inputs for a dynamic concept vector.
    
    Args:
        v_c: concept vector, shape (d_l,)
        layer_l: which layer the concept was found in
        dataset_X: list of input states (e.g. 30,000 game positions)
        teacher_model: the strong model with MCTS/planning capability
    
    Returns:
        prototypes: list of input states that pass the criterion
    """
    prototypes = []
    
    for x_i in dataset_X:
        # 1. Run MCTS (or your planning/search process) from state x_i
        mcts_result = teacher_model.run_mcts(x_i)
        
        # 2. Extract the OPTIMAL rollout (chosen line)
        #    This is the sequence of states AZ actually chose as best.
        #    Get activations at layer l for each time step t.
        #    z_plus[t] has shape (d_l,)
        optimal_rollout = mcts_result.get_optimal_rollout()
        z_plus = [
            teacher_model.forward(state, layer=layer_l)
            for state in optimal_rollout
        ]
        
        # 3. Extract the SUBPAR rollout (the next-best or inferior line)
        #    Defined as a path in the MCTS tree that is suboptimal according
        #    to value estimate or visit count.
        subpar_rollout = mcts_result.get_subpar_rollout()
        z_minus = [
            teacher_model.forward(state, layer=layer_l)
            for state in subpar_rollout
        ]
        
        # 4. CHECK THE CRITERION:
        #    v_c · z_plus[t] >= v_c · z_minus[t]  FOR ALL t
        #    The concept vector must score the optimal trajectory higher
        #    than the subpar trajectory at EVERY time step.
        is_prototype = all(
            np.dot(v_c, z_plus[t]) >= np.dot(v_c, z_minus[t])
            for t in range(min(len(z_plus), len(z_minus)))
        )
        
        if is_prototype:
            prototypes.append(x_i)
    
    return prototypes
```

#### Extended Version with Multiple Subpar Rollouts

The paper actually contrasts the optimal rollout against **multiple** subpar rollouts found at different MCTS depths (not just one). This is more robust:

```python
def find_dynamic_prototypes_extended(v_c, layer_l, dataset_X, teacher_model):
    """
    Extended version: contrast optimal against multiple subpar rollouts
    at different depths j. More robust, less noisy.
    """
    prototypes = []
    
    for x_i in dataset_X:
        mcts_result = teacher_model.run_mcts(x_i)
        
        optimal_rollout = mcts_result.get_optimal_rollout()
        T = len(optimal_rollout)
        z_plus = [
            teacher_model.forward(state, layer=layer_l)
            for state in optimal_rollout
        ]
        
        # T_tilde = T - 5 (ensure rollout is sufficiently deep)
        T_tilde = max(1, T - 5)
        
        # Get subpar rollouts branching at different depths j
        is_prototype = True
        for j in range(T_tilde):
            subpar_rollout_j = mcts_result.get_subpar_rollout_at_depth(j)
            z_minus_j = [
                teacher_model.forward(state, layer=layer_l)
                for state in subpar_rollout_j
            ]
            
            # Must hold for ALL t, ALL j
            for t in range(min(len(z_plus), len(z_minus_j))):
                if np.dot(v_c, z_plus[t]) < np.dot(v_c, z_minus_j[t]):
                    is_prototype = False
                    break
            if not is_prototype:
                break
        
        if is_prototype:
            prototypes.append(x_i)
    
    return prototypes
```

### 3.4 Key Details & Pitfalls for Prototypes

| Detail | Value / Note |
|---|---|
| Dataset size | Paper uses 30,000 input states sampled from teacher self-play games |
| Expected yield | Roughly top 2.5% pass (so ~750 prototypes from 30K). Dynamic criterion may yield fewer since it's stricter. |
| Subpar rollout definition | A path in the MCTS tree that is suboptimal by value estimate OR visit count. The paper requires a **minimum value difference of 0.20** and/or a **visit count difference of 10%** of the most-visited move. |
| Rollout depth T | Paper uses T=5 or T=10. Set T_tilde = T - 5 for subpar branching depth. |
| Single vs Both players | If rollout alternates between players (e.g. chess), you may want every-other timestep (`z_{2t}`) for single-player concepts, or all timesteps for both-player concepts. |
| Edge case: too few prototypes | If fewer than ~20 prototypes are found, the concept may be too narrow or the vector may be noise. Log a warning. Consider relaxing the criterion or discarding. |
| Edge case: too many prototypes | If >50% of inputs pass, the concept vector may not be selective enough. Investigate. |

---

## 4. Step B — Selecting the Student Model

### 4.1 Goal

Find a model that **understands the domain** (can play competently) but **does not already know this specific concept**. If the student already knows the concept, the teach-then-evaluate loop won't show improvement and you get a false negative.

### 4.2 Method: Checkpoint Selection

```python
def select_student(checkpoints, prototypes, teacher_model, threshold=0.2):
    """
    Select the latest (strongest) checkpoint where concept knowledge is low.
    
    Args:
        checkpoints: list of model checkpoints, ordered from earliest to latest
        prototypes: the prototype inputs found in Step A
        teacher_model: the fully trained teacher
        threshold: max allowable policy overlap (paper uses 0.2)
    
    Returns:
        student_model: the selected checkpoint
    """
    # Iterate from latest (strongest) to earliest
    for ckpt in reversed(checkpoints):
        # Measure how often student agrees with teacher on best action
        agreements = 0
        for x_i in prototypes:
            student_action = np.argmax(ckpt.policy(x_i))
            teacher_action = np.argmax(teacher_model.policy(x_i))
            if student_action == teacher_action:
                agreements += 1
        
        overlap = agreements / len(prototypes)
        
        if overlap < threshold:
            return ckpt
    
    # If no checkpoint is below threshold, use the earliest one
    # and log a warning — concept may already be learned very early
    print("WARNING: No checkpoint found with overlap < threshold. Using earliest.")
    return checkpoints[0]
```

### 4.3 Key Details & Pitfalls

| Detail | Value / Note |
|---|---|
| Overlap threshold | **0.2** (20%). The student should agree with the teacher on fewer than 20% of prototype positions. |
| Why "latest below threshold" | You want the strongest possible student that still doesn't know the concept. Too weak = can't learn anything. Too strong = already knows it. |
| If you only have 2 models | Just measure the overlap between your weak model and strong model on the prototypes. If <0.2, proceed. If ≥0.2, the weak model may already know the concept — the teachability test may undercount improvement. |
| policy(x) returns | The **full output distribution** (softmax over actions), not just the top action. You only use argmax for the overlap check. |
| Prototype set used | Use ALL prototypes (before train/test split) for student selection. |

---

## 5. Step C — Measure Baseline (Phase 1)

Before teaching, measure the student's performance on concept prototypes.

```python
def measure_agreement(student_model, teacher_model, eval_set):
    """
    Equation 6 from the paper:
    T = sum over x_i in eval_set of:
        1[argmax(pi_student(x_i)) == argmax(pi_teacher(x_i))]
    
    Returns both raw count and normalized (proportion).
    """
    agreements = sum(
        1 for x_i in eval_set
        if np.argmax(student_model.policy(x_i)) == np.argmax(teacher_model.policy(x_i))
    )
    return agreements, agreements / len(eval_set)
```

Split prototypes first:
```python
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(prototypes, test_size=0.2, random_state=42)

T_before_raw, T_before_norm = measure_agreement(student_model, teacher_model, X_test)
```

---

## 6. Step D — Teach the Student (Phase 2)

### 6.1 Training Loop

Train the student to mimic the teacher's policy **only on the training prototypes**.

```python
import torch
import torch.nn.functional as F

def teach_student(student_model, teacher_model, X_train, epochs=5, lr=1e-4):
    """
    Train student via KL divergence on prototype curriculum.
    
    Loss = sum over x_i in X_train of:
        KL[ pi_teacher(x_i) || pi_student(x_i) ]
    
    IMPORTANT: KL divergence direction is KL[teacher || student].
    This means the teacher distribution is the "target" (first arg).
    """
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_i in X_train:
            optimizer.zero_grad()
            
            # Get teacher's policy (detached, no grad)
            with torch.no_grad():
                teacher_logits = teacher_model.policy(x_i)
                teacher_probs = F.softmax(teacher_logits, dim=-1)
            
            # Get student's policy
            student_logits = student_model.policy(x_i)
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            
            # KL(teacher || student)
            loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(X_train):.4f}")
    
    return student_model
```

### 6.2 Hyperparameters (from the paper)

| Hyperparameter | Value | Notes |
|---|---|---|
| Optimizer | Adam | Standard, nothing special |
| Learning rate | **1e-4** | |
| Epochs | **5** | Paper says 5 is "sufficiently indicative of performance if we train for longer." For multiple concepts, 5 keeps it tractable. |
| Batch size | Not specified | Paper seems to iterate over prototypes individually. Use mini-batches if you have many prototypes for efficiency. |
| KL direction | **KL[teacher ‖ student]** | Teacher is the reference distribution. This is "forward KL" / "mean-seeking". |

### 6.3 Pitfalls

- **Freeze the teacher**: Never backprop through the teacher. Its outputs are labels.
- **Clone the student before training**: Keep an untrained copy for baseline comparison. Each concept gets a fresh student (or reload the checkpoint).
- **π is the full distribution**: You're matching the entire output distribution, not just the top action. This transfers richer information.
- **Small dataset**: You're training on maybe ~600 examples (80% of ~750 prototypes). Overfitting is a risk but OK — the test set is what matters.

---

## 7. Step E — Evaluate (Phase 3)

After training, re-measure agreement on the **held-out test prototypes**:

```python
T_after_raw, T_after_norm = measure_agreement(student_model, teacher_model, X_test)

improvement = T_after_norm - T_before_norm
print(f"Baseline: {T_before_norm:.3f}, After teaching: {T_after_norm:.3f}, Δ: {improvement:.3f}")
```

If `T_after > T_before`, the student learned something from the concept-specific curriculum → the concept is **potentially teachable**.

But this alone is not enough — you need the benchmark comparison.

---

## 8. Step F — Benchmark Against Random Control

### 8.1 Why This Is Necessary

Any training data from the teacher might improve the student. You need to prove the improvement is **concept-specific**, not just general distillation.

### 8.2 The 4-Way Comparison

Run 4 experiments total (2 training conditions × 2 evaluation conditions):

```python
def run_full_teachability_benchmark(
    student_checkpoint,   # the clean checkpoint (reload for each run)
    teacher_model,
    concept_prototypes,   # X_train_concept, X_test_concept
    random_positions,     # X_train_random, X_test_random (same size)
):
    results = {}
    
    # --- Experiment 1: Train on concept, evaluate on concept ---
    student_1 = load_checkpoint(student_checkpoint)
    student_1 = teach_student(student_1, teacher_model, X_train_concept)
    _, results['train_C_eval_C'] = measure_agreement(student_1, teacher_model, X_test_concept)
    
    # --- Experiment 2: Train on concept, evaluate on random ---
    _, results['train_C_eval_R'] = measure_agreement(student_1, teacher_model, X_test_random)
    
    # --- Experiment 3: Train on random, evaluate on concept ---
    student_2 = load_checkpoint(student_checkpoint)
    student_2 = teach_student(student_2, teacher_model, X_train_random)
    _, results['train_R_eval_C'] = measure_agreement(student_2, teacher_model, X_test_concept)
    
    # --- Experiment 4: Train on random, evaluate on random ---
    _, results['train_R_eval_R'] = measure_agreement(student_2, teacher_model, X_test_random)
    
    return results
```

### 8.3 Random Positions

- Sample random positions **from the teacher's self-play games** (NOT human games)
- Reason 1: Teacher games are higher quality
- Reason 2: Avoids out-of-distribution confounding (closer to teacher's natural training data)
- Use the **same number** of random positions as concept prototypes

### 8.4 Random Concept Baseline (Optional Additional Control)

The paper also tests with a **random concept vector** (sampled from `N(0,1)`, same shape as `v_c,l`, then rescaled by `1/n` where `n` is the number of hidden units). Use this random vector to find "random prototypes" via the same prototype-finding procedure. If those random prototypes teach just as well as the real concept's prototypes, the concept vector isn't capturing anything meaningful.

```python
def make_random_concept_vector(d_l):
    """Create a random concept vector as a control."""
    v_random = np.random.randn(d_l)
    v_random = v_random / d_l  # rescale by 1/n
    return v_random
```

### 8.5 Decision Criterion

A concept is **teachable** when:

```
results['train_C_eval_C']  >>  results['train_R_eval_C']
```

In words: training on concept-specific prototypes improves concept performance **significantly more** than training on random data. The paper's Figure 4 shows the "dark blue line" (train concept, eval concept) should be well above the "light blue line" (train random, eval concept).

The paper does not specify a hard numeric threshold for "significantly more." In practice:
- Visually inspect training curves
- The paper trained for 50 epochs in their evaluation plots and saw clear separation
- For filtering, 5 epochs is enough to see if there's signal
- A reasonable heuristic: keep concepts where `train_C_eval_C - train_R_eval_C > 0.05` (but tune this to your domain)

---

## 9. Step G — Final Decision: Keep or Discard

```python
def is_teachable(results, margin=0.05):
    """
    Decide if a concept passes the teachability filter.
    
    Args:
        results: dict from run_full_teachability_benchmark
        margin: minimum improvement over random baseline
    """
    concept_specific_improvement = results['train_C_eval_C'] - results['train_R_eval_C']
    return concept_specific_improvement > margin
```

---

## 10. Full Pipeline — Putting It All Together

```python
def run_teachability_pipeline(
    concept_vectors,    # dict: {concept_id: (v_c, layer_l)}
    teacher_model,
    student_checkpoint, # path or object to reload from
    dataset_X,          # 30,000 input states
    random_positions,   # separate random sample from teacher games
    epochs=5,
    lr=1e-4,
    overlap_threshold=0.2,
    teachability_margin=0.05,
):
    teachable_concepts = {}
    all_scores = {}
    
    for concept_id, (v_c, layer_l) in concept_vectors.items():
        print(f"\n{'='*60}")
        print(f"Processing concept: {concept_id}")
        print(f"{'='*60}")
        
        # --- A. Find prototypes ---
        prototypes = find_dynamic_prototypes(v_c, layer_l, dataset_X, teacher_model)
        print(f"  Found {len(prototypes)} prototypes")
        
        if len(prototypes) < 20:
            print(f"  WARNING: Too few prototypes ({len(prototypes)}). Skipping.")
            all_scores[concept_id] = {'status': 'skipped_few_prototypes', 'n_prototypes': len(prototypes)}
            continue
        
        # --- B. Select student ---
        # (If using same student for all concepts, do this once outside the loop)
        # student_model = select_student(checkpoints, prototypes, teacher_model, overlap_threshold)
        student_model = load_checkpoint(student_checkpoint)
        
        # --- C. Baseline ---
        X_train, X_test = train_test_split(prototypes, test_size=0.2, random_state=42)
        _, T_before = measure_agreement(student_model, teacher_model, X_test)
        print(f"  Baseline agreement: {T_before:.3f}")
        
        # --- D & E. Teach and evaluate ---
        # Reload fresh student for concept training
        student_concept = load_checkpoint(student_checkpoint)
        student_concept = teach_student(student_concept, teacher_model, X_train, epochs, lr)
        _, T_after = measure_agreement(student_concept, teacher_model, X_test)
        print(f"  After teaching: {T_after:.3f}")
        
        # --- F. Random control ---
        X_train_random = random_positions[:len(X_train)]
        X_test_random = random_positions[len(X_train):len(X_train)+len(X_test)]
        
        student_random = load_checkpoint(student_checkpoint)
        student_random = teach_student(student_random, teacher_model, X_train_random, epochs, lr)
        _, T_random_on_concept = measure_agreement(student_random, teacher_model, X_test)
        _, T_random_on_random = measure_agreement(student_random, teacher_model, X_test_random)
        
        # --- G. Decide ---
        concept_specific_gain = T_after - T_random_on_concept
        
        scores = {
            'n_prototypes': len(prototypes),
            'T_before': T_before,
            'T_after_concept_train': T_after,
            'T_after_random_train_eval_concept': T_random_on_concept,
            'T_after_random_train_eval_random': T_random_on_random,
            'concept_specific_gain': concept_specific_gain,
            'is_teachable': concept_specific_gain > teachability_margin,
        }
        all_scores[concept_id] = scores
        
        if scores['is_teachable']:
            teachable_concepts[concept_id] = {
                'vector': v_c,
                'layer': layer_l,
                'prototypes_train': X_train,
                'prototypes_test': X_test,
                'scores': scores,
            }
            print(f"  ✓ TEACHABLE (gain: {concept_specific_gain:.3f})")
        else:
            print(f"  ✗ NOT teachable (gain: {concept_specific_gain:.3f})")
    
    return teachable_concepts, all_scores
```

---

## 11. Hyperparameter Summary

| Parameter | Value | Source |
|---|---|---|
| Dataset pool size | 30,000 | §8.6 |
| Prototype percentile (static) | Top 2.5% | §4.2.1 |
| Dynamic prototype criterion | `v·z⁺(t) ≥ v·z⁻(t) ∀t` | §4.2.1 |
| MCTS subpar min value diff | 0.20 | §8.4.1 |
| MCTS subpar min visit count diff | 10% of most-visited | §8.4.1 |
| Rollout depth T | 5 or 10 | §8.4.1 |
| T_tilde (subpar branching depth) | T - 5 | §4.1.2 |
| Student overlap threshold | < 0.2 (20%) | §8.6 |
| Optimizer | Adam | §8.6 |
| Learning rate | 1e-4 | §8.6 |
| Training epochs | 5 | §8.6 |
| Train/test split | 80/20 | §5.1.1, §8.6 |
| Random vector rescale | 1/n (n = hidden dim) | §8.6 |

---

## 12. Validation Checks & Sanity Tests

Implement these checks to catch bugs early:

### 12.1 Prototype sanity
- [ ] `len(prototypes)` is between 20 and `0.5 * len(dataset_X)` for each concept
- [ ] Prototype inputs are distinct (no duplicates)
- [ ] For dynamic prototypes, verify the criterion by re-running the dot product check on a sample

### 12.2 Student selection sanity
- [ ] Student overlap with teacher on prototypes is < 0.2
- [ ] Student overlap with teacher on RANDOM positions is also reasonably low (otherwise student is just weak everywhere, not concept-ignorant)
- [ ] Student can produce valid outputs (not degenerate/uniform policy)

### 12.3 Training sanity
- [ ] KL loss decreases over epochs
- [ ] Student policy changes after training (compare logits before/after on a few examples)
- [ ] Teacher policy is NOT modified (verify weights unchanged)

### 12.4 Evaluation sanity
- [ ] `T_before < T_after` for at least some concepts (if zero concepts improve, something is wrong)
- [ ] `train_C_eval_C > train_R_eval_C` for teachable concepts
- [ ] `train_C_eval_C > train_C_eval_R` (concept training helps concept eval more than random eval)
- [ ] Random-trained student should still show SOME improvement (it's learning from teacher labels after all), just less

### 12.5 Expected outcome from the paper
- After 5 epochs, teachable concepts show clear separation between concept-trained and random-trained students
- The paper found that 50 epochs of prototype training was equivalent to 10K–100K epochs of self-play, demonstrating efficiency
- After teachability filtering, they further removed 27.1% of remaining concepts via novelty filtering (§4.2.2), which is a SEPARATE subsequent step not covered in this spec

---

## 13. Common Pitfalls

| Pitfall | Explanation | Fix |
|---|---|---|
| Using `KL[student ‖ teacher]` instead of `KL[teacher ‖ student]` | Wrong direction. The paper minimizes KL with teacher as the reference (first argument in the mathematical notation). | Double-check: `F.kl_div(student_log_probs, teacher_probs)` in PyTorch. Note PyTorch's `kl_div` expects log-probs as input and probs as target. |
| Not reloading student checkpoint per concept | If you train on concept A then concept B, the student already knows A. Each concept needs a fresh student. | Reload from checkpoint for each concept. |
| Evaluating on train set instead of test set | Overfitting masquerades as teachability. | Always evaluate on held-out `X_test`. |
| Too few prototypes | Noisy results, unreliable signal. | Require minimum ~20 prototypes. |
| Random positions from wrong source | Human games vs teacher games can confound. | Sample random positions from teacher's self-play games specifically. |
| Not accounting for player alternation | In games where players alternate (chess), rollout activations alternate perspective. | Use every-other timestep for single-player concepts (`z_{2t}`), or all for both-player concepts. See §8.4.1. |
| Forgetting the extended subpar criterion | Using only 1 subpar rollout instead of multiple at different depths. | Implement the extended version with `T_tilde = T-5` different branching points (Equation 5 in the paper). |

---

## 14. File/Module Structure Suggestion

```
teachability/
├── __init__.py
├── prototypes.py          # find_dynamic_prototypes(), find_static_prototypes()
├── student_selection.py   # select_student(), measure_agreement()
├── training.py            # teach_student()
├── benchmark.py           # run_full_teachability_benchmark()
├── pipeline.py            # run_teachability_pipeline() (orchestrator)
├── utils.py               # make_random_concept_vector(), data loading
└── tests/
    ├── test_prototypes.py
    ├── test_training.py
    └── test_benchmark.py
```

---

## 15. Dependencies

```
torch >= 1.10
numpy
scikit-learn          # for train_test_split
tqdm                  # optional, for progress bars
```

If convex optimisation is also being done in this repo (for finding concept vectors), you'll also need:
```
cvxpy
```

---

## 16. Key Equations Reference

**Equation 5** — Dynamic concept discovery (how vectors were found, for context):
```
min ||v_c,l||_1
s.t. v_c,l · z⁺_{t,l} >= v_c,l · z⁻_{t,l,j}   ∀ t ≤ T, j ≤ T̃
```

**Equation 6** — Teachability metric (top-1 policy agreement):
```
T = Σ_{x_i ∈ X_test} 1[argmax(π_s(x_i)) == argmax(π_t(x_i))]
```

**Training objective** — KL divergence distillation:
```
L = Σ_{x_i ∈ X_train} KL[π_teacher(x_i) || π_student(x_i)]
```

**Equation 17** — Student selection criterion:
```
T = Σ_{x_i ∈ X} 1[argmax(π_s(x_i)) == argmax(π_t(x_i))]
Select latest checkpoint where T / |X| < 0.2
```