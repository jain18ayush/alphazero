from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from alphazero.games.othello import OthelloBoard, OthelloNet
from alphazero.players import AlphaZeroPlayer
from src.hooks.extract import extract_features_by_layer, get_single_activation
from src.spine.spine import extract_rollout_contrasts


@dataclass
class ConceptVector:
    layer: str
    vector: np.ndarray
    path: str


def _board_from_pos(pos: dict) -> OthelloBoard:
    board = OthelloBoard(n=pos["grid"].shape[0])
    board.grid = pos["grid"].copy()
    board.player = pos["player"]
    return board


def _score_rollout_states(
    net: OthelloNet,
    states: Sequence[dict],
    layer: str,
    v: np.ndarray,
) -> List[float]:
    if len(states) == 0:
        return []
    feats = extract_features_by_layer(net, list(states), [{"name": layer}])
    Z = feats[layer]  # (N, D)
    return (Z @ v).tolist()


def dynamic_prototypes_for_concept(
    net: OthelloNet,
    positions: Sequence[dict],
    layer: str,
    v: np.ndarray,
    n_sim: int,
    max_depth: int,
    sort_key: str = "N",
    min_margin: float = 0.0,
    min_value_gap: float = 0.20,
    min_visit_gap_ratio: float = 0.10,
    t_offset: int = 5,
    top_percent: float | None = None,
    max_prototypes: int | None = None,
) -> Tuple[List[dict], List[dict], Dict[str, int | float]]:
    """
    Dynamic prototype selection with multi-depth subpar rollout constraints.

    A candidate passes when for every eligible subpar rollout j and every compared
    timestep t:
        v·z_plus[t] >= v·z_minus_j[t]
    """
    player = AlphaZeroPlayer(nn=net, n_sim=n_sim)

    accepted: List[dict] = []
    accepted_meta: List[dict] = []

    stats = {
        "n_positions": int(len(positions)),
        "n_no_optimal": 0,
        "n_missing_subpar": 0,
        "n_failed_margin": 0,
        "n_accepted": 0,
    }

    for idx, pos in enumerate[dict](positions):
        board = _board_from_pos(pos)
        if board.is_game_over():
            stats["n_no_optimal"] += 1
            continue

        player.reset()
        _ = player.get_move(board)

        contrasts = extract_rollout_contrasts(
            player.mct,
            board.clone(),
            max_depth=max_depth,
            sort_key=sort_key,
            min_value_gap=min_value_gap,
            min_visit_gap_ratio=min_visit_gap_ratio,
            t_offset=t_offset,
        )

        optimal_rollout = contrasts["optimal_rollout"]
        subpar_rollouts = contrasts["subpar_rollouts"]
        required_depths = int(contrasts["required_depths"])

        if len(optimal_rollout) == 0:
            stats["n_no_optimal"] += 1
            continue

        if len(subpar_rollouts) < required_depths:
            stats["n_missing_subpar"] += 1
            continue

        # Batch all states (optimal + all subpar) into one forward pass.
        all_states = list(optimal_rollout)
        segment_lengths = [len(optimal_rollout)]
        for subpar in subpar_rollouts:
            all_states.extend(subpar["states"])
            segment_lengths.append(len(subpar["states"]))

        all_scores = _score_rollout_states(net, all_states, layer, v)

        # Slice back into optimal and subpar segments.
        offset = 0
        optimal_scores = all_scores[offset:offset + segment_lengths[0]]
        offset += segment_lengths[0]
        subpar_scores_list = []
        for seg_len in segment_lengths[1:]:
            subpar_scores_list.append(all_scores[offset:offset + seg_len])
            offset += seg_len

        passed = True
        fail_reason = None
        n_constraints = 0
        min_margin_obs = float("inf")

        for subpar, subpar_scores in zip(subpar_rollouts, subpar_scores_list):
            T = min(len(optimal_scores), len(subpar_scores))
            for t in range(T):
                margin = optimal_scores[t] - subpar_scores[t]
                n_constraints += 1
                min_margin_obs = min(min_margin_obs, margin)
                if margin < min_margin:
                    passed = False
                    fail_reason = {
                        "reason": "margin_violation",
                        "depth": int(subpar["depth"]),
                        "t": int(t),
                        "margin": float(margin),
                    }
                    break
            if not passed:
                break

        if passed and n_constraints > 0:
            accepted.append(pos)
            accepted_meta.append({
                "source_idx": int(idx),
                "n_constraints": int(n_constraints),
                "min_margin": float(min_margin_obs),
                "required_depths": int(required_depths),
                "available_depths": int(len(subpar_rollouts)),
            })
        else:
            stats["n_failed_margin"] += 1

    if len(accepted) == 0:
        return [], [], stats

    if top_percent is not None:
        k = max(1, int(len(accepted) * (top_percent / 100.0)))
        order = np.argsort([-m["min_margin"] for m in accepted_meta[:len(accepted)]])[:k]
        accepted = [accepted[i] for i in order]
        accepted_meta = [accepted_meta[i] for i in order]

    if max_prototypes is not None and len(accepted) > max_prototypes:
        accepted = accepted[:max_prototypes]
        accepted_meta = accepted_meta[:max_prototypes]

    stats["n_accepted"] = int(len(accepted))
    return accepted, accepted_meta, stats


def mcts_policy_for_positions(
    net: OthelloNet,
    positions: Sequence[dict],
    n_sim: int,
    temp: float = 1.0,
) -> Tuple[List[np.ndarray], List[int]]:
    player = AlphaZeroPlayer(nn=net, n_sim=n_sim)
    policies: List[np.ndarray] = []
    top1_indices: List[int] = []

    for pos in positions:
        board = _board_from_pos(pos)
        if board.is_game_over():
            pi = np.zeros(net.action_size, dtype=np.float32)
            pi[-1] = 1.0
            policies.append(pi)
            top1_indices.append(int(np.argmax(pi)))
            continue
        player.reset()
        _, action_probs, _, _ = player.get_move(board, temp=temp)
        pi = net.to_neural_output(action_probs).astype(np.float32)
        policies.append(pi)
        top1_indices.append(int(np.argmax(pi)))

    return policies, top1_indices


def measure_top1_agreement(
    teacher: OthelloNet,
    student: OthelloNet,
    positions: Sequence[dict],
    n_sim: int,
    temp: float = 1.0,
) -> Tuple[int, float]:
    if len(positions) == 0:
        return 0, 0.0

    _, teacher_top1 = mcts_policy_for_positions(teacher, positions, n_sim=n_sim, temp=temp)
    _, student_top1 = mcts_policy_for_positions(student, positions, n_sim=n_sim, temp=temp)

    total = min(len(teacher_top1), len(student_top1))
    if total == 0:
        return 0, 0.0

    matches = sum(int(a == b) for a, b in zip(teacher_top1[:total], student_top1[:total]))
    return matches, matches / total


def distill_student(
    student: OthelloNet,
    positions: Sequence[dict],
    target_pis: Sequence[np.ndarray],
    epochs: int,
    lr: float,
    batch_size: int,
) -> List[float]:
    if len(positions) == 0:
        return []

    device = next(student.parameters()).device
    x = torch.tensor(
        np.array([p["player"] * p["grid"] for p in positions]),
        dtype=torch.float32,
        device=device,
    )
    y = torch.tensor(np.stack(target_pis), dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    losses: List[float] = []

    student.train()
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0], device=device)
        for i in range(0, x.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            xb = x[idx]
            yb = y[idx]
            student_log_probs, _ = student(xb)
            # KL(teacher || student)
            loss = F.kl_div(student_log_probs, yb, reduction="batchmean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
    student.eval()
    return losses


def _measure_top1_agreement_with_cached_teacher(
    teacher_top1: List[int],
    student: OthelloNet,
    positions: Sequence[dict],
    n_sim: int,
    temp: float = 1.0,
) -> Tuple[int, float]:
    if len(teacher_top1) == 0:
        return 0, 0.0
    _, student_top1 = mcts_policy_for_positions(student, positions, n_sim=n_sim, temp=temp)
    total = min(len(teacher_top1), len(student_top1))
    if total == 0:
        return 0, 0.0
    matches = sum(int(a == b) for a, b in zip(teacher_top1[:total], student_top1[:total]))
    return matches, matches / total


def run_teachability_benchmark(
    load_student: Callable[[], OthelloNet],
    teacher: OthelloNet,
    X_train_concept: Sequence[dict],
    X_test_concept: Sequence[dict],
    X_train_random: Sequence[dict],
    X_test_random: Sequence[dict],
    n_sim: int,
    temp: float,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Dict[str, float | int | List[float]]:
    # Pre-compute teacher MCTS once per position set (saves 3 redundant runs).
    teacher_pis_concept, teacher_top1_train_C = mcts_policy_for_positions(
        teacher, X_train_concept, n_sim=n_sim, temp=temp,
    )
    teacher_pis_random, teacher_top1_train_R = mcts_policy_for_positions(
        teacher, X_train_random, n_sim=n_sim, temp=temp,
    )
    _, teacher_top1_test_C = mcts_policy_for_positions(
        teacher, X_test_concept, n_sim=n_sim, temp=temp,
    )
    _, teacher_top1_test_R = mcts_policy_for_positions(
        teacher, X_test_random, n_sim=n_sim, temp=temp,
    )

    # Baseline on concept test.
    student_baseline = load_student()
    _, baseline = _measure_top1_agreement_with_cached_teacher(
        teacher_top1_test_C, student_baseline, X_test_concept, n_sim=n_sim, temp=temp,
    )

    # Train concept student.
    student_concept = load_student()
    losses_concept = distill_student(
        student_concept,
        X_train_concept,
        teacher_pis_concept,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    _, train_C_eval_C = _measure_top1_agreement_with_cached_teacher(
        teacher_top1_test_C, student_concept, X_test_concept, n_sim=n_sim, temp=temp,
    )
    _, train_C_eval_R = _measure_top1_agreement_with_cached_teacher(
        teacher_top1_test_R, student_concept, X_test_random, n_sim=n_sim, temp=temp,
    )

    # Train random student.
    student_random = load_student()
    losses_random = distill_student(
        student_random,
        X_train_random,
        teacher_pis_random,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    _, train_R_eval_C = _measure_top1_agreement_with_cached_teacher(
        teacher_top1_test_C, student_random, X_test_concept, n_sim=n_sim, temp=temp,
    )
    _, train_R_eval_R = _measure_top1_agreement_with_cached_teacher(
        teacher_top1_test_R, student_random, X_test_random, n_sim=n_sim, temp=temp,
    )

    return {
        "baseline_eval_C": float(baseline),
        "train_C_eval_C": float(train_C_eval_C),
        "train_C_eval_R": float(train_C_eval_R),
        "train_R_eval_C": float(train_R_eval_C),
        "train_R_eval_R": float(train_R_eval_R),
        "loss_tail_concept": [float(x) for x in losses_concept[-10:]],
        "loss_tail_random": [float(x) for x in losses_random[-10:]],
    }


def is_teachable(results: Dict[str, float], margin: float = 0.05) -> Tuple[bool, float]:
    gain = float(results["train_C_eval_C"] - results["train_R_eval_C"])
    return bool(gain > margin), gain


def _checkpoint_sort_key(path: str) -> Tuple[int, str]:
    name = Path(path).name
    nums = re.findall(r"\d+", name)
    step = int(nums[-1]) if nums else -1
    return step, name


def select_student_checkpoint(
    checkpoint_paths: Sequence[str],
    make_model_cfg: Callable[[str], dict],
    load_model: Callable[[dict], OthelloNet],
    teacher: OthelloNet,
    prototypes: Sequence[dict],
    overlap_threshold: float,
    n_sim: int,
    temp: float,
    teacher_top1: List[int] | None = None,
) -> Dict:
    ordered = sorted(checkpoint_paths, key=_checkpoint_sort_key)
    checked = []

    # Pre-compute teacher MCTS once if not provided.
    if teacher_top1 is None:
        _, teacher_top1 = mcts_policy_for_positions(
            teacher, prototypes, n_sim=n_sim, temp=temp,
        )

    selected_path = ordered[0]
    selected_overlap = None

    for path in reversed(ordered):
        cfg = make_model_cfg(path)
        student = load_model(cfg)
        _, overlap = _measure_top1_agreement_with_cached_teacher(
            teacher_top1, student, prototypes, n_sim=n_sim, temp=temp,
        )
        checked.append({"path": path, "overlap": float(overlap)})
        if overlap < overlap_threshold:
            selected_path = path
            selected_overlap = overlap
            break

    if selected_overlap is None:
        cfg = make_model_cfg(selected_path)
        student = load_model(cfg)
        _, selected_overlap = _measure_top1_agreement_with_cached_teacher(
            teacher_top1, student, prototypes, n_sim=n_sim, temp=temp,
        )

    return {
        "selected_path": selected_path,
        "selected_overlap": float(selected_overlap),
        "overlap_threshold": float(overlap_threshold),
        "checked": checked,
    }
