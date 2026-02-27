from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Tuple

import numpy as np
import torch

from alphazero.games.othello import OthelloBoard
from alphazero.players import AlphaZeroPlayer


def amplify_activation(
    z: torch.Tensor,
    v: torch.Tensor,
    alpha: float,
    beta: float = 0.01,
) -> torch.Tensor:
    """
    Apply concept amplification to an activation tensor.

    z' = (1 - alpha) * z + alpha * beta * (||z|| / ||v||) * v

    Supports shapes: (D,), (N, D), or (N, C, H, W).
    """
    if not torch.is_tensor(z):
        raise TypeError("z must be a torch.Tensor")
    if not torch.is_tensor(v):
        raise TypeError("v must be a torch.Tensor")
    if v.dim() != 1:
        raise ValueError("v must be a 1D concept vector")

    if z.dim() == 1:
        z_norm = torch.norm(z, p=2)
        v_norm = torch.norm(v, p=2)
        if v_norm < 1e-10:
            return z
        scale = beta * (z_norm / v_norm)
        return (1 - alpha) * z + alpha * scale * v

    if z.dim() == 2:
        if v.numel() != z.shape[1]:
            raise ValueError("v dimension does not match z feature dimension")
        z_norm = torch.norm(z, p=2, dim=1, keepdim=True)
        v_norm = torch.norm(v, p=2)
        if v_norm < 1e-10:
            return z
        scale = beta * (z_norm / v_norm)
        return (1 - alpha) * z + alpha * scale * v

    if z.dim() == 4:
        n, c, h, w = z.shape
        z_flat = z.view(n, -1)
        if v.numel() != z_flat.shape[1]:
            raise ValueError("v dimension does not match flattened z dimension")
        v_norm = torch.norm(v, p=2)
        if v_norm < 1e-10:
            return z
        z_norm = torch.norm(z_flat, p=2, dim=1, keepdim=True)
        scale = beta * (z_norm / v_norm)
        z_tilde = (1 - alpha) * z_flat + alpha * scale * v
        return z_tilde.view(n, c, h, w)

    raise ValueError(f"Unsupported activation shape: {tuple(z.shape)}")


@contextmanager
def amplification_hook(
    net: torch.nn.Module,
    layer: str,
    concept_vector: np.ndarray | torch.Tensor,
    alpha: float,
    beta: float = 0.01,
):
    """
    Context manager that registers a forward hook and applies amplification.
    """
    modules = dict(net.named_modules())
    if layer not in modules:
        available = [name for name, _ in net.named_modules() if name]
        raise KeyError(f"Layer '{layer}' not found. Available: {available}")

    if isinstance(concept_vector, np.ndarray):
        v = torch.tensor(concept_vector, dtype=torch.float32, device=next(net.parameters()).device)
    else:
        v = concept_vector.to(next(net.parameters()).device)

    def hook_fn(_module, _inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not torch.is_tensor(out):
            raise TypeError("Hook output is not a torch.Tensor")
        return amplify_activation(out, v, alpha=alpha, beta=beta)

    handle = modules[layer].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@dataclass
class AmplificationResult:
    alpha: float
    move_change_rate: float
    value_shift_mean: float
    value_shift_std: float
    n_positions: int


def _board_from_position(position: dict) -> OthelloBoard:
    grid = np.asarray(position["grid"])
    player = int(position["player"])
    return OthelloBoard(n=grid.shape[0], grid=grid.copy(), player=player)


def _run_mcts(
    net: torch.nn.Module,
    board: OthelloBoard,
    n_sims: int,
    layer: Optional[str] = None,
    concept_vector: Optional[np.ndarray | torch.Tensor] = None,
    alpha: float = 0.0,
    beta: float = 0.01,
) -> Tuple[tuple, float]:
    player = AlphaZeroPlayer(n_sim=n_sims, nn=net)

    if layer is not None and concept_vector is not None and alpha != 0.0:
        with amplification_hook(net, layer, concept_vector, alpha=alpha, beta=beta):
            move, _, _, _ = player.get_move(board, temp=0)
    else:
        move, _, _, _ = player.get_move(board, temp=0)

    value = float(player.mct.root.Q) if player.mct.root is not None else 0.0
    return move, value


def run_amplification_experiment(
    net: torch.nn.Module,
    concept_vector: np.ndarray,
    layer: str,
    positions: Iterable[dict],
    alpha_range: Iterable[float],
    n_sims: int,
    beta: float = 0.01,
) -> Dict[float, AmplificationResult]:
    """
    Run concept amplification via MCTS and report move/value shifts.

    Returns a dict keyed by alpha.
    """
    positions_list = list(positions)
    if len(positions_list) == 0:
        raise ValueError("No positions provided")

    baseline_moves = []
    baseline_values = []

    for pos in positions_list:
        board = _board_from_position(pos)
        move, value = _run_mcts(net, board, n_sims)
        baseline_moves.append(move)
        baseline_values.append(value)

    results: Dict[float, AmplificationResult] = {}

    for alpha in alpha_range:
        move_changes = 0
        value_shifts = []

        for idx, pos in enumerate(positions_list):
            board = _board_from_position(pos)
            move, value = _run_mcts(
                net,
                board,
                n_sims,
                layer=layer,
                concept_vector=concept_vector,
                alpha=alpha,
                beta=beta,
            )
            if move != baseline_moves[idx]:
                move_changes += 1
            value_shifts.append(value - baseline_values[idx])

        value_shifts_arr = np.asarray(value_shifts, dtype=np.float32)
        results[alpha] = AmplificationResult(
            alpha=float(alpha),
            move_change_rate=float(move_changes / len(positions_list)),
            value_shift_mean=float(value_shifts_arr.mean()) if len(value_shifts_arr) else 0.0,
            value_shift_std=float(value_shifts_arr.std()) if len(value_shifts_arr) else 0.0,
            n_positions=len(positions_list),
        )

    return results


def compute_normalized_improvement(
    baseline_accuracy: float,
    amplified_accuracy: float,
) -> float:
    if baseline_accuracy == 0.0:
        if amplified_accuracy > 0.0:
            return float("inf")
        return 0.0
    return (amplified_accuracy - baseline_accuracy) / baseline_accuracy


def _solution_to_action(solution: Any) -> Action:
    if solution == "pass" or solution == ["pass"]:
        return ("pass",)
    if isinstance(solution, (list, tuple)) and len(solution) == 2:
        return (int(solution[0]), int(solution[1]))
    if isinstance(solution, dict) and "row" in solution and "col" in solution:
        return (int(solution["row"]), int(solution["col"]))
    raise ValueError(f"Cannot parse solution move: {solution}")


def _normalize_solution_set(raw_solutions: Iterable[Any]) -> set[Action]:
    if raw_solutions is None:
        return set()
    actions: set[Action] = set()
    for solution in raw_solutions:
        actions.add(_solution_to_action(solution))
    return actions


def load_amplification_eval_dataset(path: str, max_positions: int | None = None) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)

    if isinstance(data, dict) and "positions" in data:
        data = data["positions"]

    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must be a list of positions or contain a 'positions' key.")

    if max_positions is not None:
        data = data[: max_positions]

    required_keys = {"grid", "player", "solutions"}
    for entry in data:
        if not required_keys.issubset(entry.keys()):
            raise ValueError("Each evaluation position must include 'grid', 'player', and 'solutions'.")

    return data


def compute_accuracy_on_dataset(
    net: torch.nn.Module,
    dataset: Iterable[dict],
    layer: Optional[str] = None,
    concept_vector: Optional[np.ndarray | torch.Tensor] = None,
    alpha: float = 0.0,
    beta: float = 0.01,
    n_sims: int = 200,
) -> float:
    if len(list(dataset)) == 0:
        raise ValueError("Evaluation dataset is empty")

    correct = 0
    total = 0

    for position in dataset:
        board = _board_from_position(position)
        solutions = _normalize_solution_set(position.get("solutions"))
        if not solutions:
            continue

        move, _ = _run_mcts(
            net,
            board,
            n_sims,
            layer=layer,
            concept_vector=concept_vector,
            alpha=alpha,
            beta=beta,
        )

        if move in solutions or tuple(move) in solutions:
            correct += 1
        total += 1

    if total == 0:
        return 0.0
    return correct / total


def evaluate_concept_on_dataset(
    net: torch.nn.Module,
    concept_vector: np.ndarray,
    layer: str,
    eval_dataset: Iterable[dict],
    alpha_range: Iterable[float],
    n_sims: int,
    beta: float = 0.01,
) -> dict:
    baseline = compute_accuracy_on_dataset(
        net,
        eval_dataset,
        layer=layer,
        concept_vector=None,
        alpha=0.0,
        beta=beta,
        n_sims=n_sims,
    )

    results: dict[str, dict] = {}
    for alpha in alpha_range:
        accuracy = compute_accuracy_on_dataset(
            net,
            eval_dataset,
            layer=layer,
            concept_vector=concept_vector,
            alpha=alpha,
            beta=beta,
            n_sims=n_sims,
        )
        results[str(alpha)] = {
            "accuracy": accuracy,
            "normalized_improvement": compute_normalized_improvement(baseline, accuracy),
        }

    return {
        "baseline_accuracy": baseline,
        "n_positions": len(list(eval_dataset)),
        "alpha_results": results,
    }
