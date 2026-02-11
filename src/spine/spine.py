import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from alphazero.games.othello import OthelloBoard
from alphazero.players import AlphaZeroPlayer
from src.hooks.extract import get_single_activation


def find_disagreement_positions(
    positions: List[Dict],
    strong_net,
    weak_net,
    board_size: int,
    n_sims_pass1: int = 100,
    n_sims_pass2: int = 1000,
) -> List[Dict]:
    """
    Two-pass disagreement filter: find positions where strong and weak models
    choose different moves.

    Pass 1: quick MCTS with n_sims_pass1 to find candidates.
    Pass 2: deeper MCTS with n_sims_pass2 to confirm disagreements.

    Returns list of positions where models still disagree after both passes.
    """
    board = OthelloBoard(n=board_size)

    # Pass 1
    strong_player = AlphaZeroPlayer(nn=strong_net, n_sim=n_sims_pass1)
    weak_player = AlphaZeroPlayer(nn=weak_net, n_sim=n_sims_pass1)

    pass1_survivors = []
    for pos in tqdm(positions, desc="Disagreement pass 1"):
        board.grid = pos['grid'].copy()
        board.player = pos['player']

        strong_player.reset()
        strong_move, _, _, _ = strong_player.get_move(board)

        weak_player.reset()
        weak_move, _, _, _ = weak_player.get_move(board)

        if weak_move != strong_move:
            pass1_survivors.append(pos)

    print(f"Pass 1: {len(pass1_survivors)}/{len(positions)} positions disagree")

    # Pass 2
    strong_player = AlphaZeroPlayer(nn=strong_net, n_sim=n_sims_pass2)
    weak_player = AlphaZeroPlayer(nn=weak_net, n_sim=n_sims_pass2)

    pass2_survivors = []
    for pos in tqdm(pass1_survivors, desc="Disagreement pass 2"):
        board.grid = pos['grid'].copy()
        board.player = pos['player']

        strong_player.reset()
        strong_move, _, _, _ = strong_player.get_move(board)

        weak_player.reset()
        weak_move, _, _, _ = weak_player.get_move(board)

        if weak_move != strong_move:
            pass2_survivors.append(pos)

    print(f"Pass 2: {len(pass2_survivors)}/{len(pass1_survivors)} positions still disagree")
    return pass2_survivors


def build_spine_tree(
    mct,
    board: OthelloBoard,
    net,
    max_depth: int,
    layer_name: str,
    sort_key: str = "N",
) -> List[List[Optional[Dict]]]:
    """
    Build a 'spine' decision tree from an MCTS tree.

    Main path (best moves) spawns second-best alternatives at each level.
    Second-best branches only follow their own best moves (no further branching).

    Heap-style indexing: parent (d, idx) -> children (d+1, 2*idx) best, (d+1, 2*idx+1) second.

    At each node, extracts activations via get_single_activation.

    Returns levels[depth][idx] = node_dict or None.
    """
    levels = [[] for _ in range(max_depth + 1)]
    gid_counter = 0

    def walk(node, board_state, depth, label, move, idx, is_main_path):
        nonlocal gid_counter

        if depth > max_depth:
            return

        act = get_single_activation(net, board_state.grid.copy(), board_state.player, layer_name)

        # Ensure list is long enough
        while len(levels[depth]) <= idx:
            levels[depth].append(None)

        levels[depth][idx] = {
            "grid": board_state.grid.copy(),
            "player": board_state.player,
            "move": move,
            "label": label,
            "n_children": len(node.children),
            "N": getattr(node, 'N', None),
            "Q": getattr(node, 'Q', None),
            "depth": depth,
            "activations": act,
            "main": is_main_path,
            "id": gid_counter,
        }
        gid_counter += 1

        if not node.children:
            return

        items = sorted(
            node.children.items(),
            key=lambda x: getattr(x[1], sort_key),
            reverse=True,
        )

        # Best child - always follow
        best_move, best_node = items[0]
        b_best = board_state.clone()
        b_best.play_move(best_move)
        walk(best_node, b_best, depth + 1, "best", best_move, 2 * idx, is_main_path)

        # Second-best child - only if on main path
        if len(items) >= 2 and is_main_path:
            second_move, second_node = items[1]
            b_second = board_state.clone()
            b_second.play_move(second_move)
            walk(second_node, b_second, depth + 1, "2nd", second_move, 2 * idx + 1, False)

    walk(mct.root, board.clone(), 0, "root", None, 0, True)
    return levels


def _sorted_children_by_key(node, sort_key: str):
    if node is None or not node.children:
        return []
    return sorted(
        node.children.items(),
        key=lambda x: getattr(x[1], sort_key),
        reverse=True,
    )


def _snapshot_board(board: OthelloBoard) -> Dict:
    return {
        "grid": board.grid.copy(),
        "player": board.player,
    }


def extract_rollout_contrasts(
    mct,
    board: OthelloBoard,
    max_depth: int,
    sort_key: str = "N",
    min_value_gap: float = 0.20,
    min_visit_gap_ratio: float = 0.10,
    t_offset: int = 5,
) -> Dict:
    """
    Build one optimal rollout and multiple eligible subpar rollouts.

    Subpar rollout at depth j must branch from the optimal path at depth j
    and satisfy:
      |Q_best - Q_alt| >= min_value_gap
       OR
      (N_best - N_alt) / N_best >= min_visit_gap_ratio
    """
    if mct is None or mct.root is None:
        return {
            "optimal_rollout": [],
            "subpar_rollouts": [],
            "required_depths": 0,
            "available_depths": 0,
        }

    # Build optimal rollout from root.
    optimal_rollout = []
    optimal_nodes = [mct.root]       # node at depth j before taking move j
    optimal_boards = [board.clone()] # board state at depth j

    current_node = mct.root
    current_board = board.clone()

    for _ in range(max_depth):
        children = _sorted_children_by_key(current_node, sort_key)
        if not children:
            break
        best_move, best_node = children[0]
        current_board.play_move(best_move)
        optimal_rollout.append(_snapshot_board(current_board))
        current_node = best_node
        optimal_nodes.append(current_node)
        optimal_boards.append(current_board.clone())

    T = len(optimal_rollout)
    if T == 0:
        return {
            "optimal_rollout": [],
            "subpar_rollouts": [],
            "required_depths": 0,
            "available_depths": 0,
        }

    t_tilde = max(1, T - t_offset)
    t_tilde = min(t_tilde, T)

    subpar_rollouts = []
    for j in range(t_tilde):
        node_j = optimal_nodes[j]
        board_j = optimal_boards[j].clone()
        children = _sorted_children_by_key(node_j, sort_key)

        if len(children) < 2:
            continue

        best_move, best_node = children[0]
        best_q = float(getattr(best_node, "Q", 0.0))
        best_n = float(max(1, getattr(best_node, "N", 0)))

        chosen_alt = None
        for alt_move, alt_node in children[1:]:
            alt_q = float(getattr(alt_node, "Q", 0.0))
            alt_n = float(getattr(alt_node, "N", 0))
            value_gap = abs(best_q - alt_q)
            visit_gap_ratio = (best_n - alt_n) / best_n
            if value_gap >= min_value_gap or visit_gap_ratio >= min_visit_gap_ratio:
                chosen_alt = (alt_move, alt_node, value_gap, visit_gap_ratio)
                break

        if chosen_alt is None:
            continue

        alt_move, alt_node, value_gap, visit_gap_ratio = chosen_alt

        # Prefix matches optimal rollout up to depth j-1.
        rollout_states = [
            {"grid": s["grid"].copy(), "player": s["player"]}
            for s in optimal_rollout[:j]
        ]

        # Diverge at depth j with subpar move, then follow best descendants.
        board_alt = board_j.clone()
        board_alt.play_move(alt_move)
        rollout_states.append(_snapshot_board(board_alt))
        current_alt_node = alt_node

        # Continue until comparable horizon T.
        for _ in range(j + 1, T):
            alt_children = _sorted_children_by_key(current_alt_node, sort_key)
            if not alt_children:
                break
            next_move, next_node = alt_children[0]
            board_alt.play_move(next_move)
            rollout_states.append(_snapshot_board(board_alt))
            current_alt_node = next_node

        subpar_rollouts.append({
            "depth": j,
            "states": rollout_states,
            "value_gap": float(value_gap),
            "visit_gap_ratio": float(visit_gap_ratio),
            "best_move": best_move,
            "subpar_move": alt_move,
        })

    return {
        "optimal_rollout": optimal_rollout,
        "subpar_rollouts": subpar_rollouts,
        "required_depths": int(t_tilde),
        "available_depths": int(len(subpar_rollouts)),
    }


def extract_pairs_from_spine(
    levels: List[List[Optional[Dict]]],
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Extract (Z_plus, Z_minus) activation pairs from spine tree levels.

    At each depth: main node (index 0) = Z+, each alternative node = Z-.

    Returns:
        Z_plus: (n_pairs, dim) - activations from main path nodes
        Z_minus: (n_pairs, dim) - activations from alternative nodes
        pairs_metadata: list of dicts with pair info (ids, depth, etc.)
    """
    Z_plus_list = []
    Z_minus_list = []
    pairs_metadata = []

    for d, level in enumerate(levels):
        if not level or level[0] is None:
            continue
        main = level[0]

        for node in level[1:]:
            if node is None:
                continue
            Z_plus_list.append(main['activations'])
            Z_minus_list.append(node['activations'])
            pairs_metadata.append({
                'depth': d,
                'plus_id': main['id'],
                'minus_id': node['id'],
                'plus_label': main['label'],
                'minus_label': node['label'],
                'plus_N': main.get('N'),
                'minus_N': node.get('N'),
                'plus_Q': main.get('Q'),
                'minus_Q': node.get('Q'),
            })

    if len(Z_plus_list) == 0:
        return np.empty((0, 0)), np.empty((0, 0)), []

    Z_plus = np.stack(Z_plus_list)
    Z_minus = np.stack(Z_minus_list)
    return Z_plus, Z_minus, pairs_metadata
