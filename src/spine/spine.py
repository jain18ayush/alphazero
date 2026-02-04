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
