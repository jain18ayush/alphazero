from alphazero.arena import Arena
from alphazero.players import RandomPlayer, MCTSPlayer
from alphazero.games.othello import OthelloBoard

from src.registries import DATASETS

@DATASETS.register("self_play") 
def generate_games(cfg):
    # declaring the attributes up top 
    board_size = cfg['board_size']
    n_games = cfg['n_games']

    """Generate games and collect all positions"""
    positions = []
    
    board = OthelloBoard(n=board_size)
    p1 = RandomPlayer()  # Fast
    p2 = RandomPlayer()
    
    for game_idx in range(n_games):
        board.reset()
        game_positions = []
        
        while not board.is_game_over():
            # Record position
            game_positions.append({
                'grid': board.grid.copy(),
                'player': board.player,
                'move_number': len(game_positions)
            })
            
            # Play move
            if board.player == 1:
                move, _, _, _ = p1.get_move(board)
            else:
                move, _, _, _ = p2.get_move(board)
            board.play_move(move)
        
        positions.extend(game_positions)
    
    return positions

import numpy as np
from typing import List, Dict, Optional
from src.registries import DATASETS


def follow_trajectory(node, depth: int) -> List[Dict]:
    """
    Follow most-visited path down from node, collect board states.
    
    Returns list of dicts with 'grid' and 'player' for each state.
    """
    states = []
    current = node
    
    for _ in range(depth):
        if current is None:
            break
        
        # Store state info
        states.append({
            'grid': current.state.grid.copy(),
            'player': current.state.player,
        })
        
        # Move to most-visited child
        if not current.children:
            break
        
        best_child = max(current.children.values(), key=lambda c: c.visit_count)
        current = best_child
    
    return states


def get_trajectory_pair(mct, board, cfg: dict) -> Optional[Dict]:
    """
    Extract chosen vs rejected trajectory from MCTS root.
    Returns None if difference isn't meaningful (scenario 3).
    
    Args:
        mct: The MCT object after search
        board: The board at root position
        cfg: Config with min_value_gap, min_visit_ratio, depth
    """
    root = mct.root
    
    if root is None or not root.children:
        return None
    
    # Sort children by visit count
    children_list = list(root.children.values())
    children_list.sort(key=lambda c: c.visit_count, reverse=True)
    
    if len(children_list) < 2:
        return None
    
    best = children_list[0]
    second = children_list[1]
    
    # Get values (Q-values from MCTS)
    # In your MCT, node.value is the accumulated value, need to normalize
    best_value = best.value / best.visit_count if best.visit_count > 0 else 0
    second_value = second.value / second.visit_count if second.visit_count > 0 else 0
    
    # Scenario 3 filter: require meaningful gap
    value_gap = abs(best_value - second_value)
    visit_ratio = best.visit_count / (second.visit_count + 1e-8)
    
    min_value_gap = cfg.get('min_value_gap', 0.20)
    min_visit_ratio = cfg.get('min_visit_ratio', 1.10)
    
    if value_gap < min_value_gap and visit_ratio < min_visit_ratio:
        return None
    
    depth = cfg.get('depth', 5)
    
    # Extract trajectories
    chosen_states = follow_trajectory(best, depth)
    rejected_states = follow_trajectory(second, depth)
    
    if len(chosen_states) < 2 or len(rejected_states) < 2:
        return None
    
    return {
        'chosen': chosen_states,
        'rejected': rejected_states,
        'root_grid': board.grid.copy(),
        'root_player': board.player,
        'value_gap': float(value_gap),
        'visit_ratio': float(visit_ratio),
        'best_value': float(best_value),
        'second_value': float(second_value),
        'best_visits': int(best.visit_count),
        'second_visits': int(second.visit_count),
    }


@DATASETS.register("mcts_trajectories")
def collect_trajectory_pairs(cfg: dict) -> List[Dict]:
    """
    Collect trajectory pairs from MCTS for dynamic concept learning.
    
    Config:
        board_size: int
        n_pairs: int - target number of pairs
        max_attempts: int - give up after this many tries
        n_sims: int - MCTS simulations per position
        depth: int - trajectory depth
        min_value_gap: float - minimum value difference (default 0.20)
        min_visit_ratio: float - minimum visit ratio (default 1.10)
        checkpoint_path: str - path to model checkpoint
        model_name: str - name of model file
    """
    from alphazero.games.othello import OthelloBoard, OthelloNet
    from alphazero.players import AlphaZeroPlayer
    
    board_size = cfg['board_size']
    n_pairs = cfg.get('n_pairs', 50)
    max_attempts = cfg.get('max_attempts', n_pairs * 10)
    n_sims = cfg.get('n_sims', 200)
    
    # Load model
    net = OthelloNet(n=board_size)
    checkpoint_path = cfg.get('checkpoint_path', '')
    model_name = cfg.get('model_name', 'alphazero-othello')
    
    import torch
    import os
    model_file = os.path.join(checkpoint_path, f"{model_name}.pt")
    if os.path.exists(model_file):
        net.load_state_dict(torch.load(model_file, map_location='cpu'))
        print(f"Loaded model from {model_file}")
    else:
        print(f"Warning: No model found at {model_file}, using random weights")
    
    net.eval()
    
    # Create player
    player = AlphaZeroPlayer(n_sim=n_sims, nn=net)
    
    pairs = []
    stats = {
        'attempts': 0,
        'game_over_skips': 0,
        'no_children_skips': 0,
        'scenario_3_filtered': 0,
        'short_trajectory_skips': 0,
        'accepted': 0,
    }
    
    rng = np.random.RandomState(cfg.get('seed', 42))
    
    while len(pairs) < n_pairs and stats['attempts'] < max_attempts:
        stats['attempts'] += 1
        
        # Create fresh board
        board = OthelloBoard(n=board_size)
        
        # Play to random mid-game position
        n_moves = rng.randint(4, 20)
        
        for _ in range(n_moves):
            if board.is_game_over():
                break
            legal = board.get_moves()
            if not legal or legal == [board.pass_move]:
                break
            move = legal[rng.randint(len(legal))]
            board.play_move(move)
        
        if board.is_game_over():
            stats['game_over_skips'] += 1
            continue
        
        # Reset player's tree and run MCTS
        player.reset()
        player.mct.search(board=board, n_sim=n_sims)
        
        # Extract trajectory pair
        pair = get_trajectory_pair(player.mct, board, cfg)
        
        if pair is None:
            if not player.mct.root or not player.mct.root.children:
                stats['no_children_skips'] += 1
            else:
                # Check if it was scenario 3 or short trajectory
                children_list = list(player.mct.root.children.values())
                if len(children_list) >= 2:
                    stats['scenario_3_filtered'] += 1
                else:
                    stats['short_trajectory_skips'] += 1
            continue
        
        pairs.append(pair)
        stats['accepted'] += 1
        
        if stats['accepted'] % 10 == 0:
            print(f"  Collected {stats['accepted']}/{n_pairs} pairs...")
    
    print(f"Trajectory collection stats: {stats}")
    return pairs