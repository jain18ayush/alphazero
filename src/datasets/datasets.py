from alphazero.arena import Arena
from alphazero.players import RandomPlayer, MCTSPlayer, AlphaZeroPlayer
from alphazero.games.othello import OthelloBoard

from src.registries import DATASETS
import numpy as np
from tqdm import trange
import uuid as uuid_lib
from pathlib import Path

@DATASETS.register("batched_load_npy")
def batched_load_npy(cfg):
    """
    Load a batch of .npy files and concatenate them into a single array.
    """
    folder = Path(cfg["folder"])
    arrays = [np.load(p, allow_pickle=True) for p in sorted(folder.glob("*.npy"))]
    return np.concatenate(arrays, axis=0)

@DATASETS.register("model_play")
def generate_model_games(cfg):
    """
    Generate games using a neural network model and collect all positions.

    Config:
        board_size: int - board size (e.g., 8 for standard Othello)
        n_games: int - number of games to generate
        net: PolicyValueNetwork - the neural network to use for move selection
        n_sims: int - number of MCTS simulations per move (default 50)
        temp: float - temperature for move selection (default 1.0, higher = more exploration)
        min_move_number: int - skip positions before this move (default 0)
        max_move_number: int - skip positions after this move (default 100)
        seed: int - random seed (default 42)

    Returns:
        List of position dicts with 'grid', 'player', 'move_number'
    """
    from alphazero.games.othello import OthelloBoard, OthelloNet

    board_size = cfg['board_size']
    n_games = cfg['n_games']
    n_sims = cfg.get('n_sims', 50)
    temp = cfg.get('temp', 1.0)
    min_move = cfg.get('min_move_number', 0)
    max_move = cfg.get('max_move_number', 100)
    seed = cfg.get('seed', 42)
    save_path = cfg.get("save_path", "az_games")          # e.g. "positions.npy"
    save_every = cfg.get("save_every")  # save every N positions

    import os
    import numpy as np

    positions = []

    def maybe_save():
        nonlocal positions

        if save_every is None:
            return

        if len(positions) < save_every:
            return

        batch = positions[-save_every:]      # only last batch
        out = f"/Users/ayushjain/Development/Research/alphazero/batches/{save_path}_{uuid_lib.uuid4()}.npy"

        print(f"Saving {len(batch)} positions → {out}")

        np.save(out, np.array(batch, dtype=object), allow_pickle=True)

              # keep memory bounded

    # Get the network - must be provided
    if 'net' not in cfg or cfg['net'] is None:
        raise ValueError("model_play requires 'net' in config - pass a loaded neural network")

    net = cfg['net']

    # Create AlphaZero players using the network
    p1 = AlphaZeroPlayer(n_sim=n_sims, nn=net)
    p2 = AlphaZeroPlayer(n_sim=n_sims, nn=net)

    rng = np.random.RandomState(seed)

    for game_idx in trange(n_games, desc="Generating games"):
        board = OthelloBoard(n=board_size)
        board.reset()
        p1.reset()
        p2.reset()

        move_number = 0

        while not board.is_game_over():
            print(f"Game {game_idx + 1}, Move {move_number + 1}")
            
            # Record position if within range
            if move_number > max_move: 
                break 

            if min_move <= move_number <= max_move:
                positions.append({
                    'grid': board.grid.copy(),
                    'player': board.player,
                    'move_number': move_number
                })

                if save_every is not None and len(positions) % save_every == 0:
                    maybe_save()


            # Get move from the appropriate player
            if board.player == 1:
                move, _, _, _ = p1.get_move(board, temp=temp)
                p1.apply_move(move, player=board.player)
                p2.apply_move(move, player=board.player)
            else:
                move, _, _, _ = p2.get_move(board, temp=temp)
                p1.apply_move(move, player=board.player)
                p2.apply_move(move, player=board.player)

            board.play_move(move)
            move_number += 1

        if (game_idx + 1) % 10 == 0:
            print(f"  Generated {game_idx + 1}/{n_games} games, {len(positions)} positions...")

    maybe_save()

    print(f"Model play: collected {len(positions)} positions from {n_games} games")
    return positions


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


def follow_trajectory(node, board, depth: int) -> List[Dict]:
    """
    Follow most-visited path down from node, collect board states.

    Args:
        node: Starting node (root of subtree)
        board: Board at the starting node position (will be copied and modified)
        depth: Maximum depth to follow

    Returns:
        List of dicts with 'grid' and 'player' for each state
    """
    from alphazero.games.othello import OthelloBoard

    states = []
    current = node

    # Make a copy of the board so we don't modify the original
    board_copy = OthelloBoard(n=board.n)
    board_copy.grid = board.grid.copy()
    board_copy.player = board.player

    for _ in range(depth):
        if current is None:
            break

        # Store current state
        states.append({
            'grid': board_copy.grid.copy(),
            'player': board_copy.player,
        })

        # Move to most-visited child
        if not current.children:
            break

        best_child = max(current.children.values(), key=lambda c: c.N)

        # Apply the move that leads to best_child
        if best_child.move is not None:
            board_copy.play_move(best_child.move)

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
    children_list.sort(key=lambda c: c.N, reverse=True)
    
    if len(children_list) < 2:
        return None
    
    best = children_list[0]
    second = children_list[1]
    
    # Get values (Q-values from MCTS)
    # Q is already the average win rate (normalized)
    best_value = best.Q
    second_value = second.Q

    # Scenario 3 filter: require meaningful gap
    value_gap = abs(best_value - second_value)
    visit_ratio = best.N / (second.N + 1e-8)
    
    min_value_gap = cfg.get('min_value_gap', 0.20)
    min_visit_ratio = cfg.get('min_visit_ratio', 1.10)
    
    if value_gap < min_value_gap and visit_ratio < min_visit_ratio:
        return None
    
    depth = cfg.get('depth', 5)

    # Extract trajectories by replaying moves from root board
    # We need to create separate board copies for each trajectory
    from alphazero.games.othello import OthelloBoard

    # For chosen trajectory: start from root, apply best move
    board_chosen = OthelloBoard(n=board.n)
    board_chosen.grid = board.grid.copy()
    board_chosen.player = board.player
    board_chosen.play_move(best.move)
    chosen_states = follow_trajectory(best, board_chosen, depth)

    # For rejected trajectory: start from root, apply second-best move
    board_rejected = OthelloBoard(n=board.n)
    board_rejected.grid = board.grid.copy()
    board_rejected.player = board.player
    board_rejected.play_move(second.move)
    rejected_states = follow_trajectory(second, board_rejected, depth)
    
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
        'best_visits': int(best.N),
        'second_visits': int(second.N),
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
        net: Optional - pre-loaded model (preferred). If not provided, will load from:
            checkpoint_path: str - path to model checkpoint
            model_name: str - name of model file
    """
    from alphazero.games.othello import OthelloBoard, OthelloNet
    from alphazero.players import AlphaZeroPlayer
    
    board_size = cfg['board_size']
    n_pairs = cfg.get('n_pairs', 50)
    max_attempts = cfg.get('max_attempts', n_pairs * 10)
    n_sims = cfg.get('n_sims', 200)
    
    # Use pre-loaded model if provided, otherwise load it (backward compatibility)
    if 'net' in cfg and cfg['net'] is not None:
        net = cfg['net']
    else:
        # Fallback: load model from config (for backward compatibility)
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


@DATASETS.register("kaggle_othello_games")
def load_kaggle_othello_games(cfg):
    """
    Load positions from Kaggle Othello games dataset.
    Dataset: andrefpoliveira/othello-games
    
    Config:
        board_size: int (must be 8 for standard Othello)
        n_positions: int - target number of positions
        min_move_number: int - skip early game positions
        max_move_number: int - skip late game positions
        seed: int
    """
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError:
        raise ImportError("Please install kagglehub: pip install kagglehub")
    
    from alphazero.games.othello import OthelloBoard
    
    board_size = cfg.get('board_size', 8)
    if board_size != 8:
        print(f"Warning: Kaggle dataset uses standard 8x8 Othello, but config specifies {board_size}x{board_size}")
    
    n_positions = cfg.get('n_positions', 17184)
    min_move = cfg.get('min_move_number', 8)  # Skip opening (consistent with paper)
    max_move = cfg.get('max_move_number', 50)  # Skip very late game
    seed = cfg.get('seed', 42)
    
    print(f"Loading Kaggle Othello dataset...")
    
    # Download the dataset (cached locally after first download)
    import os
    import pandas as pd
    dataset_path = kagglehub.dataset_download("andrefpoliveira/othello-games")
    files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No CSV files found in dataset. Files: {os.listdir(dataset_path)}")
    csv_file = files[0]  # Use the first CSV file found
    
    # Load the CSV file directly using pandas since we already have the path
    csv_path = os.path.join(dataset_path, csv_file)
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} games from Kaggle")

    # Parse games and extract positions
    positions = []
    rng = np.random.RandomState(seed)

    # Statistics
    stats = {
        'total_games': 0,
        'no_moves_column': 0,
        'too_short': 0,
        'parse_failed': 0,
        'no_positions_in_range': 0,
        'success': 0,
    }

    # Shuffle games to get diverse positions
    game_indices = rng.permutation(len(df))

    for idx in game_indices:
        if len(positions) >= n_positions:
            break

        stats['total_games'] += 1
        row = df.iloc[idx]

        # Find game moves column
        game_moves = None
        if 'game_moves' in row:
            game_moves = row['game_moves']
        elif 'moves' in row:
            game_moves = row['moves']
        elif 'game' in row:
            game_moves = row['game']
        elif 'transcript' in row:
            game_moves = row['transcript']
        else:
            if idx == game_indices[0]:
                print(f"Available columns: {list(df.columns)}")
            stats['no_moves_column'] += 1
            continue

        if game_moves is None or len(str(game_moves)) < 8:
            stats['too_short'] += 1
            continue

        # Parse game moves
        try:
            game_positions = parse_othello_game(
                str(game_moves),
                board_size=board_size,
                min_move=min_move,
                max_move=max_move
            )

            if len(game_positions) == 0:
                stats['no_positions_in_range'] += 1
            else:
                stats['success'] += 1
                positions.extend(game_positions)

        except Exception as e:
            stats['parse_failed'] += 1
            if stats['parse_failed'] <= 5:  # Print first 5 errors
                print(f"Warning: Failed to parse game {idx}: {e}")
            continue

        if stats['total_games'] % 5000 == 0:
            print(f"  Processed {stats['total_games']} games, collected {len(positions)} positions...")

    print(f"\nCollection statistics:")
    print(f"  Total games processed: {stats['total_games']}")
    print(f"  Successful parses: {stats['success']} ({100*stats['success']/max(stats['total_games'],1):.1f}%)")
    print(f"  No positions in range [{min_move},{max_move}]: {stats['no_positions_in_range']}")
    print(f"  Parse failures: {stats['parse_failed']}")
    print(f"  Too short: {stats['too_short']}")
    print(f"  Total positions collected: {len(positions)}")
    
    # Subsample if we got too many
    if len(positions) > n_positions:
        indices = rng.choice(len(positions), n_positions, replace=False)
        positions = [positions[i] for i in indices]
    
    return positions


def parse_othello_game(moves_str: str, board_size: int = 8,
                       min_move: int = 4, max_move: int = 50) -> list:
    """
    Parse an Othello game string and extract board positions.

    The Kaggle dataset uses standard Othello starting position:
        d4=White(-1), e4=Black(1), d5=Black(1), e5=White(-1)

    OthelloBoard uses a non-standard starting position with colors swapped:
        d4=Black(1), e4=White(-1), d5=White(-1), e5=Black(1)

    This function simulates the game with standard rules and then flips
    all colors to match OthelloBoard's convention.

    Args:
        moves_str: String of moves in standard notation (e.g., "f5d6c4...")
        board_size: Board size (should be 8 for standard)
        min_move: Minimum move number to include
        max_move: Maximum move number to include

    Returns:
        List of position dicts compatible with OthelloBoard
    """
    positions = []

    # Initialize standard Othello board
    grid = np.zeros((board_size, board_size), dtype=np.float32)
    mid = board_size // 2
    # Standard starting position
    grid[mid - 1, mid - 1] = -1  # d4 = White
    grid[mid - 1, mid] = 1       # e4 = Black
    grid[mid, mid - 1] = 1        # d5 = Black
    grid[mid, mid] = -1           # e5 = White

    player = 1  # Black moves first in standard Othello
    moves_str = moves_str.strip().lower()

    move_num = 0
    i = 0

    while i < len(moves_str) - 1:
        # Get next move (2 characters)
        move_notation = moves_str[i:i+2]
        i += 2

        # Handle pass moves
        if move_notation in ['pa', '--', 'ps']:
            player = -player
            move_num += 1
            continue

        # Parse algebraic notation (e.g., "f5" -> row=4, col=5 in 0-indexed)
        try:
            col = ord(move_notation[0]) - ord('a')  # a=0, b=1, ..., h=7
            row = int(move_notation[1]) - 1  # 1=0, 2=1, ..., 8=7

            if row < 0 or row >= board_size or col < 0 or col >= board_size:
                break

            # Record position if in range (BEFORE playing the move)
            # Flip colors to match OthelloBoard convention
            if min_move <= move_num <= max_move:
                positions.append({
                    'grid': -grid.copy(),  # Flip all colors
                    'player': -player,      # Flip player
                    'move_number': move_num,
                })

            # Simulate the move with standard Othello rules
            if not _play_standard_othello_move(grid, row, col, player):
                # Invalid move, skip this game
                break

            # Switch player
            player = -player
            move_num += 1

        except (ValueError, IndexError):
            break

    return positions


def _play_standard_othello_move(grid: np.ndarray, row: int, col: int, player: int) -> bool:
    """
    Play a move in standard Othello. Returns True if valid, False otherwise.
    Modifies grid in-place.
    """
    if grid[row, col] != 0:
        return False  # Square occupied

    board_size = grid.shape[0]
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1)]

    flips = []

    # Check all 8 directions
    for dr, dc in directions:
        temp_flips = []
        r, c = row + dr, col + dc

        # Look for opponent pieces followed by player piece
        while 0 <= r < board_size and 0 <= c < board_size:
            if grid[r, c] == 0:
                break  # Empty square
            elif grid[r, c] == player:
                # Found our piece - these flips are valid
                flips.extend(temp_flips)
                break
            else:
                # Opponent piece - could be flipped
                temp_flips.append((r, c))
                r, c = r + dr, c + dc

    if not flips:
        return False  # No valid flips, invalid move

    # Place piece and flip
    grid[row, col] = player
    for r, c in flips:
        grid[r, c] = player

    return True


# Also create a convenience function that auto-detects best human source
@DATASETS.register("human_games")
def load_human_games(cfg):
    """
    Load human game positions. Auto-detects best available source.
    
    Priority:
    1. Kaggle Othello dataset (if available)
    2. Self-play as fallback
    """
    # Try Kaggle first
    try:
        import kagglehub
        print("Using Kaggle Othello games dataset for human baseline")
        return load_kaggle_othello_games(cfg)
    except (ImportError, Exception) as e:
        print(f"Kaggle dataset not available ({e}), using self-play as baseline")
        print("Warning: This is not ideal - install kagglehub for better results")
        
        # Fallback to self-play
        from alphazero.games.othello import OthelloBoard
        from alphazero.players import RandomPlayer
        
        board_size = cfg['board_size']
        n_positions = cfg.get('n_positions', 17184)
        
        positions = []
        p1 = RandomPlayer()
        p2 = RandomPlayer()
        
        while len(positions) < n_positions:
            board = OthelloBoard(n=board_size)
            
            while not board.is_game_over() and len(positions) < n_positions:
                positions.append({
                    'grid': board.grid.copy(),
                    'player': board.player,
                })
                
                move, _, _, _ = (p1 if board.player == 1 else p2).get_move(board)
                board.play_move(move)
        
        return positions[:n_positions]


@DATASETS.register("human_games_npy")
def load_human_games_npy(cfg):
    """
    Load human game positions from a local .npy file.

    Config:
        path: str - path to .npy file (array of dicts)
        n_positions: int - optional cap on number of positions
        min_move_number: int - optional min move filter (inclusive)
        max_move_number: int - optional max move filter (inclusive)
    """
    path = cfg["path"]
    n_positions = cfg.get("n_positions")
    min_move = cfg.get("min_move_number")
    max_move = cfg.get("max_move_number")

    data = np.load(path, allow_pickle=True)
    positions = list(data)

    if min_move is not None or max_move is not None:
        filtered = []
        for pos in positions:
            move_number = pos.get("move_number")
            if move_number is None:
                filtered.append(pos)
                continue
            if min_move is not None and move_number < min_move:
                continue
            if max_move is not None and move_number > max_move:
                continue
            filtered.append(pos)
        positions = filtered

    if n_positions is not None:
        positions = positions[:n_positions]

    return positions


@DATASETS.register("concept_vectors_from_runs")
def load_concept_vectors_from_runs(cfg):
    """
    Load concept vectors from previous pipeline run artifacts.

    Config:
        run_root: str
        layers: Optional[list[str]]
        max_concepts: Optional[int]
    """
    run_root = Path(cfg["run_root"])
    layers = cfg.get("layers")
    max_concepts = cfg.get("max_concepts")

    concepts = []
    for path in sorted(run_root.rglob("concept_vector.npy")):
        layer_dir = path.parent.name
        if not layer_dir.startswith("layer="):
            continue

        layer = layer_dir.split("layer=", 1)[1].replace("_", "/")
        if layers is not None and layer not in layers:
            continue

        concepts.append({
            "layer": layer,
            "vector": np.load(path),
            "path": str(path),
        })

    if max_concepts is not None:
        concepts = concepts[:max_concepts]

    return concepts
