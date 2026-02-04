"""Load Kaggle Othello board positions ready for AlphaZero player evaluation."""

import numpy as np
from src.datasets.datasets import load_kaggle_othello_games

cfg = {
    'board_size': 8,
    'n_positions': 8000,      # ~100 games × ~80 moves; loader caps at this
    'min_move_number': 4,     # skip opening
    'max_move_number': 60,    # include late game (standard Othello ~60 moves)
    'seed': 42,
}

positions = load_kaggle_othello_games(cfg)
print(f"Loaded {len(positions)} positions")

# Each position is a dict: {'grid': np.array(8,8), 'player': int, 'move_number': int}
# Grids are already flipped to match OthelloBoard convention.

# Convert to arrays for batch use
grids = np.stack([p['grid'] for p in positions])          # (N, 8, 8)
players = np.array([p['player'] for p in positions])      # (N,)
move_nums = np.array([p.get('move_number', -1) for p in positions])

# Player-relative inputs (what the network expects): input = player * grid
model_inputs = players[:, None, None] * grids             # (N, 8, 8)

print(f"grids shape:        {grids.shape}")
print(f"model_inputs shape: {model_inputs.shape}")
print(f"players unique:     {np.unique(players)}")
print(f"move range:         [{move_nums.min()}, {move_nums.max()}]")

# --- Example: feed one position to AlphaZero player ---
if True:  # set to True to run
    import torch
    from alphazero.games.othello import OthelloBoard, OthelloNet
    from alphazero.players import AlphaZeroPlayer

    net = OthelloNet(n=8)
    net.load_state_dict(torch.load("models/alphazero-othello/alphazero-othello.pt", map_location="cpu"))
    net.eval()

    player = AlphaZeroPlayer(n_sim=200, nn=net)

    # Reconstruct a board from a stored position and query the player
    pos = positions[0]
    board = OthelloBoard(n=8)
    board.grid = pos['grid'].copy()
    board.player = pos['player']

    move, pi, v, info = player.get_move(board)
    print(f"Suggested move: {move}, value: {v:.3f}")
