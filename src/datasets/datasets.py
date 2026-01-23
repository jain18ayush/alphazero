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