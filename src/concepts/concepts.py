import numpy as np
from src.registries import CONCEPTS, COUNTERFACTUALS

@CONCEPTS.register("has_corner")
def has_corner(grid, player, board_size=6) -> bool:
    corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]
    return any(grid[r, c] == player for r, c in corners)


@COUNTERFACTUALS.register("has_corner")
def remove_corner(pos: dict, board_size=6) -> dict:
    """Create counterfactual by removing one corner."""
    grid = pos['grid'].copy()
    player = pos['player']
    corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]
    
    for r, c in corners:
        if grid[r, c] == player:
            grid[r, c] = 0
            break
    
    return {**pos, 'grid': grid}
