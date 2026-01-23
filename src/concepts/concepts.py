import numpy as np
from src.registries import CONCEPTS

@CONCEPTS.register("has_corner")
def has_corner(grid, player, board_size=6) -> bool:
    n = board_size
    corners = [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]
    return any(grid[r, c] == player for r, c in corners)
