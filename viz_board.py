"""Visualize an Othello board grid using matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def viz_board(grid, title=None, ax=None, save_path=None):
    """
    Visualize an Othello board from a grid array.

    Args:
        grid: (8, 8) numpy array. 1 = black, -1 = white, 0 = empty.
        title: Optional title string.
        ax: Optional matplotlib Axes. Creates a new figure if None.
        save_path: Optional path to save the figure.
    """
    n = grid.shape[0]
    show = ax is None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Draw green board
    ax.set_facecolor('#2e8b57')
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect('equal')

    # Grid lines
    for i in range(n + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
        ax.axvline(i - 0.5, color='black', linewidth=1)

    # Pieces
    for r in range(n):
        for c in range(n):
            val = grid[r, c]
            if val == 1:
                circle = plt.Circle((c, r), 0.38, color='black', zorder=2)
                ax.add_patch(circle)
            elif val == -1:
                circle = plt.Circle((c, r), 0.38, color='white', edgecolor='black',
                                    linewidth=0.5, zorder=2)
                ax.add_patch(circle)

    # Labels
    cols = 'abcdefgh'[:n]
    ax.set_xticks(range(n))
    ax.set_xticklabels(range(0, n))
    ax.set_yticks(range(n))
    ax.set_yticklabels(range(n-1, -1, -1))
    ax.tick_params(length=0)

    if title:
        ax.set_title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to {save_path}")

    if show:
        plt.show()


if __name__ == '__main__':
    grid = np.array([
        [-0., -0., -0., -0.,  1., -0., -0., -0.],
        [-0., -0., -0., -0.,  1.,  1., -0., -0.],
        [-0., -0., -0., -1.,  1., -1.,  1., -0.],
        [-0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [-0., -0., -1., -1.,  1., -1.,  1.,  1.],
        [-0., -0., -1., -1., -1.,  1., -0.,  1.],
        [-0., -0., -0., -0., -0., -0., -0.,  1.],
        [-0., -0., -0., -0., -0., -0., -0., -0.],
    ], dtype=np.float32)

    viz_board(grid, title="Othello Position")
