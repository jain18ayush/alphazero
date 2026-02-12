"""
Interactive Streamlit quiz: can you guess AlphaZero's move?

Run:
    streamlit run quiz_app.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------

BOARD_COLOR = "#2e8b57"
PIECE_COLORS = {1: "black", -1: "white"}


def render_board(grid, player, legal_moves=None, highlight_move=None, board_size=8):
    """
    Draw an Othello board.

    Args:
        grid: nested list or np.array (board_size x board_size)
        player: 1 or -1
        legal_moves: list of [row, col] to mark with semi-transparent dots
        highlight_move: [row, col] to highlight in red (AZ's move)
        board_size: n
    Returns:
        matplotlib Figure
    """
    grid = np.asarray(grid, dtype=float)
    n = board_size

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_facecolor(BOARD_COLOR)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")

    # Grid lines
    for i in range(n + 1):
        ax.axhline(i - 0.5, color="black", linewidth=1)
        ax.axvline(i - 0.5, color="black", linewidth=1)

    # Pieces
    for r in range(n):
        for c in range(n):
            val = grid[r, c]
            if val == 1:
                ax.add_patch(plt.Circle((c, r), 0.38, color="black", zorder=2))
            elif val == -1:
                ax.add_patch(
                    plt.Circle(
                        (c, r), 0.38, color="white", edgecolor="black",
                        linewidth=0.5, zorder=2,
                    )
                )

    # Legal move dots
    if legal_moves:
        for m in legal_moves:
            r, c = m[0], m[1]
            ax.add_patch(
                plt.Circle(
                    (c, r), 0.15, color="yellow", alpha=0.5, zorder=3,
                )
            )

    # Highlight AZ move
    if highlight_move is not None:
        r, c = highlight_move[0], highlight_move[1]
        ax.add_patch(
            mpatches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=3, zorder=4,
            )
        )

    ax.set_xticks(range(n))
    ax.set_xticklabels(range(n))
    ax.set_yticks(range(n))
    ax.set_yticklabels(range(n))
    ax.tick_params(length=0)

    player_name = "Black" if player == 1 else "White"
    ax.set_title(f"{player_name} to move", fontsize=13)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spine tree rendering (adapted from spine_pipeline.py)
# ---------------------------------------------------------------------------

def render_spine_tree(spine_data, board_size=8, board_scale=1.0):
    """
    Render a spine decision tree from JSON-deserialized data.

    Returns a matplotlib Figure.
    """
    levels = spine_data["levels"]
    n = board_size
    max_depth = len(levels) - 1

    # Collect nodes, converting grids to np.array
    node_positions = {}
    for d, level in enumerate(levels):
        for idx, info in enumerate(level):
            if info is not None:
                info_copy = dict(info)
                info_copy["grid"] = np.asarray(info["grid"], dtype=float)
                if info_copy["move"] is not None and isinstance(info_copy["move"], list):
                    info_copy["move"] = tuple(info_copy["move"])
                node_positions[(d, idx)] = info_copy

    if not node_positions:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No spine data", ha="center", va="center")
        ax.axis("off")
        return fig

    # Lane assignment
    def lane_for(d, idx):
        if d == 0:
            return 0
        bits = f"{idx:0{d}b}"
        first_one = bits.find("1")
        return 0 if first_one == -1 else (first_one + 1)

    lane = {(d, idx): lane_for(d, idx) for (d, idx) in node_positions}
    max_lane = max(lane.values())

    # Layout params
    bs = 1.0 * board_scale
    meta_w = 1.25 * board_scale
    pad = 0.18 * board_scale
    card_w = bs + meta_w + 2 * pad
    card_h = bs + 2 * pad + 0.25 * board_scale

    lane_gap = 0.55 * board_scale
    depth_gap = 0.65 * board_scale
    x_step = card_w + lane_gap
    y_step = card_h + depth_gap

    xy = {}
    for (d, idx) in node_positions:
        L = lane[(d, idx)]
        xy[(d, idx)] = (L * x_step, d * y_step)

    fig_w = max(8, (max_lane + 1) * (2.2 * board_scale))
    fig_h = max(8, (max_depth + 1) * (1.75 * board_scale))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-x_step * 0.6, (max_lane + 1) * x_step - x_step * 0.4)
    ax.set_ylim(-y_step * 0.6, (max_depth + 1) * y_step - y_step * 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()

    # Drawing helpers
    def draw_card(cx, cy, info, is_root=False):
        grid = info["grid"]
        move = info["move"]
        label = info["label"]
        n_ch = info["n_children"]
        N_val = info.get("N")
        Q_val = info.get("Q")

        if is_root:
            bg, accent = "#e3f2fd", "#1565c0"
        elif label == "best":
            bg, accent = "#e8f5e9", "#2e7d32"
        else:
            bg, accent = "#fff3e0", "#ef6c00"

        left = cx - card_w / 2
        top = cy - card_h / 2

        ax.add_patch(
            mpatches.FancyBboxPatch(
                (left, top), card_w, card_h,
                boxstyle="round,pad=0.03",
                facecolor=bg, edgecolor="#777777", linewidth=0.9,
            )
        )

        if is_root:
            title = f"ROOT  |  children={n_ch}"
        else:
            title = f"{label.upper():>4}  move={move}  |  children={n_ch}"
        ax.text(
            left + pad, top + 0.18 * board_scale, title,
            ha="left", va="center",
            fontsize=8 * board_scale, fontweight="bold", color=accent,
        )

        board_y = top + 0.32 * board_scale + pad
        board_x = left + pad

        ax.add_patch(
            mpatches.Rectangle(
                (board_x, board_y), bs, bs,
                facecolor=BOARD_COLOR, edgecolor="black", linewidth=0.8,
            )
        )

        cell = bs / n
        for i in range(1, n):
            ax.plot([board_x + i * cell, board_x + i * cell],
                    [board_y, board_y + bs], "k-", lw=0.35)
            ax.plot([board_x, board_x + bs],
                    [board_y + i * cell, board_y + i * cell], "k-", lw=0.35)

        r = cell * 0.38
        for row in range(n):
            for col in range(n):
                v = grid[row, col]
                if v != 0:
                    px = board_x + col * cell + cell / 2
                    py = board_y + (n - 1 - row) * cell + cell / 2
                    fc = "black" if v == 1 else "white"
                    ax.add_patch(
                        plt.Circle((px, py), r, facecolor=fc,
                                   edgecolor="black", linewidth=0.35)
                    )
                if move is not None and move == (row, col):
                    hx = board_x + col * cell
                    hy = board_y + (n - 1 - row) * cell
                    ax.add_patch(
                        mpatches.Rectangle(
                            (hx, hy), cell, cell,
                            fill=False, edgecolor="red", linewidth=1.8,
                        )
                    )

        meta_left = board_x + bs + pad
        meta_top = board_y + 0.02 * board_scale
        lines = []
        if N_val is not None:
            lines.append(f"N: {N_val:.0f}")
        if Q_val is not None:
            lines.append(f"Q: {Q_val:+.3f}")
        ax.text(
            meta_left, meta_top, "\n".join(lines) if lines else "(no stats)",
            ha="left", va="top",
            fontsize=8 * board_scale, family="monospace", color="#222222",
        )

        nid = info.get("id")
        if nid is not None:
            ax.text(
                left + card_w - pad * 0.6, top + card_h - pad * 0.6,
                f"idx={nid}", ha="right", va="bottom",
                fontsize=7 * board_scale, family="monospace", color="#444444",
            )

    def draw_arrow(p_xy, c_xy, is_best):
        px, py = p_xy
        cx_a, cy_a = c_xy
        start = (px, py + card_h * 0.45)
        end = (cx_a, cy_a - card_h * 0.45)
        color = "#2e7d32" if is_best else "#ef6c00"
        lw = 2.0 if is_best else 1.6
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0),
        )

    # Draw edges
    for (d, idx) in node_positions:
        if d >= max_depth:
            continue
        for k, child_idx in enumerate([2 * idx, 2 * idx + 1]):
            child = (d + 1, child_idx)
            if child in node_positions:
                draw_arrow(xy[(d, idx)], xy[child], is_best=(k == 0))

    # Draw nodes
    for (d, idx), info in node_positions.items():
        draw_card(*xy[(d, idx)], info, is_root=(d == 0))

    # Lane labels
    ax.text(
        0, -0.35 * y_step, "Lane 0: spine (best-only)",
        ha="left", va="center",
        fontsize=9 * board_scale, color="#2e7d32", fontweight="bold",
    )
    for L in range(1, max_lane + 1):
        ax.text(
            L * x_step, -0.35 * y_step,
            f"Lane {L}: first 2nd-best at depth {L}",
            ha="center", va="center",
            fontsize=8 * board_scale, color="#ef6c00",
        )

    ax.set_title(
        "AlphaZero's Reasoning: Spine Tree (best + 2nd-best branches)",
        fontsize=12, pad=18,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

def load_quiz_data(path):
    with open(path) as f:
        return json.load(f)


def format_move(m):
    return f"({m[0]}, {m[1]})"


def main():
    st.set_page_config(page_title="AlphaZero Move Quiz", layout="wide")
    st.title("AlphaZero Move Quiz")
    st.markdown("Can you guess the move AlphaZero would make?")

    # -- Sidebar --
    st.sidebar.header("Settings")
    data_path = st.sidebar.text_input("Quiz data file", value="quiz_data.json")

    # Load data
    if "quiz_data" not in st.session_state or st.session_state.get("_data_path") != data_path:
        try:
            st.session_state.quiz_data = load_quiz_data(data_path)
            st.session_state._data_path = data_path
        except FileNotFoundError:
            st.error(f"File not found: {data_path}")
            st.info("Generate quiz data first: `python generate_quiz_data.py --config configs/quiz.yaml`")
            return

    data = st.session_state.quiz_data
    items = data["items"]
    config = data["config"]

    if not items:
        st.warning("No quiz items found in the data file.")
        return

    # Session state defaults
    for key, default in [
        ("current_idx", 0),
        ("answered", False),
        ("selected_move", None),
        ("score", 0),
        ("total_attempted", 0),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Sidebar navigation
    st.sidebar.markdown("---")
    n_items = len(items)
    new_idx = st.sidebar.number_input(
        "Position", min_value=0, max_value=n_items - 1,
        value=st.session_state.current_idx, step=1,
    )
    if new_idx != st.session_state.current_idx:
        st.session_state.current_idx = new_idx
        st.session_state.answered = False
        st.session_state.selected_move = None
        st.rerun()

    # Score display
    st.sidebar.markdown("---")
    st.sidebar.metric("Score", f"{st.session_state.score} / {st.session_state.total_attempted}")

    # Config info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Config**")
    st.sidebar.text(f"Model: {config['model_name']}")
    st.sidebar.text(f"MCTS sims: {config['n_sims']}")
    st.sidebar.text(f"Spine depth: {config['spine_depth']}")

    # -- Main area --
    item = items[st.session_state.current_idx]
    pos = item["position"]
    grid = pos["grid"]
    player = pos["player"]
    legal_moves = item["legal_moves"]
    az_move = item["az_move"]
    board_size = config.get("board_size", 8)

    st.markdown(f"**Position {st.session_state.current_idx + 1} / {n_items}**  "
                f"(source idx: {item['source_idx']}, move #{pos['move_number']})")

    if not st.session_state.answered:
        # --- Before answering ---
        col_board, col_select = st.columns([2, 1])

        with col_board:
            fig = render_board(grid, player, legal_moves=legal_moves, board_size=board_size)
            st.pyplot(fig)
            plt.close(fig)

        with col_select:
            st.markdown("### Your guess")
            move_options = [format_move(m) for m in legal_moves]
            choice = st.selectbox("Pick a move (row, col):", move_options)

            if st.button("Submit", type="primary"):
                # Parse selected move
                selected = legal_moves[move_options.index(choice)]
                correct = (selected[0] == az_move[0] and selected[1] == az_move[1])

                st.session_state.selected_move = selected
                st.session_state.answered = True
                st.session_state.total_attempted += 1
                if correct:
                    st.session_state.score += 1
                st.rerun()

    else:
        # --- After answering ---
        selected = st.session_state.selected_move
        correct = (selected[0] == az_move[0] and selected[1] == az_move[1])

        if correct:
            st.success(f"Correct! AlphaZero also chose {format_move(az_move)}.")
        else:
            st.error(
                f"Incorrect. You chose {format_move(selected)}, "
                f"but AlphaZero chose {format_move(az_move)}."
            )

        # Board with AZ move highlighted
        col_board, col_stats = st.columns([2, 1])

        with col_board:
            fig = render_board(
                grid, player,
                legal_moves=legal_moves, highlight_move=az_move,
                board_size=board_size,
            )
            st.pyplot(fig)
            plt.close(fig)

        with col_stats:
            st.markdown("### Top moves (by MCTS visits)")
            top_moves = item["move_stats"][:8]
            for ms in top_moves:
                m = ms["move"]
                marker = " **<--**" if m[0] == az_move[0] and m[1] == az_move[1] else ""
                st.text(f"({m[0]},{m[1]})  N={ms['N']:>6d}  Q={ms['Q']:+.3f}{marker}")

        # Spine tree
        st.markdown("---")
        st.markdown("### AlphaZero's Reasoning (Spine Tree)")
        st.markdown(
            "The tree shows AlphaZero's best move (green) and second-best alternative "
            "(orange) at each depth. Nodes show visit count (N) and value estimate (Q)."
        )
        spine_fig = render_spine_tree(
            item["spine_tree"], board_size=board_size, board_scale=1.2,
        )
        st.pyplot(spine_fig)
        plt.close(spine_fig)

        # Next button
        st.markdown("---")
        if st.button("Next Position", type="primary"):
            st.session_state.current_idx = (st.session_state.current_idx + 1) % n_items
            st.session_state.answered = False
            st.session_state.selected_move = None
            st.rerun()


if __name__ == "__main__":
    main()
