import unittest
import importlib.util

HAS_AENUM = importlib.util.find_spec("aenum") is not None

if HAS_AENUM:
    from alphazero.games.othello import OthelloBoard
    from src.spine.spine import extract_rollout_contrasts


class DummyNode:
    def __init__(self, move=None, N=0, Q=0.0):
        self.move = move
        self.N = N
        self.Q = Q
        self.children = {}


class DummyMCT:
    def __init__(self, root):
        self.root = root


def _build_small_tree(value_gap=0.25, visit_gap_ratio=0.2):
    board = OthelloBoard(n=8)
    board.reset()

    root = DummyNode(move=None, N=200, Q=0.0)
    legal = sorted(board.get_moves())
    best_move = legal[0]
    alt_move = legal[1]

    best_n = 100
    alt_n = int(best_n * (1.0 - visit_gap_ratio))

    best = DummyNode(move=best_move, N=best_n, Q=0.8)
    alt = DummyNode(move=alt_move, N=alt_n, Q=0.8 - value_gap)
    root.children = {best_move: best, alt_move: alt}

    board_best = board.clone()
    board_best.play_move(best_move)
    best_legal = sorted(board_best.get_moves())
    if best_legal:
        best.children[best_legal[0]] = DummyNode(move=best_legal[0], N=80, Q=0.7)

    board_alt = board.clone()
    board_alt.play_move(alt_move)
    alt_legal = sorted(board_alt.get_moves())
    if alt_legal:
        alt.children[alt_legal[0]] = DummyNode(move=alt_legal[0], N=60, Q=0.6)

    return DummyMCT(root), board


@unittest.skipUnless(HAS_AENUM, "requires aenum/alphazero deps")
class TestExtractRolloutContrasts(unittest.TestCase):
    def test_extracts_optimal_and_eligible_subpar(self):
        mct, board = _build_small_tree(value_gap=0.25, visit_gap_ratio=0.05)
        out = extract_rollout_contrasts(
            mct,
            board,
            max_depth=3,
            min_value_gap=0.20,
            min_visit_gap_ratio=0.10,
            t_offset=5,
        )

        self.assertGreater(len(out["optimal_rollout"]), 0)
        self.assertEqual(out["required_depths"], 1)
        self.assertGreaterEqual(out["available_depths"], 1)
        self.assertEqual(out["subpar_rollouts"][0]["depth"], 0)

    def test_filters_ineligible_subpar(self):
        mct, board = _build_small_tree(value_gap=0.01, visit_gap_ratio=0.01)
        out = extract_rollout_contrasts(
            mct,
            board,
            max_depth=3,
            min_value_gap=0.20,
            min_visit_gap_ratio=0.10,
            t_offset=5,
        )

        self.assertEqual(out["required_depths"], 1)
        self.assertEqual(out["available_depths"], 0)
        self.assertEqual(len(out["subpar_rollouts"]), 0)


if __name__ == "__main__":
    unittest.main()
