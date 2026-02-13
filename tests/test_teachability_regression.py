"""Regression tests for the teachability pipeline.

These tests exercise the real computation (no mocking of expensive functions
like mcts_policy_for_positions or _score_rollout_states) to catch if
batching/caching changes break correctness.

Where MCTS is too expensive, only the MCTS tree-building is mocked — the
neural-network forward passes and scoring logic run unmodified.
"""
import copy
import importlib.util
import os
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_AENUM = importlib.util.find_spec("aenum") is not None

if HAS_TORCH and HAS_AENUM:
    import torch

    from alphazero.games.othello import OthelloBoard, OthelloNet
    from src.hooks.extract import get_single_activation
    from src.teachability.teachability import (
        _board_from_pos,
        _score_rollout_states,
        distill_student,
        dynamic_prototypes_for_concept,
        mcts_policy_for_positions,
        measure_top1_agreement,
        run_teachability_benchmark,
    )
    from teachability_pipeline import (
        _sample_random_positions,
        _split_positions,
        _subsample_positions,
    )

BOARD_SIZE = 6
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE + 1  # 37
LAYER = "fc2"  # output dim = 512
LAYER_DIM = 512


def _make_othello_positions(n_positions=10, board_size=BOARD_SIZE, seed=42):
    """Generate valid (non-terminal) Othello positions by playing random moves."""
    rng = np.random.RandomState(seed)
    positions = []
    attempts = 0
    while len(positions) < n_positions and attempts < n_positions * 10:
        attempts += 1
        board = OthelloBoard(n=board_size)
        n_moves = rng.randint(1, 8)
        for _ in range(n_moves):
            if board.is_game_over():
                break
            moves = board.get_moves()
            move = moves[rng.randint(len(moves))]
            board.play_move(move)
        if not board.is_game_over():
            positions.append({"grid": board.grid.copy(), "player": board.player})
    return positions


def _make_net(seed=0):
    """Create a deterministic OthelloNet(n=6) in eval mode."""
    torch.manual_seed(seed)
    net = OthelloNet(n=BOARD_SIZE)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Test 1: _score_rollout_states determinism
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH and HAS_AENUM, "requires torch + aenum deps")
class TestScoreRolloutStatesDeterminism(unittest.TestCase):

    def test_identical_on_repeated_calls(self):
        net = _make_net(seed=0)
        states = _make_othello_positions(5, seed=10)
        v = np.random.RandomState(99).randn(LAYER_DIM).astype(np.float32)

        scores1 = _score_rollout_states(net, states, LAYER, v)
        scores2 = _score_rollout_states(net, states, LAYER, v)

        self.assertEqual(len(scores1), len(states))
        for s1, s2 in zip(scores1, scores2):
            self.assertEqual(s1, s2)


# ---------------------------------------------------------------------------
# Test 2: _board_from_pos roundtrip
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH and HAS_AENUM, "requires torch + aenum deps")
class TestBoardFromPosRoundtrip(unittest.TestCase):

    def test_roundtrip(self):
        board = OthelloBoard(n=BOARD_SIZE)
        moves = board.get_moves()
        board.play_move(moves[0])

        pos = {"grid": board.grid.copy(), "player": board.player}
        reconstructed = _board_from_pos(pos)

        np.testing.assert_array_equal(reconstructed.grid, pos["grid"])
        self.assertEqual(reconstructed.player, pos["player"])
        self.assertEqual(reconstructed.n, BOARD_SIZE)

    def test_does_not_alias_grid(self):
        grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)
        pos = {"grid": grid, "player": 1}
        board = _board_from_pos(pos)
        board.grid[0, 0] = 99
        self.assertEqual(pos["grid"][0, 0], 0.0)


# ---------------------------------------------------------------------------
# Test 3: mcts_policy_for_positions output structure
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH and HAS_AENUM, "requires torch + aenum deps")
class TestMctsPolicyOutputStructure(unittest.TestCase):

    def test_output_shapes_and_validity(self):
        net = _make_net(seed=1)
        positions = _make_othello_positions(3, seed=20)

        policies, top1_indices = mcts_policy_for_positions(net, positions, n_sim=8)

        self.assertEqual(len(policies), len(positions))
        self.assertEqual(len(top1_indices), len(positions))

        for pi in policies:
            self.assertEqual(pi.shape, (ACTION_SIZE,))
            self.assertAlmostEqual(float(np.sum(pi)), 1.0, places=4)
            self.assertTrue(np.all(pi >= 0))

        for idx in top1_indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, ACTION_SIZE)


# ---------------------------------------------------------------------------
# Test 4: measure_top1_agreement with identical models
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH and HAS_AENUM, "requires torch + aenum deps")
class TestMeasureAgreementIdenticalModels(unittest.TestCase):

    def test_self_agreement_is_one(self):
        """Same model + no Dirichlet noise → deterministic MCTS → agreement = 1.0."""
        net = _make_net(seed=2)
        positions = _make_othello_positions(4, seed=30)

        matches, agreement = measure_top1_agreement(
            teacher=net, student=net, positions=positions, n_sim=8,
        )

        self.assertEqual(matches, len(positions))
        self.assertAlmostEqual(agreement, 1.0)


# ---------------------------------------------------------------------------
# Test 5: distill_student convergence with real OthelloNet
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH and HAS_AENUM, "requires torch + aenum deps")
class TestDistillStudentWithRealNet(unittest.TestCase):

    def test_loss_decreases(self):
        torch.manual_seed(5)
        student = OthelloNet(n=BOARD_SIZE)
        positions = _make_othello_positions(16, seed=40)

        target_pi = np.full(ACTION_SIZE, 1.0 / ACTION_SIZE, dtype=np.float32)
        target_pis = np.tile(target_pi, (len(positions), 1))

        losses = distill_student(
            student, positions, target_pis,
            epochs=5, lr=1e-3, batch_size=8,
        )

        self.assertGreater(len(losses), 1)
        early = np.mean(losses[:3])
        late = np.mean(losses[-3:])
        self.assertLess(late, early)


# ---------------------------------------------------------------------------
# Test 6: run_teachability_benchmark end-to-end (small scale)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH and HAS_AENUM, "requires torch + aenum deps")
class TestRunTeachabilityBenchmarkEndToEnd(unittest.TestCase):

    def test_end_to_end(self):
        torch.manual_seed(6)
        teacher = OthelloNet(n=BOARD_SIZE)
        teacher.eval()
        teacher_weights_before = copy.deepcopy(teacher.state_dict())

        template_state = copy.deepcopy(OthelloNet(n=BOARD_SIZE).state_dict())

        def load_student():
            torch.manual_seed(99)
            s = OthelloNet(n=BOARD_SIZE)
            s.load_state_dict(copy.deepcopy(template_state))
            s.eval()
            return s

        positions = _make_othello_positions(12, seed=50)
        X_train_concept = positions[:8]
        X_test_concept = positions[8:]
        X_train_random = _make_othello_positions(8, seed=51)
        X_test_random = _make_othello_positions(4, seed=52)

        out = run_teachability_benchmark(
            load_student=load_student,
            teacher=teacher,
            X_train_concept=X_train_concept,
            X_test_concept=X_test_concept,
            X_train_random=X_train_random,
            X_test_random=X_test_random,
            n_sim=8,
            temp=1.0,
            epochs=2,
            lr=1e-3,
            batch_size=4,
        )

        expected_keys = [
            "baseline_eval_C", "train_C_eval_C", "train_C_eval_R",
            "train_R_eval_C", "train_R_eval_R",
            "loss_tail_concept", "loss_tail_random",
        ]
        for k in expected_keys:
            self.assertIn(k, out)

        for k in expected_keys[:5]:
            self.assertGreaterEqual(out[k], 0.0)
            self.assertLessEqual(out[k], 1.0)

        # Teacher weights must be unchanged.
        for name, param in teacher.state_dict().items():
            self.assertTrue(
                torch.equal(param, teacher_weights_before[name]),
                f"Teacher weight {name} changed during benchmark",
            )


# ---------------------------------------------------------------------------
# Test 7: dynamic_prototypes_for_concept with mock MCTS
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH and HAS_AENUM, "requires torch + aenum deps")
class TestDynamicPrototypesWithMockContrasts(unittest.TestCase):

    def _run(self, positions, net, v, min_margin):
        """Run dynamic_prototypes_for_concept with mocked MCTS."""

        def mock_contrasts(mct, board, max_depth, sort_key,
                           min_value_gap, min_visit_gap_ratio, t_offset):
            optimal = [{"grid": board.grid.copy(), "player": board.player}]
            perturbed = board.grid.copy()
            perturbed[0, 0] = -perturbed[0, 0]
            subpar_states = [{"grid": perturbed, "player": board.player}]
            return {
                "optimal_rollout": optimal,
                "subpar_rollouts": [{"depth": 0, "states": subpar_states}],
                "required_depths": 1,
                "available_depths": 1,
            }

        with patch("src.teachability.teachability.AlphaZeroPlayer") as MockPlayer, \
             patch("src.teachability.teachability.extract_rollout_contrasts",
                   side_effect=mock_contrasts):
            inst = MagicMock()
            inst.get_move.return_value = (None, None, None, None)
            inst.mct = MagicMock()
            MockPlayer.return_value = inst

            return dynamic_prototypes_for_concept(
                net=net,
                positions=positions,
                layer=LAYER,
                v=v,
                n_sim=8,
                max_depth=5,
                min_margin=min_margin,
            )

    def test_stats_consistency(self):
        net = _make_net(seed=7)
        positions = _make_othello_positions(6, seed=60)
        v = np.random.RandomState(77).randn(LAYER_DIM).astype(np.float32)
        v /= np.linalg.norm(v)

        accepted, meta, stats = self._run(positions, net, v, min_margin=0.0)

        self.assertEqual(stats["n_positions"], len(positions))
        self.assertEqual(stats["n_accepted"], len(accepted))
        self.assertEqual(len(meta), len(accepted))

        total = (stats["n_no_optimal"] + stats["n_missing_subpar"]
                 + stats["n_failed_margin"] + stats["n_accepted"])
        self.assertEqual(total, stats["n_positions"])

        for m in meta:
            self.assertGreater(m["n_constraints"], 0)
            self.assertGreaterEqual(m["min_margin"], 0.0)

    def test_permissive_margin_accepts_all(self):
        net = _make_net(seed=7)
        positions = _make_othello_positions(4, seed=61)
        v = np.random.RandomState(78).randn(LAYER_DIM).astype(np.float32)

        accepted, meta, stats = self._run(positions, net, v, min_margin=-1e6)

        self.assertEqual(stats["n_accepted"], len(positions))
        self.assertEqual(len(accepted), len(positions))


# ---------------------------------------------------------------------------
# Test 8: Pipeline helpers
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH and HAS_AENUM, "requires torch + aenum deps")
class TestPipelineHelpers(unittest.TestCase):

    def setUp(self):
        self.positions = _make_othello_positions(20, seed=70)
        self.rng = np.random.RandomState(0)

    # -- _subsample_positions --

    def test_subsample_no_limit(self):
        result, indices = _subsample_positions(self.positions, None, self.rng)
        self.assertEqual(len(result), len(self.positions))
        self.assertEqual(indices, list(range(len(self.positions))))

    def test_subsample_with_limit(self):
        result, indices = _subsample_positions(self.positions, 5, self.rng)
        self.assertEqual(len(result), 5)
        self.assertEqual(len(indices), 5)
        for idx in indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(self.positions))

    def test_subsample_limit_exceeds_length(self):
        result, _ = _subsample_positions(self.positions, 100, self.rng)
        self.assertEqual(len(result), len(self.positions))

    # -- _split_positions --

    def test_split_sizes(self):
        train, test = _split_positions(self.positions, 0.2, self.rng)
        self.assertEqual(len(train) + len(test), len(self.positions))
        self.assertEqual(len(test), 4)
        self.assertEqual(len(train), 16)

    def test_split_at_least_one_test(self):
        small = self.positions[:3]
        train, test = _split_positions(small, 0.1, self.rng)
        self.assertGreaterEqual(len(test), 1)
        self.assertEqual(len(train) + len(test), len(small))

    # -- _sample_random_positions --

    def test_sample_random_excludes_prototypes(self):
        prototypes = self.positions[:5]
        train, test, _ = _sample_random_positions(
            prototypes, self.positions, 8, 4, self.rng,
        )

        def pos_key(pos):
            return (pos["grid"].tobytes(), pos["player"])

        proto_keys = {pos_key(p) for p in prototypes}
        for pos in train + test:
            self.assertNotIn(pos_key(pos), proto_keys)

    def test_sample_random_sizes(self):
        prototypes = self.positions[:3]
        pool = self.positions[3:]
        train, test, replacement = _sample_random_positions(
            prototypes, pool, 5, 3, self.rng,
        )
        self.assertEqual(len(train), 5)
        self.assertEqual(len(test), 3)
        self.assertFalse(replacement)

    def test_sample_random_with_replacement(self):
        pool = self.positions[:3]
        train, test, replacement = _sample_random_positions(
            [], pool, 5, 3, self.rng,
        )
        self.assertTrue(replacement)
        self.assertEqual(len(train), 5)
        self.assertEqual(len(test), 3)

    def test_sample_random_empty_pool(self):
        train, test, replacement = _sample_random_positions(
            [], [], 5, 3, self.rng,
        )
        self.assertEqual(len(train), 0)
        self.assertEqual(len(test), 0)
        self.assertTrue(replacement)


# ---------------------------------------------------------------------------
# Bench: batched vs unbatched _score_rollout_states (gated by BENCH env var)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH and HAS_AENUM, "requires torch + aenum deps")
@unittest.skipUnless(os.environ.get("BENCH"), "set BENCH=1 to run benchmarks")
class TestScoreRolloutStatesBench(unittest.TestCase):

    def test_bench_batched_vs_unbatched(self):
        net = _make_net(seed=0)
        states = _make_othello_positions(50, seed=100)
        v = np.random.RandomState(99).randn(LAYER_DIM).astype(np.float32)

        # Warm up.
        _score_rollout_states(net, states[:2], LAYER, v)

        # Batched (current implementation).
        t0 = time.monotonic()
        for _ in range(5):
            scores_batched = _score_rollout_states(net, states, LAYER, v)
        t_batched = time.monotonic() - t0

        # Unbatched baseline.
        t0 = time.monotonic()
        for _ in range(5):
            scores_unbatched = []
            for s in states:
                z = get_single_activation(net, s["grid"], s["player"], LAYER)
                scores_unbatched.append(float(np.dot(v, z)))
        t_unbatched = time.monotonic() - t0

        # Correctness: must match.
        np.testing.assert_allclose(scores_batched, scores_unbatched, atol=1e-5)

        # Report timing.
        speedup = t_unbatched / t_batched if t_batched > 0 else float("inf")
        print(f"\n  Batched: {t_batched:.3f}s  Unbatched: {t_unbatched:.3f}s  Speedup: {speedup:.1f}x")
        self.assertGreater(speedup, 1.0, "Batched should be faster than unbatched")


if __name__ == "__main__":
    unittest.main()
