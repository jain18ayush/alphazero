import copy
import unittest
from unittest.mock import patch
import importlib.util

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_AENUM = importlib.util.find_spec("aenum") is not None

if HAS_TORCH and HAS_AENUM:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from src.teachability.teachability import (
        distill_student,
        is_teachable,
        measure_top1_agreement,
        run_teachability_benchmark,
        select_student_checkpoint,
    )


if HAS_TORCH and HAS_AENUM:
    class TinyPolicyNet(nn.Module):
        def __init__(self, action_size=5):
            super().__init__()
            self.fc = nn.Linear(4, action_size)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            logits = self.fc(x)
            value = torch.zeros((x.size(0), 1), dtype=logits.dtype, device=logits.device)
            return F.log_softmax(logits, dim=1), value


def _make_positions(n):
    rng = np.random.RandomState(0)
    out = []
    for _ in range(n):
        grid = rng.randint(-1, 2, size=(2, 2)).astype(np.float32)
        out.append({"grid": grid, "player": 1})
    return out


def _fake_policy_for_positions(model, positions, n_sim, temp):
    if len(positions) == 0:
        return [], []
    x = torch.tensor(np.array([p["player"] * p["grid"] for p in positions]), dtype=torch.float32)
    with torch.no_grad():
        log_probs, _ = model(x)
    probs = torch.exp(log_probs).cpu().numpy().astype(np.float32)
    top1 = [int(np.argmax(p)) for p in probs]
    return [p for p in probs], top1


@unittest.skipUnless(HAS_TORCH and HAS_AENUM, "requires torch + aenum/alphazero deps")
class TestTeachabilityCore(unittest.TestCase):
    def test_measure_agreement_counts_matches(self):
        positions = _make_positions(3)
        with patch(
            "src.teachability.teachability.mcts_policy_for_positions",
            side_effect=[([np.array([1])], [0, 1, 1]), ([np.array([1])], [0, 0, 1])],
        ):
            matches, overlap = measure_top1_agreement(
                teacher=object(),
                student=object(),
                positions=positions,
                n_sim=10,
                temp=1.0,
            )
        self.assertEqual(matches, 2)
        self.assertAlmostEqual(overlap, 2 / 3)

    def test_distill_student_reduces_kl(self):
        torch.manual_seed(0)
        student = TinyPolicyNet(action_size=5)
        positions = _make_positions(24)

        target_pis = np.tile(np.array([0.70, 0.10, 0.10, 0.05, 0.05], dtype=np.float32), (len(positions), 1))

        x = torch.tensor(np.array([p["player"] * p["grid"] for p in positions]), dtype=torch.float32)
        y = torch.tensor(target_pis, dtype=torch.float32)

        with torch.no_grad():
            pre_log_probs, _ = student(x)
            pre_loss = F.kl_div(pre_log_probs, y, reduction="batchmean").item()

        losses = distill_student(
            student,
            positions,
            target_pis,
            epochs=20,
            lr=1e-2,
            batch_size=8,
        )

        with torch.no_grad():
            post_log_probs, _ = student(x)
            post_loss = F.kl_div(post_log_probs, y, reduction="batchmean").item()

        self.assertGreater(len(losses), 0)
        self.assertLess(post_loss, pre_loss)

    def test_benchmark_outputs_and_teacher_unchanged(self):
        torch.manual_seed(1)
        teacher = TinyPolicyNet(action_size=5)
        teacher_before = copy.deepcopy(teacher.state_dict())

        student_template = TinyPolicyNet(action_size=5)

        def load_student():
            m = TinyPolicyNet(action_size=5)
            m.load_state_dict(copy.deepcopy(student_template.state_dict()))
            m.eval()
            return m

        X_train_concept = _make_positions(8)
        X_test_concept = _make_positions(4)
        X_train_random = _make_positions(8)
        X_test_random = _make_positions(4)

        with patch("src.teachability.teachability.mcts_policy_for_positions", side_effect=_fake_policy_for_positions):
            out = run_teachability_benchmark(
                load_student=load_student,
                teacher=teacher,
                X_train_concept=X_train_concept,
                X_test_concept=X_test_concept,
                X_train_random=X_train_random,
                X_test_random=X_test_random,
                n_sim=10,
                temp=1.0,
                epochs=3,
                lr=1e-2,
                batch_size=4,
            )

        for k in [
            "baseline_eval_C",
            "train_C_eval_C",
            "train_C_eval_R",
            "train_R_eval_C",
            "train_R_eval_R",
        ]:
            self.assertIn(k, out)

        for name, value in teacher.state_dict().items():
            self.assertTrue(torch.equal(value, teacher_before[name]))

    def test_is_teachable_uses_c_minus_r_on_concept(self):
        result, gain = is_teachable(
            {
                "train_C_eval_C": 0.62,
                "train_R_eval_C": 0.50,
            },
            margin=0.10,
        )
        self.assertTrue(result)
        self.assertAlmostEqual(gain, 0.12)

    def test_select_student_checkpoint_picks_latest_below_threshold(self):
        checkpoints = [
            "chkpt-1.pt",
            "chkpt-2.pt",
            "chkpt-3.pt",
        ]

        def make_model_cfg(path):
            return {"checkpoint_path": path}

        class FakeModel:
            def __init__(self, path):
                self.path = path

        def load_model(cfg):
            return FakeModel(cfg["checkpoint_path"])

        def fake_measure(_teacher, student, _positions, n_sim, temp):
            overlap_by_path = {
                "chkpt-3.pt": 0.35,
                "chkpt-2.pt": 0.18,
                "chkpt-1.pt": 0.05,
            }
            return 0, overlap_by_path[student.path]

        with patch("src.teachability.teachability.measure_top1_agreement", side_effect=fake_measure):
            out = select_student_checkpoint(
                checkpoint_paths=checkpoints,
                make_model_cfg=make_model_cfg,
                load_model=load_model,
                teacher=object(),
                prototypes=[{"grid": np.zeros((2, 2), dtype=np.float32), "player": 1}],
                overlap_threshold=0.2,
                n_sim=10,
                temp=1.0,
            )

        self.assertEqual(out["selected_path"], "chkpt-2.pt")
        self.assertLess(out["selected_overlap"], 0.2)


if __name__ == "__main__":
    unittest.main()
