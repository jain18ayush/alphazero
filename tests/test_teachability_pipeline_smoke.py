import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import importlib.util

import numpy as np

HAS_AENUM = importlib.util.find_spec("aenum") is not None

if HAS_AENUM:
    import teachability_pipeline as tp


class DummyModel:
    def eval(self):
        return self


@unittest.skipUnless(HAS_AENUM, "requires aenum/alphazero deps")
class TestTeachabilityPipelineSmoke(unittest.TestCase):
    def test_pipeline_writes_outputs_and_uses_fresh_student_paths(self):
        cfg = {
            "seed": 0,
            "teacher_model": {"source": "teacher_src"},
            "student_model": {"source": "student_src", "checkpoint_path": "student.pt"},
            "positions": {"source": "positions_src"},
            "random_positions": {"source": "random_src"},
            "concepts": {"source": "concept_src"},
            "dynamic_prototypes": {
                "max_depth": 5,
                "sort_key": "N",
                "min_margin": 0.0,
                "min_value_gap": 0.20,
                "min_visit_gap_ratio": 0.10,
                "t_offset": 5,
                "top_percent": None,
                "max_prototypes": None,
                "max_positions": None,
            },
            "teachability": {
                "n_sim": 25,
                "temp": 1.0,
                "epochs": 2,
                "lr": 1e-3,
                "batch_size": 8,
                "test_split": 0.2,
                "min_prototypes": 20,
                "teachability_margin": 0.05,
            },
            "student_selection": {
                "overlap_threshold": 0.2,
                "checkpoint_paths": [],
            },
        }

        positions = [{"grid": np.zeros((8, 8), dtype=np.float32), "player": 1} for _ in range(50)]
        concepts = [{"layer": "bn2", "vector": np.ones(32, dtype=np.float32), "path": "fake/path.npy"}]

        def datasets_get(source):
            def load(_cfg):
                if source == "positions_src":
                    return positions
                if source == "random_src":
                    return positions
                if source == "concept_src":
                    return concepts
                raise KeyError(source)
            return load

        def models_get(_source):
            return lambda _cfg: DummyModel()

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with patch.object(tp.DATASETS, "get", side_effect=datasets_get), patch.object(tp.MODELS, "get", side_effect=models_get), patch.object(
                tp, "dynamic_prototypes_for_concept", return_value=(positions[:25], [{"min_margin": 0.1}], {"n_accepted": 25})
            ), patch.object(tp, "measure_top1_agreement", return_value=(3, 0.1)), patch.object(
                tp,
                "run_teachability_benchmark",
                return_value={
                    "baseline_eval_C": 0.1,
                    "train_C_eval_C": 0.6,
                    "train_C_eval_R": 0.4,
                    "train_R_eval_C": 0.45,
                    "train_R_eval_R": 0.5,
                    "loss_tail_concept": [0.2],
                    "loss_tail_random": [0.2],
                },
            ):
                summary = tp.run_teachability(cfg, run_dir)

            self.assertEqual(summary["n_concepts"], 1)
            self.assertEqual(summary["n_teachable"], 1)

            concept_dir = run_dir / "concept_0000"
            self.assertTrue((concept_dir / "result.json").exists())
            self.assertTrue((concept_dir / "benchmark.json").exists())
            self.assertTrue((run_dir / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
