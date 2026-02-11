import tempfile
import unittest
from pathlib import Path
import importlib.util

import numpy as np

HAS_AENUM = importlib.util.find_spec("aenum") is not None


@unittest.skipUnless(HAS_AENUM, "requires aenum/alphazero deps")
class TestConceptVectorsFromRuns(unittest.TestCase):
    def test_load_filters_layers_and_respects_cap(self):
        from src.datasets.datasets import load_concept_vectors_from_runs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "exp" / "layer=bn2"
            b = root / "exp" / "layer=conv1"
            c = root / "exp" / "misc"
            a.mkdir(parents=True)
            b.mkdir(parents=True)
            c.mkdir(parents=True)

            np.save(a / "concept_vector.npy", np.array([1.0, 2.0]))
            np.save(b / "concept_vector.npy", np.array([3.0, 4.0]))
            np.save(c / "concept_vector.npy", np.array([9.0]))

            concepts = load_concept_vectors_from_runs({
                "run_root": str(root),
                "layers": ["bn2"],
                "max_concepts": 10,
            })

            self.assertEqual(len(concepts), 1)
            self.assertEqual(concepts[0]["layer"], "bn2")
            np.testing.assert_allclose(concepts[0]["vector"], np.array([1.0, 2.0]))

    def test_stable_sort_and_max_concepts(self):
        from src.datasets.datasets import load_concept_vectors_from_runs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(3):
                d = root / f"run_{i}" / "layer=bn2"
                d.mkdir(parents=True)
                np.save(d / "concept_vector.npy", np.array([i], dtype=np.float32))

            concepts = load_concept_vectors_from_runs({
                "run_root": str(root),
                "max_concepts": 2,
            })

            self.assertEqual(len(concepts), 2)
            self.assertLessEqual(concepts[0]["path"], concepts[1]["path"])


if __name__ == "__main__":
    unittest.main()
