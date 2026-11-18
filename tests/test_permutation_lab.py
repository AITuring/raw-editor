import importlib.util
import math
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/stack-lab/permutation.py'
spec = importlib.util.spec_from_file_location('permutation_lab', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PermutationLabTest(unittest.TestCase):
    def test_ncc_identity_and_sign(self):
        a = [float(i % 17) for i in range(200)]
        self.assertAlmostEqual(module.ncc(a, a), 1.0, places=12)
        self.assertAlmostEqual(module.ncc(a, [-x for x in a]), -1.0, places=12)
        self.assertIsNone(module.ncc([1.0] * 100, [2.0] * 100))

    def test_projective_identity_and_invalid_depth(self):
        identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.assertEqual(module.transform(identity, 12.5, 8.0), (12.5, 8.0))
        self.assertIsNone(module.transform([1.0, 0, 0, 0, 1.0, 0, 0, 0, 0], 12, 8))

    def test_quantiles_are_deterministic_and_fail_closed(self):
        self.assertEqual(module.quantiles([]), {'count': 0})
        q = module.quantiles([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(q['count'], 4)
        self.assertTrue(math.isclose(q['p99'], 0.4))


if __name__ == '__main__':
    unittest.main()
