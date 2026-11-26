"""Unit tests for the bounded determinism & resilience matrix helpers."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / 'scripts/stack-lab/determinism.py'
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location('stack_lab_determinism', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DeterminismHelpersTest(unittest.TestCase):
    def test_signature_ignores_runtime_observations_but_keeps_output_contract(self):
        base = {
            'status': 'rejected',
            'checks': {'memory_gate_pass': True, 'output_contract': False},
            'quality_gate': {'verdict': 'fail', 'roi_count': 8, 'criteria': {'boundary': 'fail'}},
            'closure': 'converged', 'outcome': 'rejected',
            'sources': {'inputs': 3, 'report_sources': 3, 'ordered': 3, 'streamed': 3,
                        'lending': 0, 'expected_skipped': [], 'covered_pixels': 8, 'owned_pixels': 8},
            'output': {'width': 2, 'height': 2, 'bit_depth': 16, 'alpha_preserved': True,
                       'written': True, 'artifact_validation': 'pass'},
            'output_sha256': 'same', 'peak_rss_bytes': 100, 'wall_seconds': 1,
            'report': '/tmp/one.json', 'report_sha256': 'run-specific',
        }
        changed_runtime = dict(base, peak_rss_bytes=200, wall_seconds=9,
                               report='/tmp/two.json', report_sha256='different')
        self.assertEqual(MODULE.stable_signature(base), MODULE.stable_signature(changed_runtime))

    def test_signature_detects_output_or_verdict_change(self):
        base = MODULE.stable_signature({'status': 'rejected', 'quality_gate': {'verdict': 'fail'},
                                        'output_sha256': 'a'})
        changed = dict(base, output_sha256='b')
        self.assertTrue(MODULE.compare_signatures(base, changed))

    def test_thread_parser_is_unique_and_bounded_by_exclusive_slot(self):
        self.assertEqual(MODULE.parse_threads('1,2,2,8'), [1, 2, 8])
        with self.assertRaises(Exception):
            MODULE.parse_threads('16')

    def test_matrix_classifies_uniform_runner_block_as_blocked(self):
        runs = [
            {'blocking_reason': 'required_nice_priority_unavailable', 'mismatches': [{'field': 'analysis.status'}]},
            {'blocking_reason': 'required_nice_priority_unavailable', 'mismatches': [{'field': 'analysis.status'}]},
        ]
        self.assertEqual(
            MODULE.classify_matrix_status(runs),
            ('blocked', 'required_nice_priority_unavailable'),
        )

    def test_matrix_keeps_mixed_or_material_mismatch_as_failure(self):
        runs = [
            {'blocking_reason': 'required_nice_priority_unavailable', 'mismatches': [{'field': 'analysis.status'}]},
            {'blocking_reason': None, 'mismatches': [{'field': 'output_sha256'}]},
        ]
        self.assertEqual(MODULE.classify_matrix_status(runs), ('fail', None))

    def test_scenario_parser_and_dynamic_not_run_tracking(self):
        self.assertEqual(
            MODULE.parse_scenarios('threads,order_permutation,cancellation,order_permutation'),
            ['threads', 'order_permutation', 'cancellation'],
        )
        self.assertEqual(
            MODULE.remaining_not_run(['threads']),
            ['order_permutation', 'concurrent_load', 'cancellation', 'oom', 'provider_fallback'],
        )
        self.assertEqual(
            MODULE.remaining_not_run(['threads', 'order_permutation', 'concurrent_load', 'cancellation', 'oom']),
            ['provider_fallback'],
        )
        with self.assertRaises(Exception):
            MODULE.parse_scenarios('threads,unknown_mode')

    def test_permuted_symlink_view_preserves_targets_and_reverses_sort_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_dir = root / 'sources'
            src_dir.mkdir()
            files = []
            for name in ('DSC_0001.NEF', 'DSC_0002.NEF', 'DSC_0003.NEF'):
                p = src_dir / name
                p.write_bytes(name.encode())
                files.append(p)
            view_dir = root / 'permuted'
            mapping = MODULE.build_permuted_symlink_view(files, view_dir, mode='reverse')
            self.assertEqual(len(mapping), 3)
            links = sorted(view_dir.iterdir())
            self.assertEqual([link.resolve().name for link in links], ['DSC_0003.NEF', 'DSC_0002.NEF', 'DSC_0001.NEF'])
            self.assertTrue(all(link.is_symlink() for link in links))

    def test_cancellation_and_oom_probes_fail_closed_when_output_published(self):
        self.assertEqual(
            MODULE.evaluate_cancellation_probe(143, {'acceptance_state': 'cancelled'}, False),
            [],
        )
        self.assertTrue(
            MODULE.evaluate_cancellation_probe(0, {'acceptance_state': 'accepted'}, True),
        )
        self.assertEqual(
            MODULE.evaluate_oom_probe(1, {'status': 'resource_failed', 'memory': {'threshold_exceeded': True}}, False),
            [],
        )
        self.assertTrue(
            MODULE.evaluate_oom_probe(1, {'status': 'rejected', 'memory': {'threshold_exceeded': False}}, False),
        )
        self.assertTrue(
            MODULE.evaluate_oom_probe(1, {'status': 'resource_failed', 'memory': {'threshold_exceeded': True}}, True),
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
