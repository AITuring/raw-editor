"""Unit tests for the offline Quality_Gate what-if replay (tasks.md 2B.5)."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'qg_whatif', Path(__file__).resolve().parents[2] / 'scripts' / 'stack-lab' / 'qg_whatif.py')
qg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qg)


def criterion(name, verdict, measurable=8, technical=0, failed=(), measured=(), diagnostic=False, unresolved=0):
    return {'name': name, 'verdict': verdict, 'measurable_count': measurable,
            'technical_unmeasurable_count': technical, 'failed': list(failed), 'measured': list(measured),
            'diagnostic': diagnostic, 'unresolved_pixel_count': unresolved}


def passing_criteria():
    out = []
    for name in qg.QUALITY_CRITERIA:
        if name in qg.WHOLE_IMAGE:
            out.append(criterion(name, 'pass', measurable=1))
        else:
            out.append(criterion(name, 'pass'))
    return out


def layered_report(criteria, tone_max=0.5, density=None, closure=None):
    return {
        'selected_path': 'layered_virtual_tile',
        'station_relations': {'connectivity': {'components': 1}},
        'closure': closure or {'status': 'converged', 'max_direct_pair_p95_px': 1.0,
                               'residual_median_px': 0.2, 'max_corner_correction_px': 5.0},
        'quality_gate': {'verdict': qg.overall_verdict([c['verdict'] for c in criteria]), 'criteria': criteria},
        'tone': {'status': 'applied', 'boundary_delta_e': {'max': tone_max}},
        'composition': {'output_density': density or {'status': 'unchanged', 'factor': 1.0}},
    }


class CriterionVerdictTests(unittest.TestCase):
    def test_mirrors_rust_ordering(self):
        self.assertEqual(qg.criterion_verdict('noise_sigma_ratio', 2, 0, True, False, False), 'fail')
        self.assertEqual(qg.criterion_verdict('noise_sigma_ratio', 2, 0, False, False, False), 'not_applicable')
        self.assertEqual(qg.criterion_verdict('roi_delta_e00', 2, 0, False, False, False), 'insufficient_evidence')
        self.assertEqual(qg.criterion_verdict('roi_delta_e00', 8, 3, True, False, False), 'insufficient_evidence')
        self.assertEqual(qg.criterion_verdict('local_scale_median', 8, 0, True, True, False), 'not_applicable')
        self.assertEqual(qg.criterion_verdict('effective_pixel_count', 0, 1, False, False, True), 'insufficient_evidence')
        self.assertEqual(qg.criterion_verdict('effective_pixel_count', 1, 0, False, False, False), 'pass')

    def test_overall(self):
        self.assertEqual(qg.overall_verdict(['pass', 'not_applicable']), 'pass')
        self.assertEqual(qg.overall_verdict(['pass', 'insufficient_evidence']), 'insufficient_evidence')
        self.assertEqual(qg.overall_verdict(['insufficient_evidence', 'fail']), 'fail')
        self.assertEqual(qg.overall_verdict([]), 'insufficient_evidence')


class OptionTests(unittest.TestCase):
    def test_c3_ignores_p95_only_failure_with_few_samples(self):
        values = [{'measured': 3.0}, {'measured': 1.0}]
        boundary = criterion('boundary_stroke_alignment', 'fail', measurable=2,
                             failed=[{'measured': 3.0}], measured=values)
        self.assertEqual(qg.recompute_criterion(boundary), 'fail')
        self.assertEqual(qg.recompute_criterion(boundary, c3=True), 'not_applicable')
        over = criterion('boundary_stroke_alignment', 'fail', measurable=2,
                         failed=[{'measured': 3.5}], measured=[{'measured': 3.5}])
        self.assertEqual(qg.recompute_criterion(over, c3=True), 'fail')
        many = criterion('boundary_stroke_alignment', 'fail', measurable=9,
                         failed=[{'measured': 2.0}], measured=[{'measured': 2.0}])
        self.assertEqual(qg.recompute_criterion(many, c3=True), 'fail', 'C3 only applies below 8 samples')

    def test_c1_uses_base_closure(self):
        report = layered_report(passing_criteria(), density={
            'status': 'rejected', 'bypass_reason': 'output_pixel_closure_limit_exceeded', 'factor': 1.37890625},
            closure={'status': 'converged', 'max_direct_pair_p95_px': 2.5386, 'residual_median_px': 0.3,
                     'max_corner_correction_px': 10.0})
        report['quality_gate'] = {'verdict': 'not_run', 'criteria': []}
        self.assertEqual(qg.evaluate(report)['density'], 'rejected')
        what_if = qg.evaluate(report, c1=True)
        self.assertEqual(what_if['density'], 'applied')
        self.assertEqual(what_if['quality_gate'], 'needs_rerun')

    def test_bypass_is_never_redecided(self):
        report = layered_report(passing_criteria(), density={
            'status': 'bypassed', 'bypass_reason': 'nonidentity_world_residual', 'factor': 1.0})
        self.assertEqual(qg.evaluate(report, c1=True)['density'], 'bypassed')


class NativeAcceptanceTests(unittest.TestCase):
    def test_accepts_complete_pass_and_content_not_applicable(self):
        criteria = passing_criteria()
        for c in criteria:
            if c['name'] == 'mtf50_normalized':
                c.update(verdict='not_applicable', measurable=0)
        self.assertEqual(qg.evaluate(layered_report(criteria))['native'], 'accepted')

    def test_rejects_tone_and_diagnostic_local_scale(self):
        self.assertEqual(qg.evaluate(layered_report(passing_criteria(), tone_max=2.0))['native'], 'rejected')
        criteria = passing_criteria()
        criteria[0].update(diagnostic=True, verdict='not_applicable')
        result = qg.evaluate(layered_report(criteria))
        self.assertEqual(result['native'], 'rejected')
        self.assertIn('local_scale_median', result['native_reason'])

    def test_legacy_path_is_not_layered(self):
        report = layered_report(passing_criteria())
        report['selected_path'] = 'legacy_single_layer_mosaic'
        self.assertEqual(qg.evaluate(report)['native'], 'not_layered')


class EndToEndTests(unittest.TestCase):
    def test_replay_reports_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'output' / 'acceptance' / 'lane' / 'case' / 'run' / 'x-stack-reports'
            run.mkdir(parents=True)
            (run / 'r.json').write_text(json.dumps(layered_report(passing_criteria())))
            cases = root / 'cases.json'
            cases.write_text(json.dumps({'cases': [{'id': 'c', 'report': 'lane/case'}]}))
            output = root / 'out.json'
            self.assertEqual(qg.main(['--cases', str(cases), '--root', str(root), '--output', str(output)]), 0)
            data = json.loads(output.read_text())
            self.assertEqual(data['consistency'], 'pass')
            self.assertEqual(data['cases'][0]['current']['native'], 'accepted')
            # A tampered recorded verdict must be caught.
            bad = layered_report(passing_criteria())
            bad['quality_gate']['criteria'][1]['verdict'] = 'fail'
            (run / 'r.json').write_text(json.dumps(bad))
            self.assertEqual(qg.main(['--cases', str(cases), '--root', str(root)]), 1)


if __name__ == '__main__':
    unittest.main()
