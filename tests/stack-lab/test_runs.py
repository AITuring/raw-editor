"""CLI integration tests with a fake image runner; no original images or Rust builds."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import test_tools as fixtures
SOURCE, PYTHON = fixtures.SOURCE, fixtures.PYTHON

FAKE_RUNNER = r'''
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const a=process.argv.slice(2),get=k=>a[a.indexOf(k)+1];
const out=get('--out-dir');fs.mkdirSync(out,{recursive:true});
const dataset=get('--dataset');
const manifest=JSON.parse(fs.readFileSync('tests/acceptance/focus-stack-datasets.json'));
const entry=manifest.datasets.find(x=>x.id===dataset);
let files=fs.readdirSync(process.env.RAW_EDITOR_FOCUS_STACK_DATA_ROOT+'/sleeping').sort().map(n=>process.env.RAW_EDITOR_FOCUS_STACK_DATA_ROOT+'/sleeping/'+n);
if(a.includes('--subset'))files=files.slice(0,2);
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const mode=process.env.RAW_EDITOR_TEST_MODE;
if(mode==='wait'){
 fs.writeFileSync(path.join(out,'native.pid'),String(process.pid));
 await new Promise(resolve=>setTimeout(resolve,30000));
}
await new Promise(resolve=>setTimeout(resolve,150));
const tiff=path.join(out,'fixture.tiff');
const data=mode==='vary_by_slot' && process.env.STACK_LAB_SLOT==='run'?'changed fixture':'tiff fixture';
if(mode!=='missing_output')fs.writeFileSync(tiff,data);
const failed=mode==='known_failure';
const r={input:{source_count:files.length,sources:files.map(p=>({path:p,sha256_before:sha(p),sha256_after:sha(p)}))},
 selected_path:'layered_virtual_tile',grouping:{station_count:1},closure:{status:'converged'},degradation:{result:failed?'rejected':'accepted'},
 quality_gate:{verdict:failed?'fail':'pass',roi_count:8,criteria:[{name:'sharpness',verdict:failed?'fail':'pass'}]},
 composition:{source_covered_pixels:8,source_owned_pixels:8},
 output:{path:tiff,width:2,height:2,written:mode!=='missing_output',artifact_validation:'pass',bit_depth:16,alpha_preserved:true},
 performance:{source_preflight:{raw_decode_count:4},native_fusion_cache:{max_bytes:1073741824,peak_bytes:200,fusion_count:1},source_render_cache:{max_bytes:536870912,peak_bytes:100}}};
if(mode==='bad_source')r.input.sources[0].sha256_after='0'.repeat(64);
const reports=path.join(out,'fixture-stack-reports');fs.mkdirSync(reports);
fs.writeFileSync(path.join(reports,'stack-report-main.json'),JSON.stringify(r));
if(mode==='aux_report')fs.writeFileSync(path.join(reports,'stack-report-aux.json'),JSON.stringify({output:{}}));
if(mode==='two_reports')fs.writeFileSync(path.join(reports,'stack-report-duplicate.json'),JSON.stringify(r));
fs.writeFileSync(path.join(out,'fixture.summary.json'),JSON.stringify({verdict:failed?'failed':'identical',failures:failed?['known gate failure']:[],summary:{inputs:files.length,orderedInputs:files.length}}));
if(mode!=='missing_usage')fs.writeFileSync(path.join(out,'fixture.log'),'NATIVE_PROCESS_USAGE '+JSON.stringify({max_rss_bytes:mode==='over_memory'?25769803777:mode==='at_memory_limit'?25769803776:mode==='invalid_usage'?'unknown':100000,child_exit_code:failed?101:0})+'\n');
process.exit(failed?1:0);
'''


class RunsTest(fixtures.ToolsTest):
    # Reuse fixture helpers without inheriting the task-1 test methods.
    def setUp(self):
        super().setUp()
        for name in ('run.py', 'matrix.py', 'monitor.py', 'lanting_audit.py'):
            (self.script / name).write_bytes((SOURCE / name).read_bytes())
        self.data = self.root / 'datasets'
        folder = self.data / 'sleeping'
        folder.mkdir(parents=True)
        for i in range(3):
            (folder / f'{i}.NEF').write_text('source ' + str(i))
        (self.root / 'tests/acceptance').mkdir(parents=True)
        self.manifest = {'datasets': [dict(id='sleeping', directory='sleeping', sourceExtensions=['.nef'],
            excludePatterns=[], expectedSourceCount=3, regressionSubsets=[dict(id='sleeping-9-6479', first=2)])]}
        (self.root / 'tests/acceptance/focus-stack-datasets.json').write_text(json.dumps(self.manifest))
        (self.root / 'tests/focus-stack-regression.mjs').write_text(FAKE_RUNNER)
        self.env['RAW_EDITOR_FOCUS_STACK_DATA_ROOT'] = str(self.data)
        self.artifact.write_text('#!' + PYTHON + '\n' + "import os,json,sys\nif '--exact' in sys.argv:\n n=int(os.environ['RAYON_NUM_THREADS']);print('test probe ... STACK_RESOURCE_CONTRACT '+json.dumps(dict(version=1,rayon_threads=n,preparation_threads=n)));print('test result: ok. 1 passed; 0 failed')\nelse: print('frozen fixture')\n")
        self.artifact.chmod(0o755)
        self.evidence['sha256']=self.digest(self.artifact);self.write_evidence()
        self.assertEqual(self.adopt('fixture')[0], 0)
        self.expected = {'status':'quality_accepted','closure':'converged','exit_code':0,'quality_gate.verdict':'pass',
            'quality_gate.roi_count':8,'output_sha256':hashlib.sha256(b'tiff fixture').hexdigest(),
            'checks.output_contract':True}
        self.config = dict(version=1,baseline=dict(label='fixture',binary_sha256=self.evidence['sha256']),
                           cases=[dict(id='nine',dataset='sleeping',subset='sleeping-9-6479',scale=1,tiers=['medium','full'],expected=self.expected)])
        self.matrix_file = self.root / 'tests/acceptance/regression-matrix.json'
        self.matrix_file.write_text(json.dumps(self.config))

    def run_one(self, label='r1', mode=None, *extra):
        command = ['--binary','fixture','--lane','task-2','--dataset','sleeping','--subset','sleeping-9-6479','--label',label]
        if mode:
            command += ['--env', 'RAW_EDITOR_TEST_MODE=' + mode]
        result = self.run_tool('run.py', *command, *extra)
        dest = self.root / 'output/acceptance/task-2' / ('sleeping-9-6479-' + label)
        return result, dest

    def matrix(self, label='m1'):
        return self.run_tool('matrix.py','--tier','medium','--binary','fixture','--lane','task-2','--label',label)

    def test_unverified_legacy_binary_is_forced_exclusive(self):
        p=self.root/'output/acceptance/binaries/fixture/build.json'
        d=json.loads(p.read_text());d.pop('resource_contract');p.chmod(0o644);p.write_text(json.dumps(d))
        (code,_,err),dest=self.run_one('legacy')
        self.assertEqual(code,0,err)
        self.assertEqual(json.loads((dest/'execution.json').read_text())['slot']['slot'],'exclusive')

    def test_legacy_binary_cannot_be_manually_wrapped_in_shared_slot(self):
        p=self.root/'output/acceptance/binaries/fixture/build.json'
        d=json.loads(p.read_text());d.pop('resource_contract');p.chmod(0o644);p.write_text(json.dumps(d))
        code,_,err=self.run_tool('slot.py','run','--estimate-gib','12','--',PYTHON,str(self.script/'run.py'),
            '--binary','fixture','--lane','task-2','--dataset','sleeping','--label','legacy-shared')
        self.assertNotEqual(code,0)
        self.assertIn('requires exclusive',err)
        self.assertFalse((self.root/'output/acceptance/task-2/sleeping-legacy-shared').exists())

    def test_freeze_rejects_a_failed_native_budget_probe(self):
        self.artifact.write_text('#!' + PYTHON + '\nimport json\nprint("STACK_RESOURCE_CONTRACT "+json.dumps(dict(version=1,rayon_threads=1,preparation_threads=6)))\nprint("test result: ok. 1 passed; 0 failed")\n')
        self.artifact.chmod(0o755);self.evidence['sha256']=self.digest(self.artifact);self.write_evidence()
        code,_,_=self.adopt('bad-contract')
        self.assertNotEqual(code,0)
        self.assertFalse((self.root/'output/acceptance/binaries/bad-contract').exists())

    def test_single_run_integrity_memory_summary_and_exit(self):
        (code, _, err), dest = self.run_one()
        self.assertEqual(code, 0, err)
        execution=json.loads((dest/'execution.json').read_text());actual=json.loads((dest/'analysis.json').read_text())
        self.assertEqual(execution['resource_contract']['status'],'verified')
        self.assertEqual(actual['status'],'quality_accepted')
        self.assertTrue(actual['memory_gate_pass'])
        self.assertGreater(actual['peak_rss_bytes'],0)
        self.assertTrue(all(actual['checks'].values()))
        self.assertEqual(execution['slot']['slot'],'run')
        self.assertEqual(execution['slot']['caps']['RAYON_NUM_THREADS'],'5')
        self.assertEqual(len(execution['inputs']),2)
        self.assertEqual(self.run_one()[0][0],1)

    def test_missing_usage_and_over_limit_fail_even_with_pass_qg(self):
        for mode in ('missing_usage','over_memory','invalid_usage'):
            (code, _, _), dest=self.run_one(mode,mode)
            self.assertNotEqual(code,0)
            a=json.loads((dest/'analysis.json').read_text())
            self.assertEqual(a['quality_gate']['verdict'],'pass')
            self.assertEqual(a['status'],'resource_failed')
            self.assertFalse(a['memory_gate_pass'])
        self.assertEqual(self.run_one('boundary','at_memory_limit')[0][0],0)

    def test_source_digest_failure_and_missing_output_do_not_pass(self):
        for mode in ('bad_source','missing_output'):
            (code, _, _), dest=self.run_one(mode,mode)
            self.assertNotEqual(code,0)
            self.assertEqual(json.loads((dest/'analysis.json').read_text())['status'],'rejected')

    def test_select_measured_report_and_reject_ambiguous_evidence(self):
        self.assertEqual(self.run_one('aux','aux_report')[0][0],0)
        (code, _, _),dest=self.run_one('ambiguous','two_reports')
        self.assertNotEqual(code,0)
        self.assertEqual(json.loads((dest/'analysis.json').read_text())['status'],'analysis_error')

    def test_unknown_scope_uses_exclusive_and_rejects_path_escape(self):
        code, _, err=self.run_tool('run.py','--binary','fixture','--lane','task-2','--dataset','sleeping','--label','full','--estimate-gib','1')
        self.assertEqual(code,0,err)
        dest=self.root/'output/acceptance/task-2/sleeping-full'
        self.assertEqual(json.loads((dest/'execution.json').read_text())['slot']['slot'],'exclusive')
        self.assertNotEqual(self.run_one('escape',None,'--out',str(self.data/'escape'))[0][0],0)
        self.assertFalse((self.data/'escape').exists())
        self.assertNotEqual(self.run_one('../escape')[0][0],0)

    def test_input_manifest_matches_global_sort_reverse_and_case_insensitive_exclusions(self):
        for folder,name in [('a','Z.NEF'),('b','A.NEF'),('b','copyFOCUSSTACK.NEF'),('b','literalQ.NEF'),('b','literal?.NEF')]:
            target=self.data/folder/name;target.parent.mkdir(exist_ok=True);target.write_text('fixture')
        self.manifest['datasets']=[dict(id='mix',directories=['b','a'],sourceExtensions=['.nef'],
            excludePatterns=['*FocusStack*','literal?.NEF'],expectedSourceCount=3,inputOrder='reverse')]
        (self.root/'tests/acceptance/focus-stack-datasets.json').write_text(json.dumps(self.manifest))
        env=dict(self.env,PYTHONPATH=str(self.script))
        code='from common import dataset_files;print([str(p) for p in dataset_files("mix")[1]])'
        result=subprocess.run([PYTHON,'-c',code],cwd=self.root,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertLess(result.stdout.index('/b/A.NEF'),result.stdout.index('/a/Z.NEF'))
        self.assertNotIn('FOCUSSTACK',result.stdout)
        self.assertIn('literalQ.NEF',result.stdout)
        self.assertNotIn('literal?.NEF',result.stdout)

    def test_baseline_writes_and_protected_env_are_refused(self):
        before=self.matrix_file.read_bytes()
        self.assertNotEqual(self.run_one('bad',None,'--update-baseline')[0][0],0)
        self.assertNotEqual(self.run_one('bad2',None,'--env','RAW_EDITOR_STACK_ACCEPTANCE_OUTPUT=/tmp/escape')[0][0],0)
        self.assertEqual(before,self.matrix_file.read_bytes())

    def test_matrix_retries_wrong_expectation_exclusively_and_fails(self):
        self.expected['output_sha256']='0'*64
        self.matrix_file.write_text(json.dumps(self.config))
        code, _, err=self.matrix()
        self.assertEqual(code,1,err)
        report=json.loads((self.root/'output/acceptance/task-2/matrix-m1-medium/matrix.json').read_text())
        self.assertEqual(report['status'],'fail')
        attempts=report['cases'][0]['attempts']
        self.assertEqual(len(attempts),2)
        self.assertFalse(attempts[0]['exclusive_requested'])
        self.assertTrue(attempts[1]['exclusive_requested'])
        self.assertEqual(json.loads((Path(attempts[1]['directory'])/'execution.json').read_text())['slot']['slot'],'exclusive')
        self.assertTrue(attempts[0]['mismatches'])
        self.assertTrue(attempts[1]['mismatches'])

    def test_matrix_matching_known_failure_is_not_quality_acceptance(self):
        # A fake runner selected by the fixture, without weakening matrix source/resource checks.
        p=self.root/'tests/focus-stack-regression.mjs'
        p.write_text(FAKE_RUNNER.replace('const mode=process.env.RAW_EDITOR_TEST_MODE;', "const mode='known_failure';"))
        self.expected.update(status='rejected',exit_code=1,**{'quality_gate.verdict':'fail'})
        self.config['cases'][0]['known_failure']=True
        self.matrix_file.write_text(json.dumps(self.config))
        code, _, err=self.matrix()
        self.assertEqual(code,0,err)
        report=json.loads((self.root/'output/acceptance/task-2/matrix-m1-medium/matrix.json').read_text())
        self.assertFalse(report['cases'][0]['quality_accepted'])
        self.assertEqual(report['cases'][0]['status'],'pass')

    def test_exclusive_retry_can_resolve_a_first_attempt_mismatch(self):
        (self.root/'tests/focus-stack-regression.mjs').write_text(FAKE_RUNNER.replace(
            'const mode=process.env.RAW_EDITOR_TEST_MODE;', "const mode='vary_by_slot';"))
        code, _, err=self.matrix()
        self.assertEqual(code,0,err)
        report=json.loads((self.root/'output/acceptance/task-2/matrix-m1-medium/matrix.json').read_text())
        self.assertEqual(report['cases'][0]['status'],'pass_after_exclusive_retry')

    def test_cancelled_run_terminates_its_runner_and_records_interruption(self):
        proc=self.spawn('run.py','--binary','fixture','--lane','task-2','--dataset','sleeping','--subset','sleeping-9-6479',
                        '--label','cancel','--env','RAW_EDITOR_TEST_MODE=wait')
        dest=self.root/'output/acceptance/task-2/sleeping-9-6479-cancel'
        pidfile=dest/'run/native.pid'
        self.wait_for(pidfile.exists)
        pid=int(pidfile.read_text())
        proc.terminate();proc.communicate(timeout=8)
        self.assertEqual(proc.returncode,143)
        self.assertEqual(json.loads((dest/'analysis.json').read_text())['status'],'cancelled')
        with self.assertRaises(ProcessLookupError): os.kill(pid,0)
        self.assertNotIn('busy',self.run_tool('slot.py','status')[1])

    def test_cancelled_matrix_does_not_launch_a_retry(self):
        (self.root/'tests/focus-stack-regression.mjs').write_text(FAKE_RUNNER.replace(
            'const mode=process.env.RAW_EDITOR_TEST_MODE;', "const mode='wait';"))
        proc=self.spawn('matrix.py','--tier','medium','--binary','fixture','--lane','task-2','--label','cancel')
        directory=self.root/'output/acceptance/task-2/matrix-cancel-medium'
        pidfile=directory/'nine-attempt-1/run/native.pid'
        self.wait_for(pidfile.exists)
        pid=int(pidfile.read_text())
        proc.terminate();proc.communicate(timeout=8)
        self.assertEqual(proc.returncode,143)
        self.assertEqual(json.loads((directory/'matrix.json').read_text())['status'],'cancelled')
        self.assertFalse((directory/'nine-attempt-2').exists())
        with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_monitor_failure_stops_the_command(self):
        env=dict(self.env,PYTHONPATH=str(self.script))
        code='from pathlib import Path;from monitor import execute;from unittest.mock import patch;import os,sys,json;'+"\nwith patch('monitor.tree_rss',side_effect=OSError('sampling refused')):\n print(json.dumps(execute([sys.executable,'-c','import time;time.sleep(30)'],cwd=Path.cwd(),env=os.environ,log_path=Path('monitor-fail.log'))))"
        result=subprocess.run([PYTHON,'-c',code],env=env,cwd=self.root,capture_output=True,text=True,timeout=8)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('sampling refused',json.loads(result.stdout)['monitor_error'])

    def test_live_monitor_kills_an_over_limit_command(self):
        env=dict(self.env,PYTHONPATH=str(self.script))
        code='from pathlib import Path;from monitor import execute;import os,sys,json;print(json.dumps(execute([sys.executable,"-c","import time;time.sleep(30)"],cwd=Path.cwd(),env=os.environ,log_path=Path("monitor.log"),threshold=1)))'
        result=subprocess.run([PYTHON,'-c',code],env=env,cwd=self.root,capture_output=True,text=True,timeout=8)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(json.loads(result.stdout)['memory_threshold_exceeded'])
        self.assertEqual(json.loads(result.stdout)['exit_code'],143)

    def test_contention_covers_a_build_that_started_and_ended_during_run(self):
        env=dict(self.env,PYTHONPATH=str(self.script))
        folder=self.root/'output/acceptance/.slots'
        now=time.time()
        (folder/'history.jsonl').write_text(json.dumps(dict(slot='build',pid=-2,lane='build',what='finished',state='running',started_epoch=now-5,ended_epoch=now-2))+'\n')
        code=f'from run import contention;print(contention(dict(slot="run",pid=-1,started_epoch={now-10}),{now}))'
        result=subprocess.run([PYTHON,'-c',code],cwd=self.root,env=env,capture_output=True,text=True)
        self.assertIn('build: finished',result.stdout)
        result=subprocess.run([PYTHON,'-c',code.replace('slot="run"','slot="exclusive"')],cwd=self.root,env=env,capture_output=True,text=True)
        self.assertEqual(result.stdout.strip(),'[]')


# Only inherit the fixture helpers, not another copy of the 17 task-1 cases.
for name in tuple(dir(fixtures.ToolsTest)):
    if name.startswith('test_') and name not in RunsTest.__dict__:
        setattr(RunsTest,name,None)

if __name__=='__main__':
    unittest.main(verbosity=2)
