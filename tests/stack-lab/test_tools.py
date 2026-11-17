"""Process-level regressions for slot locking and frozen-artifact integrity.

Runs in throwaway git repositories, with no builds or changes to real datasets.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'scripts/stack-lab'
PYTHON = sys.executable
WAIT = '''import pathlib,sys,time,json,os
p=pathlib.Path(sys.argv[1]); p.write_text(json.dumps(dict(pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),env=dict(os.environ),time=time.time())))
while not pathlib.Path(sys.argv[2]).exists(): time.sleep(.02)
'''


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='stack lab test ')
        self.root = Path(self.temp.name).resolve()
        self.script = self.root / 'scripts/stack-lab'
        self.script.mkdir(parents=True)
        for name in ('common.py', 'slot.py', 'freeze.py'):
            shutil.copyfile(SOURCE / name, self.script / name)
        (self.root / 'src-tauri/src').mkdir(parents=True)
        (self.root / 'src-tauri/src/lib.rs').write_text('// fixture\n')
        (self.root / 'src-tauri/Cargo.toml').write_text('[package]\nname="fixture"\nversion="0.1.0"\n')
        (self.root / '.gitignore').write_text('/src-tauri/target/\n/output/\n')
        self.git('init', '-q')
        self.git('add', '.')
        self.git('-c', 'user.name=Stack Lab Test', '-c', 'user.email=stack-lab@example.invalid',
                 '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture')
        self.commit = self.git('rev-parse', 'HEAD').strip()
        self.processes = []
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('STACK_LAB_')}
        self.env['PYTHONDONTWRITEBYTECODE'] = '1'
        self.env['PYTHONUNBUFFERED'] = '1'
        self.old = self.root / 'old executable'
        self.old.mkdir()
        self.artifact = self.old / 'candidate-release-binary'
        self.artifact.write_text('#!/bin/sh\nprintf "frozen fixture\\n"\n')
        self.artifact.chmod(0o755)
        self.evidence = dict(commit=self.commit, sha256=self.digest(self.artifact),
                             source_manifest=[dict(path='src-tauri/src/lib.rs', sha256=self.digest(self.root / 'src-tauri/src/lib.rs'))])
        self.write_evidence()

    def tearDown(self):
        for proc in self.processes:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.communicate(timeout=4)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, text=True, stderr=subprocess.PIPE)

    def digest(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_evidence(self):
        (self.old / 'build.json').write_text(json.dumps(self.evidence))

    def spawn(self, tool, *args, env=None):
        proc = subprocess.Popen([PYTHON, str(self.script / tool), *args], cwd=self.root,
                                env=env or self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(proc)
        return proc

    def run_tool(self, tool, *args, env=None):
        proc = self.spawn(tool, *args, env=env)
        out, err = proc.communicate(timeout=15)
        return proc.returncode, out, err

    def wait_for(self, predicate, description='condition'):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.02)
        self.fail('timed out waiting for ' + description)

    def held(self, slot, lane):
        path = self.root / 'output/acceptance/.slots' / (slot + '.json')
        return path.exists() and json.loads(path.read_text())['lane'] == lane

    def holder(self, slot='run', lane='first', env=None):
        started = self.root / (lane + '.started')
        release = self.root / (lane + '.release')
        args = [slot, '--lane', lane, '--what', 'test ' + lane]
        if slot == 'run':
            args += ['--estimate-gib', '12']
        proc = self.spawn('slot.py', *args, '--', PYTHON, '-c', WAIT, str(started), str(release), env=env)
        self.wait_for(started.exists, lane)
        self.wait_for(lambda: self.held('build' if slot == 'build' else 'run', lane))
        return proc, started, release

    def adopt(self, label='candidate'):
        return self.run_tool('freeze.py', '--label', label, '--adopt', str(self.artifact))

    def test_same_slot_queues_status_and_exit_code(self):
        for slot in ('build', 'run'):
            with self.subTest(slot=slot):
                first, started, release = self.holder(slot, 'first-' + slot)
                result = self.root / (slot + '-second.started')
                options = ['--estimate-gib', '12'] if slot == 'run' else []
                second = self.spawn('slot.py', slot, '--lane', 'second', *options, '--', PYTHON, '-c',
                                    'import pathlib,sys;pathlib.Path(sys.argv[1]).touch();sys.exit(7)', str(result))
                # The waiting message proves it reached the lock, not just that it started slowly.
                self.assertIn('waiting for ' + slot, second.stderr.readline())
                self.assertFalse(result.exists())
                code, out, _ = self.run_tool('slot.py', 'status')
                self.assertEqual(code, 0)
                self.assertIn('test first-' + slot, out)
                release.touch()
                first.communicate(timeout=8)
                second.communicate(timeout=8)
                self.assertEqual(second.returncode, 7)
                self.assertTrue(result.exists())
                self.assertIn(slot, json.loads(started.read_text())['env']['STACK_LAB_SLOT'])
        self.assertNotIn('busy', self.run_tool('slot.py', 'status')[1])

    def test_build_and_run_overlap_and_record_contention(self):
        build, _, build_release = self.holder('build', 'compile')
        run, _, run_release = self.holder('run', 'render')
        self.assertIsNone(build.poll())
        data = json.loads((self.root / 'output/acceptance/.slots/run.json').read_text())
        self.assertEqual(data['contended_with'], ['compile: test compile'])
        build_release.touch()
        run_release.touch()
        build.communicate(timeout=8)
        run.communicate(timeout=8)

    def test_exclusive_reserves_both_and_waiter_cancellation_releases_partial_lock(self):
        build, _, release = self.holder('build', 'compile')
        exclusive = self.spawn('slot.py', 'exclusive', '--lane', 'exclusive', '--', PYTHON, '-c', 'pass')
        self.assertIn('waiting for build', exclusive.stderr.readline())
        self.assertTrue(self.held('run', 'exclusive'))
        exclusive.terminate()
        exclusive.communicate(timeout=8)
        self.assertEqual(exclusive.returncode, 143)
        run, _, run_release = self.holder('run', 'render')
        run_release.touch()
        release.touch()
        build.communicate(timeout=8)
        run.communicate(timeout=8)
        held, _, release = self.holder('exclusive', 'both')
        options = ['run', '--estimate-gib', '12']
        waiting = self.spawn('slot.py', *options, '--', PYTHON, '-c', 'pass')
        self.assertIn('waiting for run', waiting.stderr.readline())
        status = self.run_tool('slot.py', 'status')[1]
        self.assertEqual(status.count('busy'), 2)
        release.touch()
        held.communicate(timeout=8)
        waiting.communicate(timeout=8)
        self.assertEqual(waiting.returncode, 0)

    def test_caps_nice_and_lower_requested_values(self):
        for slot in ('build', 'run', 'exclusive'):
            with self.subTest(slot=slot):
                env = dict(self.env, CARGO_BUILD_JOBS='99', RAYON_NUM_THREADS='99', MAKEFLAGS='-j99')
                proc, started, release = self.holder(slot, slot, env=env)
                data = json.loads(started.read_text())
                expected = {'build': {'CARGO_BUILD_JOBS': '3'}, 'run': {'RAYON_NUM_THREADS': '5'},
                            'exclusive': {'CARGO_BUILD_JOBS': '6', 'RAYON_NUM_THREADS': '8'}}[slot]
                for key, value in expected.items():
                    self.assertEqual(data['env'][key], value)
                self.assertGreaterEqual(data['nice'], 10)
                if slot != 'run':
                    self.assertNotIn('MAKEFLAGS', data['env'])
                release.touch()
                proc.communicate(timeout=8)
        env = dict(self.env, RAYON_NUM_THREADS='2')
        proc, started, release = self.holder('run', 'lower', env=env)
        self.assertEqual(json.loads(started.read_text())['env']['RAYON_NUM_THREADS'], '2')
        release.touch()
        proc.communicate(timeout=8)

    def test_resource_policy_and_invalid_command(self):
        for estimate in (None, '20.1', 'nan', 'inf', '-1', '0'):
            args = ['run'] + (['--estimate-gib', estimate] if estimate is not None else [])
            code, _, _ = self.run_tool('slot.py', *args, '--', PYTHON, '-c', 'pass')
            self.assertNotEqual(code, 0)
        self.assertEqual(self.run_tool('slot.py', 'run', '--estimate-gib', '20', '--', PYTHON, '-c', 'pass')[0], 0)
        self.assertEqual(self.run_tool('slot.py', 'build', '--', '/nonexistent-command')[0], 127)
        self.assertNotIn('busy', self.run_tool('slot.py', 'status')[1])
        # The command's arguments must never be parsed as slot options.
        code, out, _ = self.run_tool('slot.py', 'build', '--lane', 'correct', '--', PYTHON, '-c',
                                    'import sys;print(sys.argv[1:])', '--lane', 'child')
        self.assertEqual(code, 0)
        self.assertIn("['--lane', 'child']", out)

    def test_signals_stop_child_and_release(self):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            proc, started, _ = self.holder('run', 'signal-' + str(sig))
            pid = json.loads(started.read_text())['pid']
            proc.send_signal(sig)
            proc.communicate(timeout=8)
            self.assertEqual(proc.returncode, 128 + sig)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            self.assertNotIn('busy', self.run_tool('slot.py', 'status')[1])

    def test_sigkill_supervisor_keeps_child_lock(self):
        proc, started, release = self.holder('run', 'orphan')
        pid = json.loads(started.read_text())['pid']
        proc.kill()
        # communicate would block while the child still owns inherited stdout/stderr.
        self.wait_for(lambda: proc.poll() is not None)
        waiter = self.spawn('slot.py', 'run', '--estimate-gib', '12', '--', PYTHON, '-c', 'pass')
        try:
            self.assertIn('waiting for run', waiter.stderr.readline())
            self.assertIn('orphan', self.run_tool('slot.py', 'status')[1])
            release.touch()
            proc.communicate(timeout=8)
            waiter.communicate(timeout=8)
            self.assertEqual(waiter.returncode, 0)
            self.assertNotIn('busy', self.run_tool('slot.py', 'status')[1])
        finally:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_nested_slot_rejected_without_deadlock(self):
        code, _, err = self.run_tool('slot.py', 'build', '--', PYTHON, str(self.script / 'slot.py'),
                                     'build', '--', PYTHON, '-c', 'pass')
        self.assertNotEqual(code, 0)
        self.assertIn('nested', err)

    def test_adopt_provenance_readonly_wrapper_and_duplicate(self):
        # Freeze from a newer HEAD; artifact must still retain the original commit.
        (self.root / 'src-tauri/src/lib.rs').write_text('// newer source\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Stack Lab Test', '-c', 'user.email=stack-lab@example.invalid',
                 '-c', 'commit.gpgsign=false', 'commit', '-qm', 'newer')
        code, out, err = self.adopt()
        self.assertEqual(code, 0, err)
        directory = self.root / 'output/acceptance/binaries/candidate'
        metadata = json.loads((directory / 'build.json').read_text())
        self.assertEqual(metadata['commit'], self.commit)
        self.assertEqual(metadata['source_manifest'], self.evidence['source_manifest'])
        self.assertEqual(metadata['sha256'], self.evidence['sha256'])
        for name in ('candidate-release-binary', 'native-process-wrapper.py', 'build.json'):
            self.assertEqual((directory / name).stat().st_mode & 0o222, 0)
        result = subprocess.run([PYTHON, str(directory / 'native-process-wrapper.py')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('frozen fixture', result.stdout)
        usage = json.loads(result.stderr.split('NATIVE_PROCESS_USAGE ')[1])
        self.assertGreater(usage['max_rss_bytes'], 0)
        original = self.digest(directory / 'build.json')
        self.assertNotEqual(self.adopt()[0], 0)
        self.assertEqual(self.digest(directory / 'build.json'), original)
        self.assertIn('build', (self.root / 'output/acceptance/.slots/history.jsonl').read_text())

    def test_invalid_adoption_and_labels_publish_nothing(self):
        for label in ('../escape', '/absolute', '.', 'a/b', 'bad label', 'x;touch-pwned'):
            self.assertNotEqual(self.adopt(label)[0], 0)
        self.evidence['sha256'] = '0' * 64
        self.write_evidence()
        self.assertNotEqual(self.adopt()[0], 0)
        self.assertFalse((self.root / 'output/acceptance/binaries/candidate').exists())
        self.assertEqual(list((self.root / 'output/acceptance/binaries').iterdir()), [])
        (self.old / 'build.json').unlink()
        self.assertNotEqual(self.adopt()[0], 0)

    def test_malformed_adoption_metadata_is_rejected_cleanly(self):
        path = self.old / 'build.json'
        for metadata in ([], dict(self.evidence, source_manifest=['bad-entry']),
                         dict(self.evidence, commit='HEAD'),
                         dict(self.evidence, source_manifest=[dict(path='../escape', sha256='0' * 64)])):
            path.write_text(json.dumps(metadata))
            code, _, err = self.adopt('malformed')
            self.assertNotEqual(code, 0)
            self.assertNotIn('Traceback', err)
            self.assertFalse((self.root / 'output/acceptance/binaries/malformed').exists())

    def fake_cargo(self, mutate=False, fail=False, pause=False):
        folder = self.root / 'tool bin'
        folder.mkdir()
        cargo = folder / 'cargo'
        cargo.write_text('#!' + PYTHON + '\n' + f'''
import pathlib,os,json,sys,time
root=pathlib.Path.cwd().parent
(root/'cargo-env.json').write_text(json.dumps(dict(os.environ)))
(root/'cargo-entered').touch()
if {pause!r}:
 while not (root/'cargo-release').exists(): time.sleep(.02)
if {mutate!r}: (root/'src-tauri/src/lib.rs').write_text('// changed while building')
if {fail!r}: sys.exit(4)
p=pathlib.Path(os.environ['CARGO_TARGET_DIR'])/'release'/'library test binary'
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text('#!/bin/sh\\necho compiled fixture\\n');p.chmod(0o755)
print(json.dumps(dict(reason='compiler-artifact',executable=str(p),profile=dict(test=True),target=dict(kind=['staticlib','cdylib','rlib']))))
''')
        cargo.chmod(0o755)
        return dict(self.env, PATH=str(folder) + os.pathsep + self.env['PATH'], CARGO_BUILD_JOBS='99')

    def test_compile_uses_build_slot_and_snapshots_build_inputs(self):
        env = self.fake_cargo(pause=True)
        build = self.spawn('freeze.py', '--label', 'compiled', env=env)
        self.wait_for((self.root / 'cargo-entered').exists)
        waiting = self.spawn('freeze.py', '--label', 'adopted', '--adopt', str(self.artifact), env=env)
        self.assertIn('waiting for build', waiting.stderr.readline())
        self.assertFalse((self.root / 'output/acceptance/binaries/compiled').exists())
        (self.root / 'cargo-release').touch()
        _, err = build.communicate(timeout=8)
        self.assertEqual(build.returncode, 0, err)
        waiting.communicate(timeout=8)
        self.assertEqual(waiting.returncode, 0)
        data = json.loads((self.root / 'output/acceptance/binaries/compiled/build.json').read_text())
        self.assertEqual(data['commit'], self.commit)
        self.assertIn('src-tauri/Cargo.toml', [item['path'] for item in data['source_manifest']])
        env = json.loads((self.root / 'cargo-env.json').read_text())
        self.assertEqual(env['CARGO_BUILD_JOBS'], '3')
        self.assertEqual(env['CARGO_TARGET_DIR'], str(self.root / 'src-tauri/target'))

    def test_changed_source_during_build_is_rejected(self):
        code, _, err = self.run_tool('freeze.py', '--label', 'changed', env=self.fake_cargo(mutate=True))
        self.assertNotEqual(code, 0)
        self.assertIn('changed during build', err)
        self.assertFalse((self.root / 'output/acceptance/binaries/changed').exists())
        self.assertNotIn('busy', self.run_tool('slot.py', 'status')[1])

    def test_cancelled_freeze_releases_build_and_removes_staging(self):
        proc = self.spawn('freeze.py', '--label', 'cancelled', env=self.fake_cargo(pause=True))
        self.wait_for((self.root / 'cargo-entered').exists)
        proc.terminate()
        proc.communicate(timeout=8)
        self.assertEqual(proc.returncode, 143)
        self.assertNotIn('busy', self.run_tool('slot.py', 'status')[1])
        self.assertEqual(list((self.root / 'output/acceptance/binaries').iterdir()), [])
        self.assertEqual(self.adopt('cancelled')[0], 0)

    def test_failed_build_is_not_published(self):
        code, _, err = self.run_tool('freeze.py', '--label', 'failed', env=self.fake_cargo(fail=True))
        self.assertNotEqual(code, 0)
        self.assertIn('build failed', err)
        self.assertFalse((self.root / 'output/acceptance/binaries/failed').exists())
        self.assertEqual(list((self.root / 'output/acceptance/binaries').iterdir()), [])

    def test_simultaneous_duplicate_freezes_only_one_wins(self):
        first = self.spawn('freeze.py', '--label', 'duplicate', '--adopt', str(self.artifact))
        second = self.spawn('freeze.py', '--label', 'duplicate', '--adopt', str(self.artifact))
        first.communicate(timeout=8)
        second.communicate(timeout=8)
        self.assertEqual(sorted([first.returncode, second.returncode]), [0, 1])

    def test_verify_detects_binary_and_wrapper_tampering(self):
        self.assertEqual(self.adopt()[0], 0)
        check = 'from common import verify_binary;verify_binary("candidate")'
        env = dict(self.env, PYTHONPATH=str(self.script))
        directory = self.root / 'output/acceptance/binaries/candidate'
        for name in ('candidate-release-binary', 'native-process-wrapper.py'):
            path = directory / name
            content = path.read_bytes()
            path.chmod(0o755)
            path.write_bytes(content + b'\n# changed\n')
            result = subprocess.run([PYTHON, '-c', check], env=env, cwd=self.root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('modified', result.stderr)
            path.write_bytes(content)


if __name__ == '__main__':
    unittest.main(verbosity=2)
