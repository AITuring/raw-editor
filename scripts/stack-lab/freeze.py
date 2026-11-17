#!/usr/bin/env python3
"""Freeze a release library-test executable while holding the build slot.

    freeze.py --label candidate-r1 [--worktree .worktrees/task-1]
    freeze.py --label base-2026-11-12 --adopt output/acceptance/base-p1/candidate-release-binary

Adoption requires an adjacent build.json with the artifact's SHA, original
commit and source_manifest. It never attributes an old executable to HEAD.
Labels are immutable; failed attempts clean up their unpublished directory.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from common import BINARIES, SHARED_TARGET, SLOTS, repo_root, sha, slot_holders, valid_label, write

HERE = Path(__file__).resolve().parent
WRAPPER = '''#!/usr/bin/env python3
"""Run the frozen binary and emit child resource usage in bytes."""
import json, resource, subprocess, sys, time
from pathlib import Path
binary = Path(__file__).with_name("candidate-release-binary")
started = time.monotonic()
child = subprocess.run([str(binary), *sys.argv[1:]])
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
print("NATIVE_PROCESS_USAGE " + json.dumps({"method": "getrusage(RUSAGE_CHILDREN), one frozen Rust child", "max_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024), "wall_seconds": time.monotonic() - started, "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime, "child_exit_code": child.returncode}), file=sys.stderr, flush=True)
raise SystemExit(child.returncode if child.returncode >= 0 else 128 - child.returncode)
'''


class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def interrupted(signum, _frame):
    raise Interrupted(signum)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def snapshot(root):
    # Include build.rs, Cargo manifests/lock/config, resources and new untracked
    # inputs as well as Rust sources. Ignored target/output files are excluded.
    paths = git(root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--',
                'src-tauri', '.cargo', 'Cargo.toml', 'Cargo.lock', 'rust-toolchain', 'rust-toolchain.toml')
    result = []
    for relative in sorted(set(os.fsdecode(p) for p in paths.split(b'\0') if p)):
        path = root / relative
        result.append({'path': relative, 'sha256': sha(path) if path.is_file() else None})
    return result


def adopt_provenance(root, artifact):
    evidence_path = artifact.with_name('build.json')
    if not evidence_path.is_file():
        raise ValueError('adoption requires adjacent build.json with sha256, commit and source_manifest')
    evidence = json.loads(evidence_path.read_text())
    if not isinstance(evidence, dict):
        raise ValueError('adopted build.json must contain an object')
    digest = sha(artifact)
    if digest != evidence.get('sha256'):
        raise ValueError('adopted executable SHA does not match its build.json')
    commit = evidence.get('commit', '')
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('adopted build.json must name a full original commit SHA')
    git(root, 'cat-file', '-e', commit + '^{commit}')
    manifest = evidence.get('source_manifest')
    if not isinstance(manifest, list) or not manifest:
        raise ValueError('adopted build.json has no source manifest')
    seen = set()
    for item in manifest:
        if not isinstance(item, dict):
            raise ValueError('invalid adopted source manifest entry')
        path = item.get('path', '')
        digest_item = item.get('sha256', '')
        if (not isinstance(path, str) or not path or Path(path).is_absolute() or '..' in Path(path).parts
                or path in seen or not isinstance(digest_item, str) or len(digest_item) != 64
                or any(c not in '0123456789abcdef' for c in digest_item)):
            raise ValueError('invalid adopted source manifest')
        seen.add(path)
    return dict(commit=commit, source_manifest=manifest,
                uncommitted_source_changes=evidence.get('uncommitted_source_changes'),
                provenance_kind='adopted_build_record', provenance_path=str(evidence_path),
                provenance_sha256=sha(evidence_path)), digest


def compile_artifact(root, destination):
    commit = git(root, 'rev-parse', 'HEAD').decode().strip()
    before = snapshot(root)
    dirty = git(root, 'status', '--porcelain', '--untracked-files=all', '--', 'src-tauri', '.cargo').decode().strip()
    env = dict(os.environ, CARGO_TARGET_DIR=str(SHARED_TARGET))
    command = ['cargo', 'test', '--release', '--lib', '--offline', '--no-run', '--message-format=json']
    log = destination / 'cargo-build.jsonl'
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=root / 'src-tauri', env=env, stdout=stream)
    if result.returncode:
        raise ValueError(f'build failed (exit {result.returncode})')
    artifacts = set()
    for line in log.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if (record.get('reason') == 'compiler-artifact' and record.get('executable')
                and record.get('profile', {}).get('test')
                and {'lib', 'rlib', 'cdylib', 'staticlib', 'dylib'}.intersection(
                    record.get('target', {}).get('kind', []))):
            artifacts.add(Path(record['executable']).resolve())
    if len(artifacts) != 1:
        raise ValueError(f'expected one library-test executable, found {len(artifacts)}')
    if before != snapshot(root) or commit != git(root, 'rev-parse', 'HEAD').decode().strip():
        raise ValueError('source/commit changed during build; refusing ambiguous provenance')
    artifact = artifacts.pop()
    return artifact, dict(commit=commit, source_manifest=before,
                          uncommitted_source_changes=bool(dirty), provenance_kind='built_from_snapshot',
                          build_command=command, build_environment={key: env.get(key) for key in
                          ('CARGO_TARGET_DIR', 'CARGO_BUILD_JOBS', 'RAYON_NUM_THREADS', 'RUSTFLAGS', 'MACOSX_DEPLOYMENT_TARGET')}), sha(artifact)


def freeze(args, root):
    destination = BINARIES / args.label
    if os.path.lexists(destination):
        raise ValueError(f'{destination} exists; labels are immutable, pick a new one')
    BINARIES.mkdir(parents=True, exist_ok=True)
    # A label is visible only after every file and final build.json are ready.
    with tempfile.TemporaryDirectory(prefix=f'.{args.label}-', dir=BINARIES) as temporary:
        staging = Path(temporary) / 'bundle'
        staging.mkdir()
        if args.adopt:
            artifact = Path(args.adopt).resolve()
            provenance, expected = adopt_provenance(root, artifact)
        else:
            artifact, provenance, expected = compile_artifact(root, staging)
        if not artifact.is_file() or not os.access(artifact, os.X_OK):
            raise ValueError('artifact is not an executable file')
        binary = staging / 'candidate-release-binary'
        shutil.copyfile(artifact, binary)
        if sha(binary) != expected or sha(artifact) != expected:
            raise ValueError('executable changed while freezing')
        binary.chmod(0o555)
        wrapper = staging / 'native-process-wrapper.py'
        wrapper.write_text(WRAPPER)
        wrapper.chmod(0o555)
        build = dict(provenance, label=args.label, sha256=expected, worktree=str(root),
                     artifact=str(artifact), frozen_at=datetime.now(timezone.utc).isoformat(),
                     frozen_by_commit=git(root, 'rev-parse', 'HEAD').decode().strip(),
                     wrapper_sha256=sha(wrapper), slot=os.environ['STACK_LAB_SLOT'])
        write(staging / 'build.json', build)
        (staging / 'build.json').chmod(0o444)
        # All freeze clients serialize through the same build lock, including adoption.
        if os.path.lexists(destination):
            raise ValueError('label was created while freezing; refusing overwrite')
        staging.rename(destination)
    print(json.dumps({'label': args.label, 'sha256': expected, 'commit': build['commit'],
                      'provenance_kind': build['provenance_kind'], 'directory': str(destination)}))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--label', required=True)
    parser.add_argument('--worktree', default='.')
    parser.add_argument('--lane', default='L0')
    parser.add_argument('--adopt', help='existing executable, with adjacent build.json provenance')
    args = parser.parse_args(argv)
    try:
        valid_label(args.label)
        root = repo_root(Path(args.worktree).resolve())
        common_dir = Path(git(root, 'rev-parse', '--path-format=absolute', '--git-common-dir').decode().strip())
        if common_dir.resolve() != (BINARIES.parents[2] / '.git').resolve():
            raise ValueError('--worktree must share this repository Git common directory')
        if not (root / 'src-tauri/Cargo.toml').is_file():
            raise ValueError('worktree must belong to RAW Editor')
        if os.path.lexists(BINARIES / args.label):
            raise ValueError('label already exists; labels are immutable, pick a new one')
        if os.environ.get('STACK_LAB_HOLDER'):
            holders = slot_holders()
            if (os.environ.get('STACK_LAB_SLOT') not in ('build', 'exclusive') or 'build' not in holders
                    or Path(os.environ['STACK_LAB_HOLDER']) != SLOTS / (
                        'run.json' if os.environ['STACK_LAB_SLOT'] == 'exclusive' else 'build.json')
                    or holders['build']['pid'] != os.getppid()):
                raise ValueError('freeze requires its supervising build/exclusive slot')
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(signum, interrupted)
            freeze(args, root)
            return 0
        command = [sys.executable, str(HERE / 'slot.py'), 'build', '--lane', args.lane,
                   '--what', f'freeze {args.label}', '--', sys.executable, str(Path(__file__).resolve()), *argv]
        # Replace the launcher so TERM/INT of `freeze.py` reaches the slot
        # supervisor, which owns cancellation and the entire command group.
        os.execv(sys.executable, command)
    except Interrupted as exc:
        return 128 + exc.signum
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'[freeze] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
