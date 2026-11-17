"""Shared helpers for the multi-agent stack lab (plan §5–§6).

All paths are absolute so the scripts behave the same from the main checkout
and from any `.worktrees/<lane>` worktree.
"""
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

# git --git-common-dir resolves to the main checkout even from a lane worktree.
_GIT_COMMON = Path(subprocess.check_output(
    ['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
    text=True).strip())
MAIN = _GIT_COMMON.parent
DATA = Path(os.environ.get('RAW_EDITOR_FOCUS_STACK_DATA_ROOT', '/Users/dp/Downloads/书画拼图'))
ACCEPTANCE = MAIN / 'output/acceptance'
SLOTS = ACCEPTANCE / '.slots'
BINARIES = ACCEPTANCE / 'binaries'
SHARED_TARGET = MAIN / 'src-tauri/target'
CHECKOUT = Path(__file__).resolve().parents[2]
MATRIX = CHECKOUT / 'tests/acceptance/regression-matrix.json'
GIB = 1024 ** 3
# Per-run memory gate from plan §1.3 (the Lanting 25,769,803,776 B threshold).
MEMORY_GATE_BYTES = 25_769_803_776

# Resource caps that keep one build and one run below the whole machine
# (10 cores: 3 compile jobs + 5 render threads leave 2 cores free).
CAPS = {
    'build': {'CARGO_BUILD_JOBS': '3'},
    'run': {'RAYON_NUM_THREADS': '5'},
    'exclusive': {'RAYON_NUM_THREADS': '8', 'CARGO_BUILD_JOBS': '6'},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Publish JSON atomically so status/readers never see partial metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def valid_label(label):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', label):
        raise ValueError('label must be 1–128 ASCII letters/digits/._-, starting with a letter or digit')
    return label


def repo_root(start=None):
    """Repository (or worktree) that contains `start`, defaulting to the cwd."""
    return Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=start or Path.cwd(), text=True).strip())


def binary_dir(label):
    path = BINARIES / valid_label(label)
    if not (path / 'build.json').is_file():
        raise SystemExit(f'unknown frozen binary {label!r}; run freeze.py first ({path})')
    return path


def verify_binary(label):
    directory = binary_dir(label)
    build = read(directory / 'build.json')
    if build.get('label') != label or sha(directory / 'candidate-release-binary') != build['sha256']:
        raise SystemExit(f'frozen binary {label} was modified')
    if sha(directory / 'native-process-wrapper.py') != build['wrapper_sha256']:
        raise SystemExit(f'frozen wrapper {label} was modified')
    return directory, build


def dataset_entry(dataset_id):
    manifest = read(CHECKOUT / 'tests/acceptance/focus-stack-datasets.json')
    return next(item for item in manifest['datasets'] if item['id'] == dataset_id)


def dataset_files(dataset_id, subset=None):
    entry = dataset_entry(dataset_id)
    folders = [DATA / entry['directory']] if 'directory' in entry else [DATA / name for name in entry['directories']]
    files = []
    for folder in folders:
        files += sorted(
            path for path in folder.iterdir()
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in entry['sourceExtensions']
            and not any(re.fullmatch(re.escape(pattern).replace(r'\*', '.*'), path.name, flags=re.IGNORECASE) for pattern in entry.get('excludePatterns', []))
        )
    files.sort(key=str)
    if len(files) != entry['expectedSourceCount']:
        raise SystemExit(f'{dataset_id}: {len(files)} sources, manifest expects {entry["expectedSourceCount"]}')
    if subset:
        sub = next(item for item in entry['regressionSubsets'] if item['id'] == subset)
        start = sub.get('start', 0)
        files = files[start:start + sub.get('count', sub.get('first'))]
    if entry.get('inputOrder') == 'reverse':
        files.reverse()
    return entry, files


def usage_from_logs(directory):
    usage = None
    for log in Path(directory).glob('*.log'):
        found = re.findall(r'NATIVE_PROCESS_USAGE (\{[^\n]+\})', log.read_text(errors='replace'))
        if found:
            usage = json.loads(found[-1])
    return usage


def slot_holders():
    """Read only holders whose kernel lock is still held, including orphan children."""
    holders = {}
    for name in ('build', 'run'):
        path = SLOTS / f'{name}.json'
        lock_path = SLOTS / f'{name}.lock'
        if not path.is_file() or not lock_path.is_file():
            continue
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                try:
                    holders[name] = read(path)
                except (ValueError, FileNotFoundError):
                    pass
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)
    return holders
