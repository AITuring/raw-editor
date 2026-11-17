"""Run one command group, sample its process-tree RSS, and handle cancellation."""
import os
import signal
import subprocess
import time

from common import MEMORY_GATE_BYTES


class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def interrupt(signum, _frame):
    raise Interrupted(signum)


def tree_rss(pid):
    result = subprocess.run(['ps', '-axo', 'pid=,ppid=,rss='], capture_output=True, text=True, check=True)
    rows = [tuple(map(int, line.split())) for line in result.stdout.splitlines() if line.strip()]
    children = {pid}
    while True:
        expanded = children | {p for p, parent, _ in rows if parent in children}
        if expanded == children:
            break
        children = expanded
    return sum(rss * 1024 for p, _, rss in rows if p in children)


def stop(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def execute(command, *, cwd, env, log_path, threshold=MEMORY_GATE_BYTES):
    started = time.time()
    child = None
    peak = 0
    samples = 0
    error = None
    exceeded = False
    cancelled = None
    code = 127
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        with log_path.open('x') as log:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            while True:
                try:
                    measured = tree_rss(child.pid)
                    if measured > 0:
                        samples += 1
                        peak = max(peak, measured)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    error = str(exc)
                    break  # no unmonitored real-material run
                if peak > threshold:
                    exceeded = True
                    break
                result = child.poll()
                if result is not None:
                    code = result if result >= 0 else 128 - result
                    break
                time.sleep(0.5)
    except Interrupted as exc:
        cancelled = exc.signum
        code = 128 + exc.signum
    except OSError as exc:
        error = str(exc)
    finally:
        # Repeated signals must not bypass group cleanup or leave a hidden runner.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        if child is not None:
            stop(child)
            if not cancelled and (error or exceeded):
                code = child.returncode if child.returncode >= 0 else 128 - child.returncode
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return dict(exit_code=code, started_epoch=started, ended_epoch=time.time(),
                wall_seconds=time.time() - started, sampled_peak_rss_bytes=peak or None,
                rss_sample_count=samples, rss_method='ps process-tree sum, KiB converted to bytes, 0.5s interval',
                monitor_error=error, memory_threshold_exceeded=exceeded, cancelled_signal=cancelled)
