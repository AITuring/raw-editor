#!/usr/bin/env python3
"""Run a command under the shared build/run locks (tasks.md §6.1).

    slot.py build --lane T1 --what "freeze candidate" -- python3 freeze.py ...
    slot.py run --lane T2 --estimate-gib 12 -- executable args...
    slot.py exclusive --lane T3 --what "first full run" -- executable args...
    slot.py status

Options belong before the mandatory -- separator. Unknown peak RSS and peaks
above 20 GiB require exclusive; 24 GiB is the separate measured acceptance gate.
"""
import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from common import CAPS, SLOTS, slot_holders, write

ORDER = {'build': ['build'], 'run': ['run'], 'exclusive': ['run', 'build']}


class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def interrupted(signum, _frame):
    raise Interrupted(signum)


def status():
    holders = slot_holders()
    for name in ('build', 'run'):
        holder = holders.get(name)
        if holder:
            age = max(0, time.time() - holder['started_epoch'])
            print(f'{name:5} busy {holder["lane"]} {holder["what"]} '
                  f'pid={holder["pid"]} child={holder.get("child_pid")} '
                  f'state={holder["state"]} {age / 60:.1f} min')
        else:
            print(f'{name:5} free')


def capped_env(slot, inherited):
    env = dict(inherited)
    for key, maximum in CAPS[slot].items():
        try:
            requested = int(env.get(key, maximum))
        except ValueError:
            requested = int(maximum)
        env[key] = str(min(requested, int(maximum)) if requested > 0 else int(maximum))
    # Keep native compiler parallelism under Cargo's same budget.
    if 'CARGO_BUILD_JOBS' in CAPS[slot]:
        env['CMAKE_BUILD_PARALLEL_LEVEL'] = env['CARGO_BUILD_JOBS']
        env.pop('MAKEFLAGS', None)
        env.pop('MFLAGS', None)
    return env


def lower_priority():
    # /usr/bin/nice can warn and still execute at normal priority. Fail closed
    # if the OS/sandbox refuses the required scheduling priority.
    os.setpriority(os.PRIO_PROCESS, 0, max(10, os.getpriority(os.PRIO_PROCESS, 0)))


def stop_group(child):
    # Also reap background descendants after the command leader has exited.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # A descendant may ignore TERM even if the group leader exited promptly.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('slot', choices=['build', 'run', 'exclusive', 'status'])
    parser.add_argument('--lane', default='unassigned')
    parser.add_argument('--what', default='')
    parser.add_argument('--estimate-gib', type=float, help='estimated peak; unknown or >20 requires exclusive')
    options, command = (argv[:argv.index('--')], argv[argv.index('--') + 1:]) if '--' in argv else (argv, [])
    args = parser.parse_args(options)
    if args.slot == 'status':
        if command:
            parser.error('status does not take a command')
        return args, command
    if not command:
        parser.error('missing command after --')
    if args.estimate_gib is not None and (not math.isfinite(args.estimate_gib) or args.estimate_gib <= 0):
        parser.error('--estimate-gib must be finite and positive')
    if args.slot == 'run' and (args.estimate_gib is None or args.estimate_gib > 20):
        parser.error('unknown peak or estimate >20 GiB requires exclusive (measured RSS gate remains 24 GiB)')
    if os.environ.get('STACK_LAB_HOLDER'):
        parser.error('nested slot acquisition is forbidden; run the command directly inside its existing slot')
    return args, command


def main(argv=None):
    args, command = parse(sys.argv[1:] if argv is None else argv)
    if args.slot == 'status':
        status()
        return 0
    SLOTS.mkdir(parents=True, exist_ok=True)
    env = capped_env(args.slot, os.environ)
    locks = []
    child = None
    code = 1
    holder = {'slot': args.slot, 'lane': args.lane, 'what': args.what, 'command': command,
              'cwd': os.getcwd(), 'estimate_gib': args.estimate_gib, 'pid': os.getpid(),
              'started_epoch': time.time(), 'started': datetime.now(timezone.utc).isoformat(),
              'state': 'acquiring', 'caps': {k: env[k] for k in CAPS[args.slot]}, 'contended_with': []}
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        for name in ORDER[args.slot]:
            lock = (SLOTS / f'{name}.lock').open('a')
            locks.append((name, lock, False))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                busy = slot_holders().get(name, {})
                print(f'[slot] waiting for {name} (held by {busy.get("lane", "?")}: '
                      f'{busy.get("what", "?")})', file=sys.stderr, flush=True)
                fcntl.flock(lock, fcntl.LOCK_EX)
            locks[-1] = (name, lock, True)
            write(SLOTS / f'{name}.json', holder)
        other = {name: h for name, h in slot_holders().items() if name not in ORDER[args.slot]}
        holder.update(state='running', started_epoch=time.time(),
                      contended_with=[h['lane'] + ': ' + h['what'] for h in other.values()])
        env['STACK_LAB_HOLDER'] = str(SLOTS / f'{ORDER[args.slot][0]}.json')
        env['STACK_LAB_SLOT'] = args.slot
        # Children inherit locks: a SIGKILL of this supervisor cannot let another
        # command into the slot while the original command is still running.
        child = subprocess.Popen(command, env=env, start_new_session=True, preexec_fn=lower_priority,
                                 pass_fds=tuple(lock.fileno() for _, lock, _ in locks))
        holder['child_pid'] = child.pid
        for name, _, _ in locks:
            write(SLOTS / f'{name}.json', holder)
        result = child.wait()
        code = result if result >= 0 else 128 - result
    except Interrupted as exc:
        code = 128 + exc.signum
    except (OSError, subprocess.SubprocessError) as exc:
        print(f'[slot] could not start command with required nice priority: {exc}', file=sys.stderr)
        code = 127
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if child is not None:
            stop_group(child)
        try:
            record = dict(holder, exit_code=code, ended_epoch=time.time(),
                          minutes=round((time.time() - holder['started_epoch']) / 60, 6))
            # Concurrent build/run supervisors append one complete line each.
            with (SLOTS / 'history.jsonl').open('a') as history:
                fcntl.flock(history, fcntl.LOCK_EX)
                history.write(json.dumps(record, ensure_ascii=False) + '\n')
        finally:
            for name, lock, acquired in reversed(locks):
                if acquired:
                    (SLOTS / f'{name}.json').unlink(missing_ok=True)
                # close, rather than LOCK_UN: inherited locks survive supervisor death.
                lock.close()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    return code


if __name__ == '__main__':
    sys.exit(main())
