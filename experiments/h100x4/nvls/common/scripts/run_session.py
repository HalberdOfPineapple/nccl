#!/usr/bin/env python3
"""Runs a list of mm_mix arms for several rounds, in one sequential session, with guards.

Every run holds locks on GPUs 4-7 and checks that they are idle first. If memory use or utilization
stays above the limits, the session stops with exit code 10 and writes OCCUPIED.txt, because that
means another user's job may have started on the allocated GPUs. Never touches any process.
"""
import argparse
import csv
import datetime
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

COMMON = Path(__file__).resolve().parents[1]
BINARY = COMMON / 'bin/mm_mix'
GPC_MAPS = COMMON.parents[3] / 'experiments/h100x2/thread_store/observation/data_store_congest/configs/gpc_map'
DEVICES = '4,5,6,7'
OCCUPIED_EXIT = 10
# GPU 6 has held about 122 MiB at 0% with no known owner since 2026-11-01; the user accepted it as not occupying.
IDLE_MEMORY_MIB = {'4': 100, '5': 100, '6': 150, '7': 100}
RUN_TIMEOUT_SECONDS = 180


class OccupiedError(RuntimeError):
    pass


def capture(command):
    result = subprocess.run(command, text=True, capture_output=True)
    return {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def read_state():
    fields = 'index,memory.used,utilization.gpu'
    text = subprocess.check_output(['nvidia-smi', '-i', DEVICES, f'--query-gpu={fields}', '--format=csv,noheader,nounits'], text=True)
    return [[value.strip() for value in row] for row in csv.reader(io.StringIO(text))]


def ensure_idle():
    # Our own context teardown can show for a few seconds, so only lasting use counts as occupied.
    occupied = []
    for attempt in range(40):
        occupied = [row for row in read_state() if int(row[1]) > IDLE_MEMORY_MIB[row[0]] or int(row[2]) > 0]
        if not occupied:
            return
        time.sleep(0.5)
    raise OccupiedError(f'GPUs not idle after 20 s (index, MiB, %): {occupied}')


def snapshot():
    return {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'gpu': capture(['nvidia-smi', '-i', DEVICES, '-q', '-x']),
        'link': capture(['nvidia-smi', 'nvlink', '-i', DEVICES, '-gt', 'd']),
    }


def run_one(config_path, output):
    """One guarded invocation: snapshots, telemetry, log, raw JSON. Returns False on a failed run."""
    locks = []
    for device in DEVICES.split(','):
        handle = open(f'/tmp/nccl_thread_store_gpu_{device}.lock', 'a')
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    ensure_idle()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f'Refusing to overwrite {output}')
    output.with_suffix('.before.json').write_text(json.dumps(snapshot()))
    telemetry = []
    stop_event = threading.Event()

    def sample_telemetry():
        fields = 'index,timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active'
        while not stop_event.is_set():
            telemetry.append(capture(['nvidia-smi', '-i', DEVICES, f'--query-gpu={fields}', '--format=csv,noheader,nounits'])['stdout'])
            stop_event.wait(0.25)

    sampler = threading.Thread(target=sample_telemetry)
    sampler.start()
    environment = {**os.environ, 'NCCL_DEBUG': 'WARN'}
    environment.pop('CUDA_VISIBLE_DEVICES', None)
    try:
        with output.with_suffix('.log').open('w') as log:
            result = subprocess.run([str(BINARY), str(config_path), str(output), str(GPC_MAPS)], stdout=log, stderr=log, timeout=RUN_TIMEOUT_SECONDS, env=environment)
        return_code = result.returncode
    except subprocess.TimeoutExpired:
        return_code = -1
    finally:
        stop_event.set()
        sampler.join()
        output.with_suffix('.telemetry.json').write_text(json.dumps(telemetry))
        output.with_suffix('.after.json').write_text(json.dumps(snapshot()))
    return return_code


def write_config(path, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(f'{key}={value}\n' for key, value in settings.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path, help='JSON: {"arms": {"label": {settings}}, "rounds": N, "rotate": true}')
    parser.add_argument('raw', type=Path, help='new raw directory of this session')
    parser.add_argument('configs', type=Path, help='directory that keeps one .cfg per arm')
    arguments = parser.parse_args()
    plan = json.loads(arguments.plan.read_text())
    labels = list(plan['arms'])
    for label in labels:
        write_config(arguments.configs / f'{label}.cfg', plan['arms'][label])
    order_log = arguments.raw / 'order.jsonl'
    arguments.raw.mkdir(parents=True, exist_ok=True)
    failures = 0
    try:
        for round_index in range(plan['rounds']):
            shift = round_index % len(labels) if plan.get('rotate', True) else 0
            for label in labels[shift:] + labels[:shift]:
                output = arguments.raw / f'{label}_R{round_index + 1}.json'
                if output.exists():
                    continue
                started = time.time()
                return_code = run_one(arguments.configs / f'{label}.cfg', output)
                with order_log.open('a') as log:
                    log.write(json.dumps({'label': label, 'round': round_index + 1, 'return_code': return_code, 'start_unix': started, 'seconds': time.time() - started}) + '\n')
                print(f'{label} R{round_index + 1} exit={return_code} {time.time() - started:.1f}s', flush=True)
                if return_code:
                    failures += 1
                    if failures >= plan.get('max_failures', 1):
                        print(f'Stopping after {failures} failed runs; see {output.with_suffix(".log")}', flush=True)
                        return 1
    except OccupiedError as error:
        (arguments.raw / 'OCCUPIED.txt').write_text(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} {error}\n')
        print(f'OCCUPIED: {error}', flush=True)
        return OCCUPIED_EXIT
    except BlockingIOError as error:
        (arguments.raw / 'OCCUPIED.txt').write_text(f'Lock held: {error}\n')
        print(f'OCCUPIED: another guarded experiment holds a GPU lock: {error}', flush=True)
        return OCCUPIED_EXIT
    print('SESSION_DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
