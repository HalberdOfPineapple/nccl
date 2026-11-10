#!/usr/bin/env python3
"""Plan 01 E1.7: runs real NCCL collectives on GPUs 4-7 for several rounds, with the same guards as run_session.py.

Each run forces one kernel with NCCL_SYM_KERNEL and writes the NCCL log to its own file, so the result
row printed by nccl-tests stays readable. A run counts only if its log names the forced kernel.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

from run_session import DEVICES, OCCUPIED_EXIT, RUN_TIMEOUT_SECONDS, OccupiedError, ensure_idle, snapshot

TESTS = Path('/root/nccl-tests/build')
NCCL_LIBRARY = Path(__file__).resolve().parents[5] / 'build/lib'
COLLECTIVES = {
    'all_gather': ('all_gather_perf', 'allgather:AllGather_STMC', 'AllGather_STMC'),
    'reduce_scatter': ('reduce_scatter_perf', 'reducescatter:ReduceScatter_LDMC', 'ReduceScatter_LDMC'),
    'all_reduce': ('all_reduce_perf', 'allreduce:AllReduce_RSxLDMC_AGxSTMC', 'AllReduce_RSxLDMC_AGxSTMC'),
}
RESULT_ROW = re.compile(r'^\s*(\d+)\s+(\d+)\s+float\s+(\S+)\s+(-?\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$', re.MULTILINE)


def run_one(collective, size, output_stem):
    """One guarded invocation; returns its summary row."""
    program, selector, kernel = COLLECTIVES[collective]
    locks = []
    for device in DEVICES.split(','):
        handle = open(f'/tmp/nccl_thread_store_gpu_{device}.lock', 'a')
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    ensure_idle()
    output_stem.with_suffix('.before.json').write_text(json.dumps(snapshot()))
    nccl_log = output_stem.with_suffix('.nccl.log')
    environment = {
        **os.environ,
        'CUDA_VISIBLE_DEVICES': DEVICES,
        'LD_LIBRARY_PATH': str(NCCL_LIBRARY),
        'NCCL_SYM_KERNEL': selector,
        'NCCL_DEBUG': 'INFO',
        'NCCL_DEBUG_SUBSYS': 'TUNING',
        'NCCL_DEBUG_FILE': str(nccl_log),
    }
    command = [str(TESTS / program), '-t', '1', '-g', '4', '-b', size, '-e', size, '-f', '2', '-n', '20', '-w', '5', '-c', '1', '-R', '2', '-d', 'float', '-o', 'sum']
    result = subprocess.run(command, text=True, capture_output=True, env=environment, timeout=RUN_TIMEOUT_SECONDS)
    output_stem.with_suffix('.stdout.log').write_text(result.stdout + result.stderr)
    output_stem.with_suffix('.after.json').write_text(json.dumps(snapshot()))
    log_text = nccl_log.read_text(errors='replace') if nccl_log.exists() else ''
    row = RESULT_ROW.search(result.stdout)
    summary = {
        'run': output_stem.name,
        'collective': collective,
        'return_code': result.returncode,
        'kernel_lines': len(re.findall(rf'-> Kernel {kernel} ', log_text)),
        'other_kernel_lines': len(re.findall(r'-> Kernel (\S+) ', log_text)) - len(re.findall(rf'-> Kernel {kernel} ', log_text)),
        'channel_values': sorted({int(value) for value in re.findall(r'-> Kernel \S+ nchannels (\d+)', log_text)}),
        'thread_values': sorted({int(value) for value in re.findall(r'-> Kernel \S+ nchannels \d+ nthreads (\d+)', log_text)}),
    }
    if row:
        summary.update(size_bytes=int(row[1]), out_of_place_algbw_gbps=float(row[6]), out_of_place_busbw_gbps=float(row[7]), out_of_place_wrong=int(row[8]), in_place_algbw_gbps=float(row[10]), in_place_busbw_gbps=float(row[11]), in_place_wrong=int(row[12]))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw', type=Path)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--size', default='8G')
    arguments = parser.parse_args()
    arguments.raw.mkdir(parents=True, exist_ok=True)
    labels = list(COLLECTIVES)
    rows = []
    try:
        for round_index in range(arguments.rounds):
            shift = round_index % len(labels)
            for collective in labels[shift:] + labels[:shift]:
                row = run_one(collective, arguments.size, arguments.raw / f'e1_7_{collective}_R{round_index + 1}')
                rows.append(row)
                print(json.dumps(row), flush=True)
                if row['return_code'] or 'out_of_place_busbw_gbps' not in row:
                    print(f'Stopping after a failed run: {row["run"]}', flush=True)
                    return 1
    except OccupiedError as error:
        (arguments.raw / 'OCCUPIED.txt').write_text(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} {error}\n')
        print(f'OCCUPIED: {error}', flush=True)
        return OCCUPIED_EXIT
    except BlockingIOError as error:
        (arguments.raw / 'OCCUPIED.txt').write_text(f'Lock held: {error}\n')
        print(f'OCCUPIED: another guarded experiment holds a GPU lock: {error}', flush=True)
        return OCCUPIED_EXIT
    finally:
        arguments.summary.write_text(json.dumps(rows, indent=1) + '\n')
    for collective in labels:
        values = [row['out_of_place_busbw_gbps'] for row in rows if row['collective'] == collective]
        print(collective, 'median busBW', statistics.median(values), flush=True)
    print('SESSION_DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
