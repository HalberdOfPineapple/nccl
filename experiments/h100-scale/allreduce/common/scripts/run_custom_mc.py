#!/usr/bin/env python3
"""Plan 06: runs the patched contrib example program (custom multicast AllReduce kernels), one guarded run at a time.

A session file (JSON) lists cells; this script runs them in rounds, rotating the cell order between rounds, and
appends one summary JSONL line per run. One run is one `mpirun -np N` of the program with one CTA range and
one size range; the program measures the selected kernels at every CTA count and size.

Each run: lock every GPU of its set, check they are idle, snapshot (including NVLink counters), run, snapshot again,
parse the RESULT lines (see patch_custom_mc.py) and the NCCL logs. GPU locks, idle guard and snapshots are the
ones of run_allreduce.py. Output keys of the summary are read by the analysis scripts: do not rename.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from run_allreduce import NCCL_LIBRARY, OCCUPIED_EXIT, SYM_KERNEL_LINE, OccupiedError, ensure_idle, lock_devices, snapshot

BINARY = Path(__file__).resolve().parents[1] / 'bin/custom_mc/allreduce'
RUN_TIMEOUT_SECONDS = 1200
RESULT_LINE = re.compile(r'^RESULT (.*)$', re.MULTILINE)
INTEGER_KEYS = ('ranks', 'threads', 'bytes', 'iters', 'wrong')
FLOAT_KEYS = ('lat_us', 'algbw_gbps')
FORCED_NCCL_KERNEL = 'AllReduce_RSxLDMC_AGxSTMC'


def build_command(cell, gpu_count, environment):
    """mpirun with every variable the program or NCCL reads passed through with -x."""
    forwarded = [name for name in environment if name.startswith(('NCCL_', 'CUSTOM_MC_')) or name in ('CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH')]
    command = ['mpirun', '--allow-run-as-root', '-np', str(gpu_count), '--bind-to', 'none']
    for name in sorted(forwarded):
        command += ['-x', name]
    command += [str(BINARY), '-c', str(cell['min_ctas']), '-C', str(cell['max_ctas']),
                '-m', str(cell['min_size']), '-M', str(cell['max_size'])]
    return command


def build_environment(cell, devices, gpu_count, nccl_log_pattern):
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': devices, 'LD_LIBRARY_PATH': str(NCCL_LIBRARY),
                   'NCCL_DEBUG': 'INFO', 'NCCL_DEBUG_SUBSYS': 'INIT,TUNING', 'NCCL_DEBUG_FILE': nccl_log_pattern}
    # NCCL's library multicast kernel needs more than 2 GPUs; at 2 GPUs NCCL picks its own kernel and that column is not a reference.
    if gpu_count > 2 and cell.get('force_nccl_kernel', True):
        environment['NCCL_SYM_KERNEL'] = f'allreduce:{FORCED_NCCL_KERNEL}'
    if cell.get('kernels'):
        environment['CUSTOM_MC_KERNELS'] = cell['kernels']
    if cell.get('timed_iters'):
        environment['CUSTOM_MC_TIMED_ITERS'] = str(cell['timed_iters'])
    environment.update(cell.get('extra_env', {}))
    return environment


def parse_results(stdout):
    rows = []
    for match in RESULT_LINE.finditer(stdout):
        fields = dict(item.split('=', 1) for item in match[1].split())
        for key in INTEGER_KEYS:
            fields[key] = int(fields[key])
        for key in FLOAT_KEYS:
            fields[key] = float(fields[key])
        rows.append(fields)
    return rows


def parse_nccl_logs(raw_dir, stem_name):
    """Kernel names and (CTAs, threads/CTA) that NCCL logged for the nccl column, over all ranks."""
    kernels = set()
    shapes = set()
    for log_path in raw_dir.glob(f'{stem_name}.nccl.*.log'):
        for match in SYM_KERNEL_LINE.finditer(log_path.read_text(errors='replace')):
            kernels.add(match[1])
            shapes.add(f'{match[2]}x{match[3]}')
    return sorted(kernels), sorted(shapes)


def run_one(cell, output_stem):
    """One guarded invocation; returns its summary record."""
    devices = cell['devices']
    gpu_count = len(devices.split(','))
    nccl_log_pattern = str(output_stem.parent / f'{output_stem.name}.nccl.%p.log')
    environment = build_environment(cell, devices, gpu_count, nccl_log_pattern)
    command = build_command(cell, gpu_count, environment)
    locks = lock_devices(devices)
    try:
        ensure_idle(devices)
        before = snapshot(devices)
        start = time.time()
        result = subprocess.run(command, text=True, capture_output=True, env=environment, timeout=RUN_TIMEOUT_SECONDS)
        elapsed = time.time() - start
        after = snapshot(devices)
    finally:
        for handle in locks:
            handle.close()
    output_stem.with_suffix('.stdout.log').write_text(result.stdout + result.stderr)
    output_stem.with_suffix('.snapshots.json').write_text(json.dumps({'before': before, 'after': after, 'command': command,
        'env': {key: value for key, value in environment.items() if key.startswith(('NCCL_', 'CUSTOM_MC_')) or key in ('CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH')}}))
    rows = parse_results(result.stdout)
    nccl_kernels, nccl_shapes = parse_nccl_logs(output_stem.parent, output_stem.name)
    # Upstream prints PASS on stdout and FAIL details on stderr.
    correctness_pass = result.stdout.count('] PASS')
    correctness_fail = (result.stdout + result.stderr).count('FAIL')
    nccl_measured = any(row['kernel'] == 'nccl' for row in rows)
    nccl_forced = 'NCCL_SYM_KERNEL' in environment
    selection_ok = (not nccl_measured) or (not nccl_forced) or nccl_kernels == [FORCED_NCCL_KERNEL]
    wrong_total = sum(row['wrong'] for row in rows)
    record = {'run': output_stem.name, 'cell_id': cell['id'], 'gpu_count': gpu_count, 'devices': devices,
              'min_ctas': cell['min_ctas'], 'max_ctas': cell['max_ctas'], 'kernels': cell.get('kernels', 'all'),
              'timed_iters': cell.get('timed_iters'), 'return_code': result.returncode, 'wall_seconds': round(elapsed, 1),
              'utc_start': before['utc'], 'rows': rows, 'builtin_correctness_pass_lines': correctness_pass,
              'builtin_correctness_fail_lines': correctness_fail, 'nccl_forced_kernel': FORCED_NCCL_KERNEL if nccl_forced else None,
              'nccl_kernels': nccl_kernels, 'nccl_ctas_x_threads': nccl_shapes, 'selection_ok': selection_ok, 'wrong_total': wrong_total}
    record['valid'] = result.returncode == 0 and bool(rows) and wrong_total == 0 and correctness_fail == 0 and selection_ok
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=Path, help='JSON: {"rounds": n, "cells": [{id, devices, min_ctas, max_ctas, min_size, max_size, kernels?, timed_iters?, force_nccl_kernel?, extra_env?}]}')
    parser.add_argument('raw', type=Path, help='directory for raw records')
    parser.add_argument('summary', type=Path, help='JSONL summary, appended')
    arguments = parser.parse_args()
    session = json.loads(arguments.session.read_text())
    arguments.raw.mkdir(parents=True, exist_ok=True)
    cells = session['cells']
    done = set()
    if arguments.summary.exists():
        done = {json.loads(line)['run'] for line in arguments.summary.read_text().splitlines() if line.strip()}
    for round_index in range(session['rounds']):
        shift = round_index % len(cells)
        order = cells[shift:] + cells[:shift]
        for cell in order:
            stem = arguments.raw / f"{cell['id']}_r{round_index + 1}"
            if stem.name in done:
                continue
            try:
                record = run_one(cell, stem)
            except (OccupiedError, BlockingIOError) as error:
                print(f'STOP occupied: {error}', flush=True)
                sys.exit(OCCUPIED_EXIT)
            record['round'] = round_index + 1
            with arguments.summary.open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            largest_bytes = max((row['bytes'] for row in record['rows']), default=0)
            largest = {f"{row['kernel']}@{row['ctas']}": row['algbw_gbps'] for row in record['rows'] if row['bytes'] * 2 > largest_bytes}
            print(f"{stem.name}: valid={record['valid']} rc={record['return_code']} wrong={record['wrong_total']} "
                  f"nccl={record['nccl_kernels']} {record['nccl_ctas_x_threads']} {record['wall_seconds']}s top={largest}", flush=True)


if __name__ == '__main__':
    main()
