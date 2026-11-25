#!/usr/bin/env python3
"""Runs nccl-tests AllReduce (and, since 2026-11-18, AllGather and ReduceScatter) for the scalability series, one guarded run at a time.

A cell may name a `collective` (allreduce by default, allgather, reducescatter); it picks the nccl-tests binary, the
symmetric kernel of each arm and the NCCL log line to check. AllReduce cells run exactly as before.

A session file (JSON) lists cells and settings; this script runs them in rounds, rotating the arm order
between rounds, and writes one raw record per run plus a summary JSONL line per run.

Each run: lock every GPU of its set, check they are idle, snapshot, run all_reduce_perf with the arm's
environment, snapshot again, parse the result rows and the NCCL log, and check that the log names the
intended algorithm or kernel. Output keys of the summary are read by the analysis scripts: do not rename.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

TESTS = Path('/root/nccl-tests/build')
NCCL_LIBRARY = Path(__file__).resolve().parents[5] / 'build/lib'
RUN_TIMEOUT_SECONDS = 600
IDLE_MEMORY_MIB = 150
OCCUPIED_EXIT = 10

# Environment and buffer registration for each arm (see 00_common_setup.md, "How each arm is selected").
ARMS = {
    'ring': {'env': {'NCCL_ALGO': 'Ring', 'NCCL_PROTO': 'Simple'}, 'register': '0', 'cta_env': ['NCCL_MIN_CTAS', 'NCCL_MAX_CTAS']},
    'nvls': {'env': {'NCCL_ALGO': 'NVLS', 'NCCL_PROTO': 'Simple'}, 'register': '0', 'cta_env': ['NCCL_NVLS_NCHANNELS']},
    'sym_multicast': {'env': {'NCCL_SYM_KERNEL': 'allreduce:AllReduce_RSxLDMC_AGxSTMC'}, 'register': '2', 'cta_env': ['NCCL_SYM_CTAS'], 'kernel': 'AllReduce_RSxLDMC_AGxSTMC'},
    'sym_plain': {'env': {'NCCL_SYM_KERNEL': 'allreduce:AllReduce_RSxLD_AGxST'}, 'register': '2', 'cta_env': ['NCCL_SYM_CTAS'], 'kernel': 'AllReduce_RSxLD_AGxST'},
}

# What changes with the collective: binary, name in NCCL's log, prefix of NCCL_SYM_KERNEL, symmetric kernel per arm.
COLLECTIVES = {
    'allreduce': {'binary': 'all_reduce_perf', 'log_name': 'AllReduce', 'sym_prefix': 'allreduce',
                  'kernels': {'sym_multicast': 'AllReduce_RSxLDMC_AGxSTMC', 'sym_plain': 'AllReduce_RSxLD_AGxST'}},
    'allgather': {'binary': 'all_gather_perf', 'log_name': 'AllGather', 'sym_prefix': 'allgather',
                  'kernels': {'sym_multicast': 'AllGather_STMC', 'sym_plain': 'AllGather_ST'}},
    'reducescatter': {'binary': 'reduce_scatter_perf', 'log_name': 'ReduceScatter', 'sym_prefix': 'reducescatter',
                      'kernels': {'sym_multicast': 'ReduceScatter_LDMC', 'sym_plain': 'ReduceScatter_LD'}},
}


def arm_settings(cell):
    """The ARMS entry of the cell's arm, with the symmetric kernel of the cell's collective."""
    collective = COLLECTIVES[cell.get('collective', 'allreduce')]
    arm = dict(ARMS[cell['arm']])
    if 'kernel' in arm:
        kernel = collective['kernels'][cell['arm']]
        arm['kernel'] = kernel
        arm['env'] = {'NCCL_SYM_KERNEL': f"{collective['sym_prefix']}:{kernel}"}
    return arm


# AllGather rows print 'none' as the reduction operation.
RESULT_ROW = re.compile(r'^\s*(\d+)\s+(\d+)\s+float\s+(?:sum|none)\s+(-?\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s*$', re.MULTILINE)
SYM_KERNEL_LINE = re.compile(r'-> Kernel (\S+) nchannels (\d+) nthreads (\d+)')
ALGO_LINE = re.compile(r'(\w+): \d+ Bytes -> Algo (\S+) proto (\S+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}')


class OccupiedError(Exception):
    pass


def capture(command):
    result = subprocess.run(command, text=True, capture_output=True)
    return result.stdout + result.stderr


def read_state(devices):
    """Rows of (index, used MiB, utilization %) for the given devices."""
    text = capture(['nvidia-smi', '-i', devices, '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    return [[field.strip() for field in line.split(',')] for line in text.strip().splitlines()]


def ensure_idle(devices):
    # Our own context teardown can show for a few seconds, so only lasting use counts as occupied.
    occupied = []
    for attempt in range(40):
        occupied = [row for row in read_state(devices) if int(row[1]) > IDLE_MEMORY_MIB or int(row[2]) > 0]
        if not occupied:
            return
        time.sleep(0.5)
    raise OccupiedError(f'GPUs not idle after 20 s (index, MiB, %): {occupied}')


def snapshot(devices):
    return {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'gpu': capture(['nvidia-smi', '-i', devices, '--query-gpu=index,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active', '--format=csv']),
        'apps': capture(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv']),
        'link': capture(['nvidia-smi', 'nvlink', '-i', devices, '-gt', 'd']),
        # Added 2026-11-18: raw link bytes (payload plus packet headers and request packets), next to the data payload above.
        'link_raw': capture(['nvidia-smi', 'nvlink', '-i', devices, '-gt', 'r']),
    }


def lock_devices(devices):
    handles = []
    for device in devices.split(','):
        handle = open(f'/tmp/nccl_thread_store_gpu_{device}.lock', 'a')
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(handle)
    return handles


def build_command(cell, gpu_count):
    arm = arm_settings(cell)
    binary = COLLECTIVES[cell.get('collective', 'allreduce')]['binary']
    return [str(TESTS / binary), '-t', '1', '-g', str(gpu_count),
            '-b', cell['min_size'], '-e', cell['max_size'], '-f', '2',
            '-n', str(cell.get('iters', 20)), '-w', str(cell.get('warmup', 5)), '-c', '1',
            '-R', arm['register'], '-d', 'float', '-o', 'sum']


def profiler_prefix(cell, devices, output_stem):
    """Nsight Systems wrapper (added 2026-11-19 for RCA plan 01 E1.9), or [] when the cell has no "nsys" key.

    GPU metrics from the metric set file at 10 kHz on the run's devices, plus the CUDA trace, so a parser
    can keep only samples inside the collective's kernels. The report and its SQLite export sit beside the
    run's raw files. Cells without "nsys" run exactly as before.
    """
    settings = cell.get('nsys')
    if not settings:
        return []
    return ['nsys', 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
            f'--gpu-metrics-devices={devices}', f"--gpu-metrics-set=file:{settings['metrics_file']}",
            f"--gpu-metrics-frequency={settings.get('frequency', 10000)}",
            '--export=sqlite', '--force-overwrite=false', '-o', str(output_stem) + '_nsys']


def build_environment(cell, devices, nccl_log):
    arm = arm_settings(cell)
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': devices, 'LD_LIBRARY_PATH': str(NCCL_LIBRARY),
                   'NCCL_DEBUG': 'INFO', 'NCCL_DEBUG_SUBSYS': 'INIT,TUNING,NVLS', 'NCCL_DEBUG_FILE': str(nccl_log)}
    environment.update(arm['env'])
    if cell.get('ctas'):
        for name in arm['cta_env']:
            environment[name] = str(cell['ctas'])
    environment.update(cell.get('extra_env', {}))
    return environment


def parse_rows(stdout):
    rows = []
    for match in RESULT_ROW.finditer(stdout):
        rows.append({'size_bytes': int(match[1]), 'out_of_place_time_us': float(match[4]),
                     'out_of_place_algbw_gbps': float(match[5]), 'out_of_place_busbw_nccltests_gbps': float(match[6]),
                     'out_of_place_wrong': match[7], 'in_place_time_us': float(match[8]),
                     'in_place_algbw_gbps': float(match[9]), 'in_place_busbw_nccltests_gbps': float(match[10]),
                     'in_place_wrong': match[11]})
    return rows


def parse_selection(cell, log_text):
    """What NCCL ran (metric S3), from the log of rank 0 and the others."""
    arm = arm_settings(cell)
    log_name = COLLECTIVES[cell.get('collective', 'allreduce')]['log_name']
    sym_lines = SYM_KERNEL_LINE.findall(log_text)
    # Only lines of the cell's collective (ALGO_LINE captures the collective name first).
    algo_lines = [line[1:] for line in ALGO_LINE.findall(log_text) if line[0] == log_name]
    selection = {
        'sym_kernels': sorted({line[0] for line in sym_lines}),
        'sym_channel_values': sorted({int(line[1]) for line in sym_lines}),
        'sym_thread_values': sorted({int(line[2]) for line in sym_lines}),
        'algos': sorted({f'{line[0]}/{line[1]}' for line in algo_lines}),
        'algo_channel_ranges': sorted({f'{line[2]}..{line[3]}' for line in algo_lines}),
    }
    if 'kernel' in arm:
        selection['selection_ok'] = selection['sym_kernels'] == [arm['kernel']] and not selection['algos']
    else:
        wanted = arm['env']['NCCL_ALGO'] + '/' + arm['env']['NCCL_PROTO']
        selection['selection_ok'] = [name.upper() for name in selection['algos']] == [wanted.upper()] and not selection['sym_kernels']
    return selection


def run_one(cell, devices, output_stem):
    """One guarded invocation; returns its summary record."""
    gpu_count = len(devices.split(','))
    locks = lock_devices(devices)
    try:
        ensure_idle(devices)
        before = snapshot(devices)
        nccl_log = output_stem.with_suffix('.nccl.log')
        command = profiler_prefix(cell, devices, output_stem) + build_command(cell, gpu_count)
        environment = build_environment(cell, devices, nccl_log)
        start = time.time()
        result = subprocess.run(command, text=True, capture_output=True, env=environment, timeout=RUN_TIMEOUT_SECONDS)
        elapsed = time.time() - start
        after = snapshot(devices)
    finally:
        for handle in locks:
            handle.close()
    output_stem.with_suffix('.stdout.log').write_text(result.stdout + result.stderr)
    output_stem.with_suffix('.snapshots.json').write_text(json.dumps({'before': before, 'after': after, 'command': command,
        'env': {key: value for key, value in environment.items() if key.startswith('NCCL_') or key in ('CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH')}}))
    log_text = nccl_log.read_text(errors='replace') if nccl_log.exists() else ''
    rows = parse_rows(result.stdout)
    record = {'run': output_stem.name, 'cell_id': cell['id'], 'arm': cell['arm'], 'collective': cell.get('collective', 'allreduce'), 'gpu_count': gpu_count, 'devices': devices,
              'requested_ctas': cell.get('ctas'), 'return_code': result.returncode, 'wall_seconds': round(elapsed, 1),
              'utc_start': before['utc'], 'rows': rows, **parse_selection(cell, log_text)}
    wrong = [row for row in rows if row['out_of_place_wrong'] != '0' or row['in_place_wrong'] != '0']
    record['valid'] = result.returncode == 0 and bool(rows) and not wrong and record['selection_ok']
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=Path, help='JSON: {"rounds": n, "cells": [{id, arm, devices, min_size, max_size, ctas?, iters?, warmup?, extra_env?}]}')
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
                record = run_one(cell, cell['devices'], stem)
            except (OccupiedError, BlockingIOError) as error:
                print(f'STOP occupied: {error}', flush=True)
                sys.exit(OCCUPIED_EXIT)
            record['round'] = round_index + 1
            with arguments.summary.open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            best = max((row['out_of_place_algbw_gbps'] for row in record['rows']), default=0)
            print(f"{stem.name}: valid={record['valid']} algbw_max={best} sel={record['algos'] or record['sym_kernels']} "
                  f"ch={record['algo_channel_ranges'] or record['sym_channel_values']} thr={record['sym_thread_values']} {record['wall_seconds']}s", flush=True)


if __name__ == '__main__':
    main()
