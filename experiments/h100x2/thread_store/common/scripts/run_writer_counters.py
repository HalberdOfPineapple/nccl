#!/usr/bin/env python3
"""Check the writer's requested bytes against hardware counters, writer alone, one GPU pair.

Usage: run_writer_counters.py --gpus SOURCE,DESTINATION
Each run is one guarded ncu invocation of the writer alone (copy off, one copy CTA times the window).
Application replay reruns the whole program for each counter pass: kernel replay hung on two-GPU
launches of this harness. Clocks and caches are left alone. Counters are per launch and GPU-wide on
the source GPU; the harness record that remains is the last pass's.
"""
import argparse
import fcntl
import subprocess
from contextlib import ExitStack

from run_one import ensure_idle
from run_store_congest import BINARY, CONFIGS, RESULTS, ROUNDS, TMA_BINARY, rate_levels, rotated, target_gbps

GROUP = 'counter_check'
NCU = '/usr/local/cuda/bin/ncu'
METRICS = ['dram__bytes_write.sum', 'dram__bytes_read.sum', 'dram__throughput.avg.pct_of_peak_sustained_elapsed', 
           'gpu__time_duration.sum',
           'lts__t_sectors.sum', 'lts__t_sectors_op_write.sum', 'lts__t_sectors_op_read.sum', 'lts__t_sectors_srcnode_gpc.sum',
           'lts__t_sectors_srcunit_ltcfabric.sum', 'lts__t_sectors_srcunit_tex.sum', 'lts__t_sectors_srcunit_l1.sum',
           'lts__t_sectors_lookup_hit.sum', 'lts__t_sectors_lookup_miss.sum']
E73_WINDOW_NS = 5860000                         # writer active time in E7.3 at no limit


def setting(ctas, threads=1024, **overrides):
    values = dict(copy_ctas=1, copy_threads=512, copy_gpcs=255, copy_placement=0, writer_ctas=ctas, writer_threads=threads, writer_gpcs=255, writer_placement=0,
                  target_gbps=0, writer_idle=0, writer_alternate=0, use_unmapped=0, block_threads=1024, warmup=2, iterations=3, detailed=0, communication=0,
                  payload_mib=2048, solo_ns=20000000)
    values.update(overrides)
    return values


def settings():
    """label: (binary, config values)."""
    grid = {f'w64_t1024_{level}_spread': (BINARY, setting(64, target_gbps=target)) for level, target in rate_levels()}
    grid.update({
        'w16_t1024_unlimited_spread': (BINARY, setting(16)),
        'w32_t1024_unlimited_spread': (BINARY, setting(32)),
        'w64_t512_unlimited_spread': (BINARY, setting(64, 512)),
        'w96_t1024_unlimited_spread': (BINARY, setting(96)),
        'w64_t1024_unlimited_packed': (BINARY, setting(64, writer_placement=1)),
        # The E7.3 layout: writer in the 7 GPCs other than G6, timer CTA in G6, and its window.
        'w64_t1024_unlimited_separate': (BINARY, setting(64, copy_gpcs=1 << 6, copy_placement=1, writer_gpcs=255 & ~(1 << 6))),
        'w64_t1024_unlimited_separate_short': (BINARY, setting(64, copy_gpcs=1 << 6, copy_placement=1, writer_gpcs=255 & ~(1 << 6), solo_ns=E73_WINDOW_NS)),
        'w64_t1024_p60_separate_short': (BINARY, setting(64, copy_gpcs=1 << 6, copy_placement=1, writer_gpcs=255 & ~(1 << 6), target_gbps=target_gbps(60), solo_ns=E73_WINDOW_NS)),
        # E7.6's TMA store writer.
        'tma_w64_k1_unlimited_spread': (TMA_BINARY, setting(64, writer_kind=2, chunks_in_flight=1)),
    })
    return grid


def write_config(label, values):
    config = CONFIGS / GROUP / f'{label}.cfg'
    config.parent.mkdir(parents=True, exist_ok=True)
    content = ''.join(f'{key}={value}\n' for key, value in values.items())
    if config.exists() and config.read_text() != content:
        raise RuntimeError(f'Configuration changed: {config}')
    config.write_text(content)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', required=True)
    arguments = parser.parse_args()
    source_gpu, destination_gpu = arguments.gpus.split(',')
    raw = RESULTS / GROUP / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    grid = settings()
    for repetition in range(1, ROUNDS + 1):
        for label in rotated(list(grid), repetition):
            binary, values = grid[label]
            config = write_config(label, values)
            output = raw / f'{label}_R{repetition}.json'
            if output.exists():
                continue
            # Same per-device locks as run_one.py, held while checking occupancy and profiling.
            with ExitStack() as locks:
                for device in sorted((source_gpu, destination_gpu)):
                    handle = locks.enter_context(open(f'/tmp/nccl_thread_store_gpu_{device}.lock', 'a'))
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                ensure_idle(arguments.gpus)
                command = [NCU, '--replay-mode', 'application', '--clock-control', 'none', '--cache-control', 'none', '--devices', source_gpu, '--metrics', ','.join(METRICS),
                           '--csv', '--log-file', str(output.with_suffix('.ncu.csv')), str(binary), str(config), str(output), '--gpus', arguments.gpus,
                           '--gpc-map', str(CONFIGS / 'gpc_map' / f'gpu{source_gpu}.json')]
                with output.with_suffix('.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
            print(label, repetition, flush=True)


if __name__ == '__main__':
    main()
