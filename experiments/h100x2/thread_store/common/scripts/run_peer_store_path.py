#!/usr/bin/env python3
"""Plan 08: run the peer-store path experiments, one phase of the plan at a time.

Usage: run_peer_store_path.py GROUP [GROUP...] --gpus SOURCE,DESTINATION [--date DATE]
A counter run is one guarded ncu invocation. Application replay reruns the whole program for each
counter pass: kernel replay hung on two-GPU launches of this harness. Clocks and caches are left
alone. Counters are per launch and GPU-wide on the profiled GPU; the harness record that remains is
the last pass's. A timing run is one run_one.py invocation without any profiler.
"""
import argparse
import fcntl
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

from run_one import ensure_idle

COMMON = Path(__file__).resolve().parents[1]
BINARY = COMMON / 'bin/peer_store_path'
EXPERIMENT = COMMON.parent / 'observation/peer_store_l2_path'
CONFIGS = EXPERIMENT / 'configs'
NCU = '/usr/local/cuda/bin/ncu'
METRICS = ['dram__bytes_write.sum', 'dram__bytes_read.sum', 'gpu__time_duration.sum',
           'nvlrx__bytes_packet_request_data_user.sum', 'nvlrx__bytes_packet_request_data_protocol.sum', 'nvlrx__bytes.sum',
           'nvltx__bytes_packet_request_data_user.sum', 'nvltx__bytes_packet_request_data_protocol.sum', 'nvltx__bytes.sum',
           'lts__t_sectors.sum', 'lts__t_sectors_op_write.sum', 'lts__t_sectors_op_read.sum', 'lts__t_sectors_srcnode_gpc.sum',
           'lts__t_sectors_srcunit_ltcfabric.sum', 'lts__t_sectors_srcunit_ltcfabric_op_write.sum',
           'lts__t_sectors_aperture_device_op_write.sum', 'lts__t_sectors_aperture_peer_op_write.sum', 'lts__t_sectors_aperture_peer.sum',
           'lts__t_sectors_op_write_lookup_hit.sum', 'lts__t_sectors_op_write_lookup_miss.sum',
           'lts__t_sectors_op_read_lookup_hit.sum', 'lts__t_sectors_op_read_lookup_miss.sum', 'lts__d_sectors_fill_device.sum']
TOTAL_MIB = 8192
PEER = 1
LOCAL = 2
NO_STORES = 0
SOURCE_NAMES = {PEER: 'peer', LOCAL: 'local', NO_STORES: 'none'}
SHAPE_VECTORS = 1
SHAPE_ONE_WORD_PER_LINE = 2
SHAPE_LINKED_LIST = 3
PREPARE_READ = 1
PREPARE_LOCAL_STORE = 2
CHASE_AFTER_STORES = 1
CHASE_AFTER_EVICT = 2
FOOTPRINTS_MIB = [1, 4, 16, 32, 64, 128, 512, 2048]


def run_values(source, footprint_mib, passes, **overrides):
    values = dict(store_source=source, footprint_mib=footprint_mib, passes=passes, writer_ctas=16, writer_threads=512, warmup=2, iterations=3)
    values.update(overrides)
    return values


def footprint_label(source, footprint_mib, suffix=''):
    return f'{SOURCE_NAMES[source]}_f{footprint_mib}{suffix}'


def validate_settings():
    """E8.0: known traffic, the launch boundary and the eviction passes."""
    grid = {'none_window3ms': run_values(NO_STORES, 1024, 1, no_store_window_ns=3000000)}
    for source in (PEER, LOCAL):
        grid[footprint_label(source, 1024)] = run_values(source, 1024, 1)
        for evict_in_main in (0, 1):
            grid[footprint_label(source, 4, f'_evictinmain{evict_in_main}')] = run_values(source, 4, TOTAL_MIB // 4, evict_in_main_launch=evict_in_main, quiet_after_stores=0)
        for eviction_passes in (1, 2, 4):
            grid[footprint_label(source, 32, f'_evictpasses{eviction_passes}')] = run_values(source, 32, TOTAL_MIB // 32, eviction_passes=eviction_passes)
    return grid


def enter_settings():
    """E8.1: each byte stored once."""
    grid = {}
    for source in (PEER, LOCAL):
        for footprint_mib in (256, 512, 1024, 2048):
            grid[footprint_label(source, footprint_mib)] = run_values(source, footprint_mib, 1)
    for window_ms in (1, 6):
        grid[f'none_window{window_ms}ms'] = run_values(NO_STORES, 256, 1, no_store_window_ns=window_ms * 1000000)
    return grid


def stay_settings():
    """E8.2: repeated stores over one footprint, the same total."""
    grid = {}
    for footprint_mib in FOOTPRINTS_MIB:
        for source in (PEER, LOCAL):
            grid[footprint_label(source, footprint_mib)] = run_values(source, footprint_mib, TOTAL_MIB // footprint_mib)
    return grid


def knee_settings():
    """E8.2d: finer footprints around the knee, and local stores slowed to about the peer rate.

    At 32 and 64 MiB, peer and local stores differ in HBM writes, but they also differ in rate (about
    350 against 970 GB/s). Six local CTAs × 512 threads/CTA store about 360 GB/s; the CTA count is the
    only change, so a remaining peer-local difference does not come from rate.
    """
    grid = {}
    for footprint_mib in (8, 20, 24, 28, 32, 40, 48, 64):
        passes = TOTAL_MIB // footprint_mib
        grid[footprint_label(PEER, footprint_mib)] = run_values(PEER, footprint_mib, passes)
        grid[footprint_label(LOCAL, footprint_mib)] = run_values(LOCAL, footprint_mib, passes)
        grid[footprint_label(LOCAL, footprint_mib, '_slow')] = run_values(LOCAL, footprint_mib, passes, writer_ctas=6)
    return grid


def variant_settings():
    """E8.2b store shapes and E8.2c a fence after every pass."""
    grid = {}
    for footprint_mib in (4, 32, 512):
        for source in (PEER, LOCAL):
            passes = TOTAL_MIB // footprint_mib
            grid[footprint_label(source, footprint_mib, '_vectors')] = run_values(source, footprint_mib, passes, store_shape=SHAPE_VECTORS)
            # One word in each 128-byte line, so 1/32 of the bytes per pass and 256 MiB stored in total.
            grid[footprint_label(source, footprint_mib, '_onewordperline')] = run_values(source, footprint_mib, passes, store_shape=SHAPE_ONE_WORD_PER_LINE)
            grid[footprint_label(source, footprint_mib, '_fence')] = run_values(source, footprint_mib, passes, fence_each_pass=1)
    return grid


def chase_settings():
    """E8.3: how the data reached the destination, then one thread reads the list."""
    grid = {}
    for footprint_mib in (8, 32, 256):
        for source in (PEER, LOCAL):
            common = dict(store_shape=SHAPE_LINKED_LIST, quiet_after_stores=0, iterations=6)
            grid[footprint_label(source, footprint_mib, '_readnow')] = run_values(source, footprint_mib, 1, chase_kind=CHASE_AFTER_STORES, **common)
            grid[footprint_label(source, footprint_mib, '_readevicted')] = run_values(source, footprint_mib, 1, chase_kind=CHASE_AFTER_EVICT, **common)
            grid[footprint_label(source, footprint_mib, '_readwait5ms')] = run_values(source, footprint_mib, 1, chase_kind=CHASE_AFTER_STORES, chase_wait_ns=5000000, **common)
    return grid


def prior_state_settings():
    """E8.4: state of the footprint in the destination's L2 before the stores."""
    grid = {}
    for source in (PEER, LOCAL):
        common = dict(readback=1, quiet_after_stores=0, iterations=6)
        grid[footprint_label(source, 8, '_evicted')] = run_values(source, 8, 1, **common)
        grid[footprint_label(source, 8, '_presentclean')] = run_values(source, 8, 1, prepare_kind=PREPARE_READ, **common)
        grid[footprint_label(source, 8, '_presentdirty')] = run_values(source, 8, 1, prepare_kind=PREPARE_LOCAL_STORE, **common)
    return grid


def resident_settings():
    """E8.5: a resident reader on the destination, against pushes of different footprints."""
    grid = {}
    for resident_ctas in (16,):
        common = dict(resident_ctas=resident_ctas, resident_mib=16, quiet_after_stores=0)
        grid[f'r{resident_ctas}_nopush'] = run_values(NO_STORES, 1, 1, no_store_window_ns=23000000, **common)
        for footprint_mib in (1, 32, 256, 2048):
            grid[f'r{resident_ctas}_push_f{footprint_mib}'] = run_values(PEER, footprint_mib, TOTAL_MIB // footprint_mib, **common)
    return grid


GROUPS = {'validate': validate_settings, 'enter': enter_settings, 'stay': stay_settings, 'knee': knee_settings, 'variants': variant_settings, 'chase': chase_settings, 'prior': prior_state_settings, 'resident': resident_settings}


def write_config(group, label, values):
    config = CONFIGS / group / f'{label}.cfg'
    config.parent.mkdir(parents=True, exist_ok=True)
    content = ''.join(f'{key}={value}\n' for key, value in values.items())
    if config.exists() and config.read_text() != content:
        raise RuntimeError(f'Configuration changed: {config}')
    config.write_text(content)
    return config


def rotated(labels, repetition):
    offset = (repetition - 1) * max(1, len(labels) // 3) % len(labels)
    return labels[offset:] + labels[:offset]


def run_counters(config, output, gpus, profiled_rank):
    """One ncu invocation under the same per-device locks as run_one.py."""
    source_gpu, destination_gpu = gpus.split(',')
    profiled_gpu = destination_gpu if profiled_rank == 1 else source_gpu
    with ExitStack() as locks:
        for device in sorted((source_gpu, destination_gpu)):
            handle = locks.enter_context(open(f'/tmp/nccl_thread_store_gpu_{device}.lock', 'a'))
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_idle(gpus)
        command = [NCU, '--replay-mode', 'application', '--clock-control', 'none', '--cache-control', 'none', '--devices', profiled_gpu, '--metrics', ','.join(METRICS),
                   '--csv', '--log-file', str(output.with_suffix('.ncu.csv')), str(BINARY), str(config), str(output), '--gpus', gpus, '--profiled-rank', str(profiled_rank)]
        with output.with_suffix('.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)


def run_timing(config, output, gpus):
    subprocess.run([sys.executable, str(COMMON / 'scripts/run_one.py'), str(config), str(output), '--binary', str(BINARY), '--gpus', gpus], check=True, timeout=120, stdout=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('groups', nargs='+', choices=list(GROUPS))
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--date', default='2026-10-31')
    parser.add_argument('--rounds', type=int, default=3)
    # counters-destination: ncu on the destination. counters-source: ncu on the source. timing: no profiler.
    parser.add_argument('--mode', choices=['counters-destination', 'counters-source', 'timing'], default='counters-destination')
    parser.add_argument('--only', help='run only labels containing this text')
    arguments = parser.parse_args()
    for group in arguments.groups:
        grid = GROUPS[group]()
        labels = [label for label in grid if not arguments.only or arguments.only in label]
        raw = EXPERIMENT / 'results' / arguments.date / group / arguments.mode / 'raw'
        raw.mkdir(parents=True, exist_ok=True)
        for repetition in range(1, arguments.rounds + 1):
            for label in rotated(labels, repetition):
                values = grid[label]
                # The source GPU does nothing when the stores are local, so there is nothing to profile there.
                if arguments.mode == 'counters-source' and values['store_source'] != PEER:
                    continue
                config = write_config(group, label, values)
                output = raw / f'{label}_R{repetition}.json'
                if output.exists():
                    continue
                if arguments.mode == 'timing':
                    run_timing(config, output, arguments.gpus)
                else:
                    run_counters(config, output, arguments.gpus, 1 if arguments.mode == 'counters-destination' else 0)
                print(group, label, repetition, flush=True)


if __name__ == '__main__':
    main()
