#!/usr/bin/env python3
"""Source-side read pressure: calibrate reads and communication, then pair them, sequentially.

Run one phase per invocation, in this order:
  validate       new binary versus the preserved previous binary at recorded settings
  reads          standalone source reads, every M × Tm, five rotated rounds
  unroll         independent loads per read lane at selected M × Tm points
  communication  copy-only C × Tc per source-SM budget group, five rotated rounds
  controls       standalone reads with each group's exact C × Tc control reservation
  contention     adjacent copy-only / copy-with-reads pairs per group and read setting
  mutual         duration-matched reads-only / reads-with-copy pairs
  probes         minimal/detailed probe pairs and closing references
Phases after `unroll` read their calibrated choices from selection.json, written after
inspecting the preceding calibration; the runner never chooses pressure levels itself.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from collect import summarize

COMMON = Path(__file__).resolve().parents[1]
EXPERIMENT = COMMON.parent / 'observation/shared_resources'
RESULTS = EXPERIMENT / 'results/2026-10-27-source-memory-gpu6-7'
CONFIGS = EXPERIMENT / 'configs/source_memory_gpu6_7'
SELECTION = RESULTS / 'selection.json'
PREVIOUS_BINARY = COMMON / 'bin/push_harness_before_source_memory'
SOURCE_SMS = 132
# One physical block size for every role and run, so changing a role's active threads never
# changes another role's physical CTA shape. Inactive lanes join barriers but do no work.
BLOCK_THREADS = 1024
READ_CTAS = [1, 2, 4, 8, 16, 32, 64, 96, 100, 128, 131]
READ_THREADS = [128, 256, 512, 768, 1024]
COMMUNICATION_CTAS = [1, 32, 64, 128]
COMMUNICATION_THREADS = [256, 512, 768, 1024]
READ_UNROLLS = [1, 2, 4, 8, 16]
ROUNDS = 5


def base_values():
    return dict(source_gpu=6, destination_gpu=7, ctas=128, threads=512, memory_threads=512,
                block_threads=BLOCK_THREADS, read_unroll=4, co_ctas=0, kind=0, vector_reads=1,
                detailed=1, communication=1, global_stride=0, pacing=0,
                warmup=3, iterations=12, payload_mib=2048, copy_batch=8)


def run(label, binary=None, **overrides):
    """Write or confirm one config, run it once under the guard, and validate the record."""
    values = base_values()
    values.update(overrides)
    if binary is not None:
        # The previous binary rejects the role-thread keys; it only runs equal-thread configs.
        for key in ['memory_threads', 'block_threads', 'read_unroll']:
            values.pop(key)
    config = CONFIGS / f'{label}.cfg'
    content = ''.join(f'{key}={value}\n' for key, value in values.items())
    if config.exists() and config.read_text() != content:
        raise RuntimeError(f'Configuration changed: {config}')
    config.write_text(content)
    output = RESULTS / 'raw' / f'{label}.json'
    if not output.exists():
        command = [sys.executable, str(COMMON/'scripts/run_one.py'), str(config), str(output)]
        if binary is not None:
            command += ['--binary', str(binary)]
        subprocess.run(command, check=True)
    data = json.loads(output.read_text())
    assert data['verified'] and data['source_gpu'] == 6 and data['destination_gpu'] == 7
    assert data['ctas'] == values['ctas'] and data['threads'] == values['threads'] and data['co_ctas'] == values['co_ctas']
    if binary is None:
        assert data['memory_threads'] == values['memory_threads'] and data['block_threads'] == values['block_threads']
        assert data['read_unroll'] == values['read_unroll']
    observations = summarize(output)
    assert len(observations) == values['iterations']
    if values['co_ctas']:
        assert all(row['minimum_overlap_fraction'] == 1 and row['co_work_units'] > 0 for row in observations)
    copy = statistics.median(row['throughput_gbps'] for row in observations)
    reads = statistics.median(row['co_rate'] for row in observations)
    print(f'{label}: copy {copy:.3f} GB/s, reads {reads:.2f} GB/s', flush=True)
    return observations


def rotated(items, repetition):
    offset = (repetition - 1) * max(1, len(items) // ROUNDS) % len(items)
    return items[offset:] + items[:offset]


def read_only(label, memory_ctas, memory_threads, control_ctas=1, control_threads=BLOCK_THREADS, **overrides):
    """Copy disabled: control CTAs only time a window; source reads are the measured workload."""
    values = dict(ctas=control_ctas, threads=control_threads, memory_threads=memory_threads,
                  co_ctas=memory_ctas, kind=2, communication=0, solo_ns=20000000, warmup=2, iterations=5)
    values.update(overrides)
    return run(label, **values)


def pin_provenance():
    provenance = {name: hashlib.sha256((COMMON/name).read_bytes()).hexdigest()
                  for name in ['src/push_harness.cu', 'bin/push_harness']}
    saved = RESULTS / 'build_provenance.json'
    if saved.exists():
        recorded = json.loads(saved.read_text())
        # A rebuild of unchanged source can differ in host bytes only; such a binary is accepted
        # only when listed with evidence of identical device SASS (see equivalent_binaries).
        accepted = [recorded['bin/push_harness']] + [entry['sha256'] for entry in recorded.get('equivalent_binaries', [])]
        assert provenance['src/push_harness.cu'] == recorded['src/push_harness.cu'], 'Source changed since this campaign started'
        assert provenance['bin/push_harness'] in accepted, 'Program changed since this campaign started'
        return
    saved.write_text(json.dumps(provenance, indent=2) + '\n')


def validate():
    # Interleave preserved and new binaries at recorded settings. Equal-thread configs only.
    for repetition in range(1, 4):
        run(f'validate_copy_previous_R{repetition}', binary=PREVIOUS_BINARY)
        run(f'validate_copy_block512_R{repetition}', block_threads=512)
        run(f'validate_copy_block1024_R{repetition}')
        reads = dict(ctas=128, co_ctas=131, kind=3, communication=0, solo_ns=20000000, warmup=2, iterations=5)
        run(f'validate_destination_reads_previous_R{repetition}', binary=PREVIOUS_BINARY, **reads)
        run(f'validate_destination_reads_block512_R{repetition}', block_threads=512, **reads)


def reads():
    grid = [(memory_ctas, memory_threads) for memory_threads in READ_THREADS for memory_ctas in READ_CTAS]
    for repetition in range(1, ROUNDS + 1):
        for memory_ctas, memory_threads in rotated(grid, repetition):
            read_only(f'reads_m{memory_ctas}_t{memory_threads}_R{repetition}', memory_ctas, memory_threads)


def unroll(selection):
    points = [tuple(point) for point in selection['unroll_points']]
    grid = [(memory_ctas, memory_threads, depth) for memory_ctas, memory_threads in points for depth in READ_UNROLLS]
    for repetition in range(1, ROUNDS + 1):
        for memory_ctas, memory_threads, depth in rotated(grid, repetition):
            read_only(f'unroll_m{memory_ctas}_t{memory_threads}_u{depth}_R{repetition}', memory_ctas, memory_threads, read_unroll=depth)


def communication(selection):
    grid = [(ctas, threads) for ctas in COMMUNICATION_CTAS for threads in COMMUNICATION_THREADS]
    for repetition in range(1, ROUNDS + 1):
        for ctas, threads in rotated(grid, repetition):
            run(f'communication_c{ctas}_t{threads}_R{repetition}', ctas=ctas, threads=threads,
                memory_threads=selection['read_threads'], read_unroll=selection['read_unroll'])


def group_read_levels(ctas):
    return [memory for memory in READ_CTAS if ctas + memory <= SOURCE_SMS] + \
           ([SOURCE_SMS - ctas] if SOURCE_SMS - ctas not in READ_CTAS else [])


def controls(selection):
    for group in selection['groups']:
        ctas, threads = group['ctas'], group['threads']
        levels = group_read_levels(ctas)
        for repetition in range(1, ROUNDS + 1):
            for memory in rotated(levels, repetition):
                read_only(f'controls_c{ctas}_t{threads}_m{memory}_R{repetition}', memory, selection['read_threads'],
                          control_ctas=ctas, control_threads=threads, read_unroll=selection['read_unroll'])


def contention(selection):
    for group in selection['groups']:
        ctas, threads = group['ctas'], group['threads']
        common = dict(ctas=ctas, threads=threads, memory_threads=selection['read_threads'], read_unroll=selection['read_unroll'])
        levels = group_read_levels(ctas)
        run(f'contention_c{ctas}_t{threads}_opening', **common)
        for repetition in range(1, ROUNDS + 1):
            for memory in rotated(levels, repetition):
                run(f'contention_c{ctas}_t{threads}_m{memory}_H{repetition}', **common)
                run(f'contention_c{ctas}_t{threads}_m{memory}_P{repetition}', co_ctas=memory, kind=2, **common)
        run(f'contention_c{ctas}_t{threads}_closing', **common)


def mutual(selection):
    # Highest read setting in each group plus any setting chosen after inspecting contention.
    for group in selection['groups']:
        ctas, threads = group['ctas'], group['threads']
        common = dict(ctas=ctas, threads=threads, memory_threads=selection['read_threads'], read_unroll=selection['read_unroll'])
        for memory in group['mutual_levels']:
            concurrent = [row for repetition in range(1, ROUNDS + 1)
                          for row in summarize(RESULTS/'raw'/f'contention_c{ctas}_t{threads}_m{memory}_P{repetition}.json')]
            solo_ns = round(statistics.median(row['co_active_us'] for row in concurrent) * 1000)
            for repetition in range(1, ROUNDS + 1):
                run(f'mutual_c{ctas}_t{threads}_m{memory}_H{repetition}', co_ctas=memory, kind=2, communication=0, solo_ns=solo_ns, **common)
                run(f'mutual_c{ctas}_t{threads}_m{memory}_P{repetition}', co_ctas=memory, kind=2, **common)


def probes(selection):
    for group in selection['groups']:
        ctas, threads = group['ctas'], group['threads']
        memory = max(group_read_levels(ctas))
        common = dict(ctas=ctas, threads=threads, memory_threads=selection['read_threads'], read_unroll=selection['read_unroll'],
                      co_ctas=memory, kind=2)
        for repetition in range(1, ROUNDS + 1):
            for detailed in (0, 1):
                run(f'probe_c{ctas}_t{threads}_m{memory}_d{detailed}_R{repetition}', detailed=detailed, **common)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('phase', choices=['validate', 'reads', 'unroll', 'communication', 'controls', 'contention', 'mutual', 'probes'])
    arguments = parser.parse_args()
    CONFIGS.mkdir(parents=True, exist_ok=True)
    (RESULTS/'raw').mkdir(parents=True, exist_ok=True)
    pin_provenance()
    if arguments.phase == 'validate':
        validate()
    elif arguments.phase == 'reads':
        reads()
    else:
        selection = json.loads(SELECTION.read_text())
        globals()[arguments.phase](selection)
    subprocess.run([sys.executable, str(COMMON/'scripts/collect.py'), str(RESULTS/'raw')], check=True)
    print(f'PHASE {arguments.phase} COMPLETE', flush=True)


if __name__ == '__main__':
    main()
