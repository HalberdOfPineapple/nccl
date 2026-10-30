#!/usr/bin/env python3
"""HBM read-rate calibration on one GPU, one stage per invocation, run sequentially.

  chunks  bytes in flight per CTA (chunk size × chunks in flight) at 8 pressure CTAs
  ctas    number of pressure CTAs for the per-CTA settings chosen from `chunks`
  rate    target read rate (the knob) at the CTA counts chosen from `ctas`

Stages `ctas` and `rate` read their settings from selection.json, written by hand after
looking at the previous stage. Each setting runs once per round; five rounds in rotated order.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

COMMON = Path(__file__).resolve().parents[1]
EXPERIMENT = COMMON.parent / 'calibration/hbm_read_rate'
RESULTS = EXPERIMENT / 'results/2026-10-28'
CONFIGS = EXPERIMENT / 'configs'
SELECTION = RESULTS / 'selection.json'
GPU = 6
PEAK_GBPS = 3352.32                     # 2 × 2619 MHz × 5120 bits / 8, from device attributes
SHARED_LIMIT_KIB = 224                  # one CTA may use 227 KiB; leave room for barriers
ROUNDS = 5
CHUNK_SIZES_KIB = [16, 32, 64, 96, 128, 192]
CHUNK_COUNTS = [1, 2, 3, 4, 6]
SCREEN_CTAS = 8
CTA_COUNTS = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128]
TARGET_PERCENTS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]


def run(label, **values):
    config = CONFIGS / f'{label}.cfg'
    content = ''.join(f'{key}={value}\n' for key, value in dict(gpu=GPU, block_threads=1024, warmup=2, windows=5, window_ns=20000000, **values).items())
    if config.exists() and config.read_text() != content:
        raise RuntimeError(f'Configuration changed: {config}')
    config.write_text(content)
    output = RESULTS / 'raw' / f'{label}.json'
    if not output.exists():
        subprocess.run([sys.executable, str(COMMON / 'scripts/run_one.py'), str(config), str(output)], check=True)
    data = json.loads(output.read_text())
    assert data['verified'] and data['gpu'] == GPU and data['pressure_ctas'] == values['pressure_ctas']
    assert data['chunk_kib'] == values['chunk_kib'] and data['chunks_in_flight'] == values['chunks_in_flight']
    rates = [window_rate(data, window) for window in data['windows']]
    print(f'{label}: {sorted(rates)[len(rates) // 2]:.1f} GB/s', flush=True)


def window_rate(data, window):
    """Bytes read by all pressure CTAs over the union of their active intervals, in GB/s."""
    pressure = [record for record in window if record['role'] == 1]
    started = min(record['start_ns'] for record in pressure)
    ended = max(record['end_ns'] for record in pressure)
    return sum(record['chunks_read'] for record in pressure) * data['chunk_kib'] * 1024 / (ended - started)


def rotated(items, repetition):
    offset = (repetition - 1) * max(1, len(items) // ROUNDS) % len(items)
    return items[offset:] + items[:offset]


def chunk_grid():
    return [(size, count) for size in CHUNK_SIZES_KIB for count in CHUNK_COUNTS if size * count <= SHARED_LIMIT_KIB]


def stage_chunks():
    for repetition in range(1, ROUNDS + 1):
        for size, count in rotated(chunk_grid(), repetition):
            run(f'chunks_c{SCREEN_CTAS}_k{size}_n{count}_R{repetition}', pressure_ctas=SCREEN_CTAS, chunk_kib=size, chunks_in_flight=count)


def stage_ctas(selection):
    grid = [(name, ctas) for name in selection['per_cta_settings'] for ctas in CTA_COUNTS]
    for repetition in range(1, ROUNDS + 1):
        for name, ctas in rotated(grid, repetition):
            setting = selection['per_cta_settings'][name]
            run(f'ctas_{name}_c{ctas}_R{repetition}', pressure_ctas=ctas, chunk_kib=setting['chunk_kib'], chunks_in_flight=setting['chunks_in_flight'])


def stage_rate(selection):
    grid = [(name, percent) for name in selection['rate_settings'] for percent in TARGET_PERCENTS + [0]]
    for repetition in range(1, ROUNDS + 1):
        for name, percent in rotated(grid, repetition):
            setting = selection['rate_settings'][name]
            target_gbps = round(PEAK_GBPS * percent / 100)
            run(f'rate_{name}_p{percent}_R{repetition}', pressure_ctas=setting['pressure_ctas'], chunk_kib=setting['chunk_kib'],
                chunks_in_flight=setting['chunks_in_flight'], target_gbps=target_gbps)


def pin_provenance():
    provenance = {name: hashlib.sha256((COMMON / name).read_bytes()).hexdigest()
                  for name in ['src/hbm_read_calibration.cu', 'include/hbm_read_pressure.cuh', 'bin/hbm_read_calibration']}
    saved = RESULTS / 'build_provenance.json'
    if saved.exists():
        assert json.loads(saved.read_text()) == provenance, 'Program changed since this calibration started; do not rebuild mid-campaign'
    saved.write_text(json.dumps(provenance, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['chunks', 'ctas', 'rate'])
    arguments = parser.parse_args()
    CONFIGS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'raw').mkdir(parents=True, exist_ok=True)
    pin_provenance()
    if arguments.stage == 'chunks':
        stage_chunks()
    else:
        selection = json.loads(SELECTION.read_text())
        globals()[f'stage_{arguments.stage}'](selection)
    print(f'STAGE {arguments.stage} COMPLETE', flush=True)


if __name__ == '__main__':
    main()
