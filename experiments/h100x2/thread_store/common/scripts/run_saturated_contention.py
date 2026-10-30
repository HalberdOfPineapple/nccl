#!/usr/bin/env python3
"""Measure destination reads against the confirmed communication plateau, sequentially."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from collect import summarize

COMMON = Path(__file__).resolve().parents[1]
EXPERIMENT = COMMON.parent / 'observation/shared_resources'
RESULTS = EXPERIMENT / 'results/2026-10-27-saturated-contention-gpu6-7'
CONFIGS = EXPERIMENT / 'configs/saturated_contention_gpu6_7'
LEVELS = [8, 32, 100, 131]


def run(label, memory=0, **overrides):
    values = dict(source_gpu=6, destination_gpu=7, ctas=128, threads=512,
                  co_ctas=memory, kind=3 if memory else 0, vector_reads=1,
                  detailed=1, communication=1, global_stride=0, pacing=0,
                  warmup=3, iterations=12, payload_mib=2048, copy_batch=8)
    values.update(overrides)
    config = CONFIGS / f'{label}.cfg'
    content = ''.join(f'{key}={value}\n' for key, value in values.items())
    if config.exists() and config.read_text() != content:
        raise RuntimeError(f'Configuration changed: {config}')
    config.write_text(content)
    output = RESULTS / 'raw' / f'{label}.json'
    if not output.exists():
        subprocess.run([sys.executable, str(COMMON/'scripts/run_one.py'), str(config), str(output)], check=True)
    data = json.loads(output.read_text())
    assert data['verified'] and data['source_gpu'] == 6 and data['destination_gpu'] == 7
    assert data['bytes'] == 2048*1024*1024 and data['copy_batch'] == 8
    assert data['ctas'] == 128 and data['threads'] == 512 and data['co_ctas'] == memory
    observations = summarize(output)
    assert len(observations) == values['iterations']
    if memory:
        assert all(row['minimum_overlap_fraction'] == 1 and row['co_work_units'] > 0 for row in observations)
    print(label, 'copy GB/s', round(statistics.median(row['throughput_gbps'] for row in observations), 3),
          'reads GB/s', round(statistics.median(row['co_rate'] for row in observations), 3), flush=True)
    return observations


def main():
    CONFIGS.mkdir(parents=True, exist_ok=True)
    (RESULTS/'raw').mkdir(parents=True, exist_ok=True)
    provenance = {name: hashlib.sha256((COMMON/name).read_bytes()).hexdigest()
                  for name in ['src/push_harness.cu', 'bin/push_harness']}
    calibration = EXPERIMENT/'results/2026-10-27-saturation-gpu6-7/batch/build_provenance.json'
    assert provenance == json.loads(calibration.read_text()), 'Recalibrate if the kernel changes'
    saved_provenance = RESULTS/'build_provenance.json'
    if saved_provenance.exists():
        assert json.loads(saved_provenance.read_text()) == provenance, 'Program changed since campaign started'
    saved_provenance.write_text(json.dumps(provenance, indent=2)+'\n')
    run('opening_reference')
    # First characterize achieved pressure with the exact destination thread count.
    for repetition in range(1, 6):
        offset = (repetition-1) % len(LEVELS)
        for memory in LEVELS[offset:] + LEVELS[:offset]:
            run(f'calibration_m{memory}_R{repetition}', memory, communication=0, solo_ns=20000000, warmup=2, iterations=5)
    print('READ CALIBRATION COMPLETE', flush=True)
    for repetition in range(1, 6):
        offset = (repetition-1) % len(LEVELS)
        for memory in LEVELS[offset:] + LEVELS[:offset]:
            run(f'primary_m{memory}_H{repetition}')
            run(f'primary_m{memory}_P{repetition}', memory)
    print('PRIMARY PAIRS COMPLETE', flush=True)
    durations = {}
    for memory in LEVELS:
        observations = [row for repetition in range(1, 6)
                        for row in summarize(RESULTS/'raw'/f'primary_m{memory}_P{repetition}.json')]
        durations[memory] = round(statistics.median(row['co_active_us'] for row in observations)*1000)
    (RESULTS/'duration_selection_summary.json').write_text(json.dumps(durations, indent=2)+'\n')
    # Separate pairs for the secondary question; no copy throughput assigned to solo reads.
    for repetition in range(1, 6):
        offset = (repetition-1) % len(LEVELS)
        for memory in LEVELS[offset:] + LEVELS[:offset]:
            run(f'mutual_m{memory}_H{repetition}', memory, communication=0, solo_ns=durations[memory])
            run(f'mutual_m{memory}_P{repetition}', memory)
    for repetition in range(1, 6):
        for detailed in (0, 1):
            run(f'probe_d{detailed}_R{repetition}', 131, detailed=detailed)
    run('closing_reference')
    subprocess.run([sys.executable, str(COMMON/'scripts/collect.py'), str(RESULTS/'raw')], check=True)
    (RESULTS/'runs.csv').write_bytes((RESULTS/'raw/runs.csv').read_bytes())
    print('CAMPAIGN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
