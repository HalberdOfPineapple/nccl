#!/usr/bin/env python3
"""Run the three-step destination-pressure plan on physical GPUs 6 -> 7.

Raw records stay under raw/. Configurations and derived summaries remain reviewable.
Resume only records validated by the harness and the Python collector.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from collect import summarize

COMMON = Path(__file__).resolve().parents[1]
EXPERIMENT = COMMON.parent / 'observation/shared_resources'
RESULTS = EXPERIMENT / 'results/2026-10-27-gpu6-7'
CONFIGS = EXPERIMENT / 'configs/destination_followup_gpu6_7'
RAW = RESULTS / 'raw'
GEOMETRIES = [(count, threads) for threads in (256, 512) for count in (16, 32, 64)]
LEVELS = (8, 32, 100, 131)


def run(label, count=32, threads=256, memory=0, **overrides):
    configuration = dict(source_gpu=6, destination_gpu=7, ctas=count, threads=threads,
                         kind=3 if memory else 0, co_ctas=memory, vector_reads=1,
                         pacing=0, detailed=1, global_stride=0, communication=1,
                         warmup=3, iterations=12)
    configuration.update(overrides)
    config = CONFIGS / f'{label}.cfg'
    output = RAW / f'{label}.json'
    content = ''.join(f'{key}={value}\n' for key, value in configuration.items())
    if config.exists() and config.read_text() != content:
        raise RuntimeError(f'Configuration changed: {config}')
    config.write_text(content)
    if not output.exists():
        subprocess.run([sys.executable, str(COMMON/'scripts/run_one.py'), str(config), str(output)], check=True)
    data = json.loads(output.read_text())
    if not data.get('verified') or data['source_gpu'] != 6 or data['destination_gpu'] != 7:
        raise RuntimeError(f'Invalid completed run: {output}')
    observations = summarize(output)
    if len(observations) != configuration['iterations']:
        raise RuntimeError(f'Incomplete run: {output}')
    if memory and any(row['minimum_overlap_fraction'] < 1 for row in observations):
        raise RuntimeError(f'Incomplete pressure overlap: {output}')
    return observations


def main():
    for path in (RESULTS, CONFIGS, RAW):
        path.mkdir(parents=True, exist_ok=True)
    hashes = {str(path.relative_to(COMMON)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (COMMON/'src/push_harness.cu', COMMON/'bin/push_harness')}
    provenance = RESULTS / 'build_provenance.json'
    if provenance.exists() and json.loads(provenance.read_text()) != hashes:
        raise RuntimeError('Program changed since this campaign started')
    provenance.write_text(json.dumps(hashes, indent=2)+'\n')
    # One guarded pair verifies config-based device selection before the campaign.
    run('validation_H')
    run('validation_P', memory=131)
    baseline = [(count, 256, 0) for count in (4, 8, 16, 32, 64)]
    baseline += [(32, threads, 0) for threads in (64, 128, 512)]
    baseline += [(count, 512, 0) for count in (16, 64)]
    baseline += [(32, 256, 1)]
    for repetition in range(1, 6):
        rotated = baseline[repetition-1:] + baseline[:repetition-1]
        for count, threads, traversal in rotated:
            run(f'baseline_c{count}_t{threads}_g{traversal}_R{repetition}', count, threads, global_stride=traversal)
    print('STEP 1 COMPLETE', flush=True)
    read_settings = [(threads, memory) for threads in (256, 512) for memory in LEVELS]
    for repetition in range(1, 6):
        rotated = read_settings[repetition-1:] + read_settings[:repetition-1]
        for threads, memory in rotated:
            run(f'solo_t{threads}_m{memory}_R{repetition}', threads=threads, memory=memory,
                communication=0, solo_ns=20000000, warmup=2, iterations=5)
    print('STEP 2 COMPLETE', flush=True)
    conditions = [(count, threads, memory) for count, threads in GEOMETRIES for memory in LEVELS]
    for repetition in range(1, 6):
        rotated = conditions[repetition-1:] + conditions[:repetition-1]
        for count, threads, memory in rotated:
            prefix = f'mixed_c{count}_t{threads}_m{memory}'
            run(f'{prefix}_H{repetition}', count, threads)
            run(f'{prefix}_P{repetition}', count, threads, memory)
    print('STEP 3 PRIMARY COMPLETE', flush=True)
    # Match actual active windows, not the earlier 20 ms calibration window.
    import statistics
    followup = []
    for count, threads, memory in conditions:
        healthy = []
        pressure = []
        durations = []
        for repetition in range(1, 6):
            prefix = f'mixed_c{count}_t{threads}_m{memory}'
            healthy.append(statistics.median(row['throughput_gbps'] for row in summarize(RAW/f'{prefix}_H{repetition}.json')))
            observations = summarize(RAW/f'{prefix}_P{repetition}.json')
            pressure.append(statistics.median(row['throughput_gbps'] for row in observations))
            durations.extend(row['co_active_us'] for row in observations)
        # Include all repeatable decreases, not just a chosen percentage threshold.
        if memory == 131 or all(perturbed < reference for reference, perturbed in zip(healthy, pressure)):
            followup.append((count, threads, memory, round(statistics.median(durations)*1000)))
    (RESULTS/'followup_selection_summary.json').write_text(json.dumps(followup, indent=2)+'\n')
    for repetition in range(1, 6):
        for count, threads, memory, duration in followup:
            prefix = f'mutual_c{count}_t{threads}_m{memory}'
            run(f'{prefix}_H{repetition}', count, threads, memory, communication=0, solo_ns=duration)
            run(f'{prefix}_P{repetition}', count, threads, memory)
    for count, threads in ((16,256),(64,512)):
        for repetition in range(1,6):
            for detailed in (0,1):
                run(f'probe_c{count}_t{threads}_d{detailed}_R{repetition}', count, threads, 131, detailed=detailed)
    for count, threads in GEOMETRIES:
        run(f'return_c{count}_t{threads}', count, threads)
    subprocess.run([sys.executable, str(COMMON/'scripts/collect.py'), str(RAW)], check=True)
    (RESULTS/'runs.csv').write_bytes((RAW/'runs.csv').read_bytes())
    print('CAMPAIGN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
