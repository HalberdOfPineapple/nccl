#!/usr/bin/env python3
"""Rate-controlled memory pressure on the source or destination GPU, one phase per invocation.

Usage: run_rate_pressure.py {source|destination} PHASE --gpus SOURCE,DESTINATION

Phases, in order:
  validate     new build versus the previous build at the selected reference (destination only)
  validate_final  the same check for the build with in-run alternation, made after calibration
  calibrate    copy alone with idle pressure CTAs: source tries C x Tc with C <= 108;
               destination confirms 128 x 512 against 128 x 1024
  idle_check   copy with no pressure CTAs versus copy with 24 idle pressure CTAs
  alone        reads alone at each target, with the same control CTAs as the copy
  pairs        runs that alternate idle and reading launches, per target (pairs share SM placement)
  mutual       duration-matched reads-only / reads-with-copy pairs (secondary)
  probes       minimal versus detailed probes at the strongest target
  screen95     reads alone, 32/40/48 pressure CTAs, 95% target and no limit (one run each)
  level95      the 95% level chosen in selection.json ('level95'): reads alone, alternating
               runs with the copy, and (destination) duration-matched pairs
Choices made between phases are read from selection.json, written by hand after inspecting
the previous phase. Configs carry no GPU numbers; the pair comes from --gpus and is recorded
in every raw record. One experiment stays on one pair for all its phases.
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
RESULTS = EXPERIMENT / 'results/2026-10-28-rate-pressure'
CONFIGS = EXPERIMENT / 'configs/rate_pressure'
PREVIOUS_BINARY = COMMON / 'bin/push_harness_before_rate_reads'
PEAK_GBPS = 3352.32                   # 2 × 2619 MHz × 5120 bits / 8
TARGET_PERCENTS = [20, 40, 60, 80]
SHARED_BYTES = 196864                 # 3 × 64 KiB chunks + barriers, for every CTA in every run
PRESSURE_CTAS = 24
# 24 pressure CTAs stop at about 91% of peak; the 95% level needs more pressure CTAs.
SCREEN95_CTAS = [32, 40, 48]
ROUNDS = 5
SOURCE_CALIBRATION = [(ctas, threads) for ctas in (64, 96, 108) for threads in (512, 768, 1024)]
DESTINATION_CALIBRATION = [(128, 512), (128, 1024)]


def target_gbps(percent):
    return round(PEAK_GBPS * percent / 100)


class Campaign:
    def __init__(self, side, gpus):
        self.side = side
        self.gpus = gpus
        self.results = RESULTS / side
        self.configs = CONFIGS / side
        self.kind = 4 if side == 'source' else 5

    def run(self, label, binary=None, **overrides):
        """Write or confirm one config, run it once under the guard, validate the record."""
        values = dict(ctas=128, threads=512, memory_threads=1024, block_threads=1024, shared_bytes=SHARED_BYTES,
                      kind=self.kind, co_ctas=PRESSURE_CTAS, chunk_kib=64, chunks_in_flight=3, target_gbps=0, pressure_idle=1,
                      vector_reads=1, detailed=1, communication=1, global_stride=0, pacing=0,
                      warmup=3, iterations=12, payload_mib=2048, copy_batch=8)
        values.update(overrides)
        if values['co_ctas'] == 0:
            values['kind'] = 0
        config = self.configs / f'{label}.cfg'
        content = ''.join(f'{key}={value}\n' for key, value in values.items())
        if config.exists() and config.read_text() != content:
            raise RuntimeError(f'Configuration changed: {config}')
        config.write_text(content)
        output = self.results / 'raw' / f'{label}.json'
        if not output.exists():
            command = [sys.executable, str(COMMON / 'scripts/run_one.py'), str(config), str(output)]
            command += ['--binary', str(binary)] if binary else ['--gpus', self.gpus]
            subprocess.run(command, check=True)
        data = json.loads(output.read_text())
        assert data['verified'] and data['co_ctas'] == values['co_ctas'] and data['ctas'] == values['ctas']
        observations = summarize(output)
        assert len(observations) == values['iterations']
        if values['co_ctas']:
            assert all(row['minimum_overlap_fraction'] == 1 and row['co_work_units'] > 0 for row in observations if not row['pressure_idle'])
        copy = statistics.median(row['throughput_gbps'] for row in observations)
        reads = statistics.median([row['co_rate'] for row in observations if not row['pressure_idle']] or [0])
        print(f'{self.side} {label}: copy {copy:.2f} GB/s, reads {reads:.1f} GB/s', flush=True)
        return observations

    def selection(self):
        return json.loads((self.results / 'selection.json').read_text())

    def copy_geometry(self):
        selected = self.selection()
        return dict(ctas=selected['ctas'], threads=selected['threads'])

    def reads_alone_controls(self, geometry):
        # Copy disabled: the source's C control CTAs only time the window.
        return dict(geometry, communication=0, solo_ns=20000000, warmup=2, iterations=5)


def rotated(items, repetition):
    offset = (repetition - 1) * max(1, len(items) // ROUNDS) % len(items)
    return items[offset:] + items[:offset]


def phase_validate(campaign, prefix='validate'):
    # Previous build cannot use the new keys or --gpus; both builds run default shared memory here.
    plain = dict(shared_bytes=0, co_ctas=0, chunk_kib=64, chunks_in_flight=3, target_gbps=0, pressure_idle=0)
    for repetition in range(1, 4):
        legacy = dict(source_gpu=int(campaign.gpus.split(',')[0]), destination_gpu=int(campaign.gpus.split(',')[1]))
        path = campaign.configs / f'{prefix}_previous_R{repetition}.cfg'
        content = ''.join(f'{key}={value}\n' for key, value in dict(legacy, ctas=128, threads=512, kind=0, co_ctas=0, vector_reads=1, detailed=1,
                                                                     communication=1, global_stride=0, pacing=0, warmup=3, iterations=12,
                                                                     payload_mib=2048, copy_batch=8).items())
        path.write_text(content)
        output = campaign.results / 'raw' / f'{prefix}_previous_R{repetition}.json'
        if not output.exists():
            subprocess.run([sys.executable, str(COMMON / 'scripts/run_one.py'), str(path), str(output), '--binary', str(PREVIOUS_BINARY)], check=True)
        print(prefix, 'previous', repetition, round(statistics.median(row['throughput_gbps'] for row in summarize(output)), 2), flush=True)
        campaign.run(f'{prefix}_new_R{repetition}', **plain)


def phase_validate_final(campaign):
    # The alternation build changed host code and the config layout after calibration.
    phase_validate(campaign, 'validate_final')


def phase_calibrate(campaign):
    grid = SOURCE_CALIBRATION if campaign.side == 'source' else DESTINATION_CALIBRATION
    for repetition in range(1, ROUNDS + 1):
        for ctas, threads in rotated(grid, repetition):
            campaign.run(f'calibrate_c{ctas}_t{threads}_R{repetition}', ctas=ctas, threads=threads)


def phase_idle_check(campaign):
    geometry = campaign.copy_geometry()
    for repetition in range(1, ROUNDS + 1):
        campaign.run(f'idle_check_none_R{repetition}', co_ctas=0, **geometry)
        campaign.run(f'idle_check_idle_R{repetition}', **geometry)


def phase_alone(campaign):
    controls = campaign.reads_alone_controls(campaign.copy_geometry())
    for repetition in range(1, ROUNDS + 1):
        for percent in rotated(TARGET_PERCENTS, repetition):
            campaign.run(f'alone_p{percent}_R{repetition}', pressure_idle=0, target_gbps=target_gbps(percent), **controls)


def phase_pairs(campaign):
    # Each run alternates idle and reading launches (6 + 6 retained), so every pair shares the
    # run's SM placement and buffers. Odd rounds start idle, even rounds start reading.
    # Four warm-ups keep the retained launches in the same order as the warm-ups.
    geometry = campaign.copy_geometry()
    campaign.run('pairs_opening_reference', **geometry)
    for repetition in range(1, ROUNDS + 1):
        for percent in rotated(TARGET_PERCENTS, repetition):
            campaign.run(f'pairs_p{percent}_R{repetition}', pressure_idle=0, pressure_alternate=1 if repetition % 2 else 2,
                         target_gbps=target_gbps(percent), warmup=4, **geometry)
    campaign.run('pairs_closing_reference', **geometry)


def phase_mutual(campaign):
    geometry = campaign.copy_geometry()
    for percent in campaign.selection()['mutual_percents']:
        concurrent = [row for repetition in range(1, ROUNDS + 1) for row in summarize(campaign.results / 'raw' / f'pairs_p{percent}_R{repetition}.json') if not row['pressure_idle']]
        solo_ns = round(statistics.median(row['co_active_us'] for row in concurrent) * 1000)
        for repetition in range(1, ROUNDS + 1):
            campaign.run(f'mutual_p{percent}_H{repetition}', pressure_idle=0, target_gbps=target_gbps(percent), communication=0, solo_ns=solo_ns, **geometry)
            campaign.run(f'mutual_p{percent}_P{repetition}', pressure_idle=0, target_gbps=target_gbps(percent), **geometry)


def phase_screen95(campaign):
    controls = campaign.reads_alone_controls(campaign.copy_geometry())
    for pressure_ctas in SCREEN95_CTAS:
        for target_name, rate in [('t95', target_gbps(95)), ('nolimit', 0)]:
            campaign.run(f'screen95_m{pressure_ctas}_{target_name}', co_ctas=pressure_ctas, pressure_idle=0, target_gbps=rate, **controls)


def phase_level95(campaign):
    # Same method as the 20-80% levels; only the pressure CTA count and the target differ.
    level = campaign.selection()['level95']
    geometry = campaign.copy_geometry()
    pressure = dict(co_ctas=level['pressure_ctas'], pressure_idle=0, target_gbps=level['target_gbps'])
    controls = campaign.reads_alone_controls(geometry)
    for repetition in range(1, ROUNDS + 1):
        campaign.run(f'alone_p95_R{repetition}', **pressure, **controls)
    for repetition in range(1, ROUNDS + 1):
        campaign.run(f'pairs_p95_R{repetition}', pressure_alternate=1 if repetition % 2 else 2, warmup=4, **pressure, **geometry)
    if campaign.side == 'destination':
        concurrent = [row for repetition in range(1, ROUNDS + 1) for row in summarize(campaign.results / 'raw' / f'pairs_p95_R{repetition}.json') if not row['pressure_idle']]
        solo_ns = round(statistics.median(row['co_active_us'] for row in concurrent) * 1000)
        for repetition in range(1, ROUNDS + 1):
            campaign.run(f'mutual_p95_H{repetition}', communication=0, solo_ns=solo_ns, **pressure, **geometry)
            campaign.run(f'mutual_p95_P{repetition}', **pressure, **geometry)


def phase_probes(campaign):
    geometry = campaign.copy_geometry()
    for repetition in range(1, ROUNDS + 1):
        for detailed in (0, 1):
            campaign.run(f'probes_p80_d{detailed}_R{repetition}', pressure_idle=0, target_gbps=target_gbps(80), detailed=detailed, **geometry)


def pin_provenance(campaign):
    provenance = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in [
        ('src/push_harness.cu', COMMON / 'src/push_harness.cu'), ('bin/push_harness', COMMON / 'bin/push_harness'),
        ('hbm_read_pressure.cuh', COMMON.parents[2] / 'h100x1/memory_co_runner/common/include/hbm_read_pressure.cuh')]}
    saved = campaign.results / 'build_provenance.json'
    if saved.exists():
        assert json.loads(saved.read_text()) == provenance, 'Program changed since this campaign started; do not rebuild mid-campaign'
    saved.write_text(json.dumps(provenance, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('side', choices=['source', 'destination'])
    parser.add_argument('phase', choices=['validate', 'calibrate', 'validate_final', 'idle_check', 'alone', 'pairs', 'mutual', 'probes', 'screen95', 'level95'])
    parser.add_argument('--gpus', required=True)
    arguments = parser.parse_args()
    campaign = Campaign(arguments.side, arguments.gpus)
    campaign.configs.mkdir(parents=True, exist_ok=True)
    (campaign.results / 'raw').mkdir(parents=True, exist_ok=True)
    pin_provenance(campaign)
    globals()[f'phase_{arguments.phase}'](campaign)
    subprocess.run([sys.executable, str(COMMON / 'scripts/collect.py'), str(campaign.results / 'raw')], check=True)
    print(f'{arguments.side} PHASE {arguments.phase} COMPLETE', flush=True)


if __name__ == '__main__':
    main()
