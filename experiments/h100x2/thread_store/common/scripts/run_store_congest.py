#!/usr/bin/env python3
"""Plan 07 (data-store congestion) runs on one GPU pair, one phase per invocation.

Usage: run_store_congest.py PHASE --gpus SOURCE,DESTINATION

Phases, in order:
  copy_sweep        E7.1: store-only copy alone; CTAs x threads/CTA grid, copy spread over all GPCs
  copy_floor        E7.1 extension: the same at 1-16 CTAs and 128-1024 threads/CTA, because the
                    sweep's smallest setting was already on the plateau
  copy_placement    E7.1: the selected copy settings, spread over all GPCs versus packed into
                    the fewest GPCs
  writer_sweep      E7.2: writer alone, no rate limit; writer CTAs at 1024 threads/CTA and
                    threads/CTA at 32 and 64 writer CTAs; writers spread over all GPCs
  writer_rate       E7.2: the selected writer setting at 20/40/60/80% of HBM peak and no limit
  writer_placement  E7.2: the same levels with the writers packed into the fewest GPCs
Choices between phases are read from selection.json, written by hand after inspecting the
previous phase. Configs carry no GPU numbers; the pair comes from --gpus and is recorded in every
raw record, and the source GPU's SM-to-GPC map (E7.0) is passed with each run.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parents[1]
EXPERIMENT = COMMON.parent / 'observation/data_store_congest'
RESULTS = EXPERIMENT / 'results/2026-10-29'
CONFIGS = EXPERIMENT / 'configs'
BINARY = COMMON / 'bin/store_congest'
PEAK_GBPS = 3352.32                   # 2 × 2619 MHz × 5120 bits / 8
MAPPED_SMS = 124                      # SMs 124-131 are not placed by cluster launches (E7.0)
# 5 rounds until 2026-10-30 (copy_sweep, writer_sweep, part of copy_floor); 3 from then on, at the user's request.
ROUNDS = 3
MAXIMUM_ROUNDS = 5
# The copy_batch knob (1, 2, 4, 8 stores per loop step) was removed on 2026-10-30. Records made
# before then carry _b<N> in their labels and come from the earlier build; they are kept only as
# the evidence for removing the knob, and the copy phases were remeasured with the new build.
LEGACY_BATCHES = [1, 2, 4, 8]
COPY_CTAS = [16, 32, 48, 64, 96, 128]
COPY_THREADS = [256, 512, 768, 1024]
# Added 2026-10-29: copy_sweep was flat from its first point, so it did not show where the plateau starts.
COPY_FLOOR_CTAS = [1, 2, 4, 8, 16]
COPY_FLOOR_THREADS = [128, 256, 512, 1024]
WRITER_CTAS = [4, 8, 16, 24, 32, 48, 64, 96, 128]
WRITER_THREADS = [256, 512, 1024]
TARGET_PERCENTS = [20, 40, 60, 80]


def target_gbps(percent):
    return round(PEAK_GBPS * percent / 100)


def copy_grid():
    """CTA scaling at 512 and 1024 threads/CTA, thread scaling at 64 and 128 CTAs."""
    grid = [(ctas, threads) for threads in (512, 1024) for ctas in COPY_CTAS]
    grid += [(ctas, threads) for ctas in (64, 128) for threads in COPY_THREADS if (ctas, threads) not in grid]
    return grid


def writer_grid():
    grid = [(ctas, 1024) for ctas in WRITER_CTAS]
    grid += [(ctas, threads) for ctas in (32, 64) for threads in WRITER_THREADS if (ctas, threads) not in grid]
    return grid


def rotated(items, repetition):
    offset = (repetition - 1) * max(1, len(items) // ROUNDS) % len(items)
    return items[offset:] + items[:offset]


def summarize(path):
    """One row per retained launch: copy throughput and phases (A1-A6), writer rate (A9), coverage (A8)."""
    data = json.loads(path.read_text())
    rows = []
    idle_list = data['ranks'][0]['writer_idle_by_iteration']
    for index, records in enumerate(data['ranks'][0]['iterations']):
        copy = [record for record in records if record['role'] == 'copy']
        writers = [record for record in records if record['role'] == 'writer']
        row = dict(run=path.stem, iteration=index, writer_idle=idle_list[index])
        if data['communication']:
            started = min(record['start_ns'] for record in copy)
            completed = max(record['end_ns'] for record in copy)
            row.update(completed_us=(completed - started) / 1000, throughput_gbps=data['bytes'] / (completed - started))
            if data['detailed']:
                row.update(issue_us=statistics.median(record['issue_end_ns'] - record['start_ns'] for record in copy) / 1000,
                           ordering_us=statistics.median(record['fence_end_ns'] - record['issue_end_ns'] for record in copy) / 1000,
                           notification_us=statistics.median(record['publish_end_ns'] - record['fence_end_ns'] for record in copy) / 1000,
                           ack_wait_us=statistics.median(record['end_ns'] - record['publish_end_ns'] for record in copy) / 1000,
                           issue_spread_us=(max(record['issue_end_ns'] for record in copy) - min(record['issue_end_ns'] for record in copy)) / 1000)
            window_start, window_end = started, completed
        else:
            timer = [record for record in copy if record['role_index'] == 0][0]
            window_start, window_end = timer['start_ns'], timer['end_ns']
        if writers and not row['writer_idle']:
            writer_start = min(record['start_ns'] for record in writers)
            writer_end = max(record['end_ns'] for record in writers)
            per_cta = [record['work_units'] * data['writer_block_bytes'] / (record['end_ns'] - record['start_ns']) for record in writers]
            # Requested bytes over the union of writer active intervals; includes start and tail blocks.
            row.update(writer_gbps=sum(record['work_units'] for record in writers) * data['writer_block_bytes'] / (writer_end - writer_start),
                       writer_slowest_to_fastest=min(per_cta) / max(per_cta),
                       minimum_overlap_fraction=min(max(0, min(record['end_ns'], window_end) - max(record['start_ns'], window_start)) / (window_end - window_start) for record in writers))
        rows.append(row)
    return data, rows


class Campaign:
    def __init__(self, gpus):
        self.gpus = gpus
        self.source_gpu = int(gpus.split(',')[0])
        self.gpc_map = CONFIGS / 'gpc_map' / f'gpu{self.source_gpu}.json'

    def run(self, group, label, **overrides):
        """Write or confirm one config, run it once under the guard, validate the record."""
        values = dict(copy_ctas=64, copy_threads=512, copy_gpcs=255, copy_placement=0,
                      writer_ctas=0, writer_threads=1024, writer_gpcs=255, writer_placement=0, target_gbps=0,
                      writer_idle=0, writer_alternate=0, use_unmapped=0, block_threads=1024,
                      warmup=3, iterations=12, detailed=1, communication=1, payload_mib=2048, solo_ns=20000000)
        values.update(overrides)
        # Only SMs 124-131 are unmapped; allow them only when the mapped SMs cannot hold the roles.
        values['use_unmapped'] = int(values['copy_ctas'] + values['writer_ctas'] > MAPPED_SMS)
        config = CONFIGS / group / f'{label}.cfg'
        config.parent.mkdir(parents=True, exist_ok=True)
        content = ''.join(f'{key}={value}\n' for key, value in values.items())
        if config.exists() and config.read_text() != content:
            raise RuntimeError(f'Configuration changed: {config}')
        config.write_text(content)
        output = RESULTS / group / 'raw' / f'{label}.json'
        if not output.exists():
            subprocess.run([sys.executable, str(COMMON / 'scripts/run_one.py'), str(config), str(output), '--binary', str(BINARY),
                            '--gpus', self.gpus, '--gpc-map', str(self.gpc_map)], check=True)
        data, rows = summarize(output)
        assert data['verified'] and len(rows) == values['iterations']
        assert data['role_of_sm'].count(1) == values['copy_ctas'] and data['role_of_sm'].count(2) == values['writer_ctas']
        if values['writer_ctas']:
            assert all(row['minimum_overlap_fraction'] == 1 for row in rows if not row['writer_idle'])
        message = f'{group} {label}:'
        if values['communication']:
            message += f" copy {statistics.median(row['throughput_gbps'] for row in rows):.2f} GB/s"
        if values['writer_ctas']:
            message += f" writer {statistics.median(row['writer_gbps'] for row in rows if not row['writer_idle']):.0f} GB/s"
        print(message, flush=True)
        return rows


def copy_label(ctas, threads, placement):
    """Label prefix of a copy setting; the _R<run> suffix is added by the caller."""
    return f'copy_c{ctas}_t{threads}_{placement}'


def legacy_copy_label(ctas, threads, batch, placement):
    """Label prefix of records made while copy_batch existed (until 2026-10-30)."""
    return f'copy_c{ctas}_t{threads}_b{batch}_{placement}'


def selection():
    return json.loads((RESULTS / 'selection.json').read_text())


def writer_alone(ctas, threads, **overrides):
    # Copy off: copy CTA 0 only times five 20 ms windows; every other SM holds an idle CTA.
    return dict(dict(copy_ctas=1, writer_ctas=ctas, writer_threads=threads, communication=0, warmup=2, iterations=5), **overrides)


def phase_copy_sweep(campaign):
    grid = copy_grid()
    for repetition in range(1, ROUNDS + 1):
        for ctas, threads in rotated(grid, repetition):
            campaign.run('copy_alone', f'{copy_label(ctas, threads, "spread")}_R{repetition}', copy_ctas=ctas, copy_threads=threads)


def phase_copy_floor(campaign):
    grid = [(ctas, threads) for ctas in COPY_FLOOR_CTAS for threads in COPY_FLOOR_THREADS]
    for repetition in range(1, ROUNDS + 1):
        for ctas, threads in rotated(grid, repetition):
            campaign.run('copy_alone', f'{copy_label(ctas, threads, "spread")}_R{repetition}', copy_ctas=ctas, copy_threads=threads)


def phase_copy_placement(campaign):
    settings = selection()['copy_placement_settings']
    for repetition in range(1, ROUNDS + 1):
        for setting in rotated(settings, repetition):
            for placement, packed in (('spread', 0), ('packed', 1)):
                label = f"{copy_label(setting['ctas'], setting['threads'], placement)}_R{repetition}"
                campaign.run('copy_alone', label, copy_ctas=setting['ctas'], copy_threads=setting['threads'], copy_placement=packed)


def phase_writer_sweep(campaign):
    grid = writer_grid()
    for repetition in range(1, ROUNDS + 1):
        for ctas, threads in rotated(grid, repetition):
            campaign.run('writer_alone', f'writer_w{ctas}_t{threads}_unlimited_spread_R{repetition}', **writer_alone(ctas, threads))


def rate_levels():
    return [(f'p{percent}', target_gbps(percent)) for percent in TARGET_PERCENTS] + [('unlimited', 0)]


def run_writer_levels(campaign, placement, packed):
    chosen = selection()['writer']
    for repetition in range(1, ROUNDS + 1):
        for level, target in rotated(rate_levels(), repetition):
            label = f"writer_w{chosen['ctas']}_t{chosen['threads']}_{level}_{placement}_R{repetition}"
            campaign.run('writer_alone', label, **writer_alone(chosen['ctas'], chosen['threads'], target_gbps=target, writer_placement=packed))


def phase_writer_rate(campaign):
    run_writer_levels(campaign, 'spread', 0)


def phase_writer_placement(campaign):
    run_writer_levels(campaign, 'packed', 1)


PHASES = dict(copy_sweep=phase_copy_sweep, copy_floor=phase_copy_floor, copy_placement=phase_copy_placement, writer_sweep=phase_writer_sweep,
              writer_rate=phase_writer_rate, writer_placement=phase_writer_placement)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=PHASES)
    parser.add_argument('--gpus', required=True)
    arguments = parser.parse_args()
    PHASES[arguments.phase](Campaign(arguments.gpus))


if __name__ == '__main__':
    main()
