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
  contention        E7.3, E7.4, E7.4b: copy and writer together, three placements interleaved
                    by round at each write level; each run alternates idle and writing transfers.
                    Also idle-writer references, writer-alone runs on the same SMs with a matched
                    window, and a probe-cost check
  read_contrast     E7.5: TMA bulk reads instead of writes, separate GPCs, read targets matched to
                    the write rates achieved in E7.3; thread-store writer runs repeated in each round
  tma_writer_sweep  E7.6 calibration: TMA bulk-store writer alone, CTAs x chunks in flight
  tma_writer_rate   E7.6 calibration: the selected TMA writer at 20/40/60/80% and no limit
  tma_store_contrast E7.6: TMA bulk-store writer next to the copy in separate GPCs, interleaved
                    with the thread-store writer at the same targets
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
TMA_BINARY = COMMON / 'bin/store_congest_tma'   # same harness plus TMA co-runners (writer_kind 1, 2)
TMA_READ_SHARED_BYTES = 196864                # 3 × 64 KiB chunks + barriers, as plan 04's read co-runner
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

    def run(self, group, label, binary=BINARY, **overrides):
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
            subprocess.run([sys.executable, str(COMMON / 'scripts/run_one.py'), str(config), str(output), '--binary', str(binary),
                            '--gpus', self.gpus, '--gpc-map', str(self.gpc_map)], check=True)
        data, rows = summarize(output)
        assert data['verified'] and len(rows) == values['iterations']
        assert data['role_of_sm'].count(1) == values['copy_ctas'] and data['role_of_sm'].count(2) == values['writer_ctas']
        if values['writer_ctas']:
            assert all(row['minimum_overlap_fraction'] == 1 for row in rows if not row['writer_idle'])
        message = f'{group} {label}:'
        if values['communication']:
            message += f" copy {statistics.median(row['throughput_gbps'] for row in rows):.2f} GB/s"
        writing = [row['writer_gbps'] for row in rows if not row['writer_idle']]
        if values['writer_ctas'] and writing:
            message += f" writer {statistics.median(writing):.0f} GB/s"
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


# E7.3/E7.4 placements of the selected copy (16 CTAs) and writer (64 CTAs). GPC masks: bit g = GPC g.
CONTENTION_PLACEMENTS = {
    # E7.3: copy packed into G6; writer spread over the other seven GPCs.
    'separate': dict(copy_gpcs=1 << 6, copy_placement=1, writer_gpcs=255 & ~(1 << 6), writer_placement=0),
    # E7.4: both spread over all eight GPCs, so every GPC hosts copy and writer CTAs (about 2 : 8).
    'shared': dict(copy_gpcs=255, copy_placement=0, writer_gpcs=255, writer_placement=0),
    # E7.4b: copy spread; writer packed, so G6, G7, G1, G2 are full of writers next to 2 copy CTAs each.
    'crowded': dict(copy_gpcs=255, copy_placement=0, writer_gpcs=255, writer_placement=1),
}


def contention_levels():
    return rate_levels()


def mixed(placement, target, round_number, **overrides):
    """One run of 6 idle + 6 writing transfers; odd rounds start idle, even rounds start writing."""
    chosen = selection()
    values = dict(copy_ctas=chosen['copy']['ctas'], copy_threads=chosen['copy']['threads'], writer_ctas=chosen['writer']['ctas'],
                  writer_threads=chosen['writer']['threads'], target_gbps=target, writer_alternate=1 if round_number % 2 else 2)
    values.update(CONTENTION_PLACEMENTS[placement])
    values.update(overrides)
    return values


def writer_active_ns(rows_path):
    """Median writer active interval over the writing transfers of one mixed run."""
    data = json.loads(rows_path.read_text())
    spans = []
    for idle, records in zip(data['ranks'][0]['writer_idle_by_iteration'], data['ranks'][0]['iterations']):
        writers = [record for record in records if record['role'] == 'writer']
        if not idle:
            spans.append(max(record['end_ns'] for record in writers) - min(record['start_ns'] for record in writers))
    return int(statistics.median(spans))


def phase_contention(campaign):
    group = 'contention'
    for placement in CONTENTION_PLACEMENTS:
        campaign.run(group, f'reference_{placement}_open', **mixed(placement, 0, 1, writer_alternate=0, writer_idle=1))
    for repetition in range(1, ROUNDS + 1):
        for level, target in rotated(contention_levels(), repetition):
            for placement in rotated(list(CONTENTION_PLACEMENTS), repetition):
                label = f'mixed_{placement}_{level}_R{repetition}'
                campaign.run(group, label, **mixed(placement, target, repetition))
                # Writer alone on the same SMs (copy CTAs present but idle; copy CTA 0 times the
                # window), window matched to this run's measured writer active time.
                window = writer_active_ns(RESULTS / group / 'raw' / f'{label}.json')
                campaign.run(group, f'writer_{placement}_{level}_R{repetition}', **mixed(placement, target, repetition, writer_alternate=0, communication=0,
                                                                                         solo_ns=window, warmup=2, iterations=6))
    for placement in CONTENTION_PLACEMENTS:
        campaign.run(group, f'reference_{placement}_close', **mixed(placement, 0, 1, writer_alternate=0, writer_idle=1))
    # Probe cost: the same mixed runs with the per-CTA phase timers and checkpoints off.
    for repetition in range(1, ROUNDS + 1):
        for detailed in (1, 0):
            campaign.run(group, f'probe_separate_unlimited_d{detailed}_R{repetition}', **mixed('separate', 0, repetition, detailed=detailed))


def contention_summary():
    import csv
    return list(csv.DictReader((RESULTS / 'contention_summary.csv').open()))


def matched_levels():
    """E7.5/E7.6 targets: the write rates the thread-store writer achieved in E7.3 (separate GPCs)."""
    achieved = {row['level']: round(float(row['writer_with_copy_gbps'])) for row in contention_summary() if row['placement'] == 'separate'}
    return [(level, achieved[level]) for level, _ in rate_levels()]


def reader(**overrides):
    """E7.5 co-runner in the writer role: 48 CTAs of TMA bulk reads, 3 × 64 KiB in flight."""
    return dict(dict(writer_ctas=48, writer_threads=1024, writer_kind=1, chunks_in_flight=3, shared_bytes=TMA_READ_SHARED_BYTES), **overrides)


def phase_read_contrast(campaign):
    group = 'direction'
    for when in ('open',):
        campaign.run(group, f'reference_reads_{when}', binary=TMA_BINARY, **mixed('separate', 0, 1, writer_alternate=0, writer_idle=1, **reader()))
    read_targets = dict(matched_levels())
    for repetition in range(1, ROUNDS + 1):
        for level, write_target in rotated(rate_levels(), repetition):
            target = read_targets[level]
            for kind in rotated(['writes', 'reads'], repetition):
                label = f'mixed_{kind}_{level}_R{repetition}'
                if kind == 'writes':
                    # The E7.3 setting repeated in this session: thread-store writer at its own target.
                    campaign.run(group, label, **mixed('separate', write_target, repetition))
                    continue
                campaign.run(group, label, binary=TMA_BINARY, **mixed('separate', target, repetition, **reader()))
                window = writer_active_ns(RESULTS / group / 'raw' / f'{label}.json')
                campaign.run(group, f'reader_alone_{level}_R{repetition}', binary=TMA_BINARY,
                             **mixed('separate', target, repetition, writer_alternate=0, communication=0, solo_ns=window, warmup=2, iterations=6, **reader()))
    campaign.run(group, 'reference_reads_close', binary=TMA_BINARY, **mixed('separate', 0, 1, writer_alternate=0, writer_idle=1, **reader()))


TMA_WRITER_CTAS = [4, 8, 16, 24, 32, 48, 64]
TMA_WRITER_CHUNKS = [1, 2, 4]


def tma_writer(ctas, chunks, **overrides):
    return dict(dict(writer_ctas=ctas, writer_threads=1024, writer_kind=2, chunks_in_flight=chunks), **overrides)


def phase_tma_writer_sweep(campaign):
    grid = [(ctas, chunks) for ctas in TMA_WRITER_CTAS for chunks in TMA_WRITER_CHUNKS]
    for repetition in range(1, ROUNDS + 1):
        for ctas, chunks in rotated(grid, repetition):
            campaign.run('tma_writer_alone', f'tma_writer_w{ctas}_k{chunks}_unlimited_spread_R{repetition}', binary=TMA_BINARY,
                         **writer_alone(ctas, 1024, **tma_writer(ctas, chunks)))


def phase_tma_writer_rate(campaign):
    chosen = selection()['tma_writer']
    for repetition in range(1, ROUNDS + 1):
        for level, target in rotated(rate_levels(), repetition):
            campaign.run('tma_writer_alone', f"tma_writer_w{chosen['ctas']}_k{chosen['chunks']}_{level}_spread_R{repetition}", binary=TMA_BINARY,
                         **writer_alone(chosen['ctas'], 1024, target_gbps=target, **tma_writer(chosen['ctas'], chosen['chunks'])))


def phase_tma_store_contrast(campaign):
    group = 'mechanism'
    chosen = selection()['tma_writer']
    tma = tma_writer(chosen['ctas'], chosen['chunks'])
    campaign.run(group, 'reference_tma_open', binary=TMA_BINARY, **mixed('separate', 0, 1, writer_alternate=0, writer_idle=1, **tma))
    for repetition in range(1, ROUNDS + 1):
        for level, target in rotated(rate_levels(), repetition):
            for kind in rotated(['thread', 'tma'], repetition):
                label = f'mixed_{kind}_{level}_R{repetition}'
                if kind == 'thread':
                    campaign.run(group, label, **mixed('separate', target, repetition))
                    continue
                campaign.run(group, label, binary=TMA_BINARY, **mixed('separate', target, repetition, **tma))
                window = writer_active_ns(RESULTS / group / 'raw' / f'{label}.json')
                campaign.run(group, f'tma_alone_{level}_R{repetition}', binary=TMA_BINARY,
                             **mixed('separate', target, repetition, writer_alternate=0, communication=0, solo_ns=window, warmup=2, iterations=6, **tma))
    campaign.run(group, 'reference_tma_close', binary=TMA_BINARY, **mixed('separate', 0, 1, writer_alternate=0, writer_idle=1, **tma))


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
              writer_rate=phase_writer_rate, writer_placement=phase_writer_placement, contention=phase_contention,
              read_contrast=phase_read_contrast, tma_writer_sweep=phase_tma_writer_sweep, tma_writer_rate=phase_tma_writer_rate,
              tma_store_contrast=phase_tma_store_contrast)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=PHASES)
    parser.add_argument('--gpus', required=True)
    arguments = parser.parse_args()
    PHASES[arguments.phase](Campaign(arguments.gpus))


if __name__ == '__main__':
    main()
