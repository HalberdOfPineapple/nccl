#!/usr/bin/env python3
"""TMA plan 01 (TMA push next to TMA memory co-runners) runs on one GPU pair, one phase per invocation.

Usage: run_tma_congest.py PHASE --gpus SOURCE,DESTINATION [--date YYYY-MM-DD]

Phases, in order (E-numbers from observation/data_store_congest/01_data_store_congest.md):
  smoke             one short run of every role combination, to check the build before calibration
  copy_sweep        E1.1: TMA push alone; copy CTAs x staging chunks in flight, 64 KiB chunks
  copy_chunk        E1.1: at the selected CTA count, chunk size x fill threads/CTA, and slots filled
                    once (no per-chunk fill) as a contrast
  writer_sweep      E1.2: TMA store writer alone, no limit; CTAs x chunks in flight
  writer_rate       E1.2: the selected TMA store writer at 20/40/60/80% of HBM peak and no limit
  reader_rate       E1.2: the TMA read co-runner (48 CTAs x 3 chunks) at the same targets
  colocated_rate    E1.2: a TMA store writer in each copy CTA's last two warps, copy off, at its levels
  contention        E1.3: copy and TMA store writer; idle references; writer alone on the same SMs;
                    probe cost
  colocated         E1.5: writer on the copy's own SMs versus the same number of writer SMs in
                    other GPCs, at the same total targets
  read_contrast     E1.6: TMA reads instead of TMA stores, separate GPCs, targets matched to E1.3
  destination       E1.7: TMA store writer on the destination GPU, back to back with E1.3 separate
  thread_writer     E1.8 (optional): the plan-07 thread-store writer next to the TMA push
  destination_inflight  E1.7b (2026-11-13): writer on the destination at no limit, with more requests
                    outstanding on either side: writer CTAs x chunks in flight, copy CTAs x slots
  colocated_followup  Follow-up to E1.5 (2026-11-10): writer on the copy's SMs at very low rates
                    (1, 5, 10% of the same-SM ceiling) and at 25% / no limit, as TMA bulk stores
                    and as thread stores from the same two warps, back to back
Choices between phases are read from selection.json in the dated results directory, written by hand
after inspecting the previous phase. Configs carry no GPU numbers; the pair comes from --gpus and is
recorded in every raw record. Both GPUs' SM-to-GPC maps come from thread_store plan 07 (E7.0).
One placement throughout (user's choice, 2026-11-10): the copy packed into the largest GPCs (G6
first), co-runners spread over the 7 GPCs other than G6. Calibration uses the same layout.
"""
import argparse
import datetime
import json
import statistics
import subprocess
import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parents[1]
EXPERIMENT = COMMON.parent / 'observation/data_store_congest'
CONFIGS = EXPERIMENT / 'configs'
BINARY = COMMON / 'bin/tma_congest'
# Guarded wrapper shared with the thread-store harness: per-device locks, idle check, snapshots, telemetry.
RUN_ONE = COMMON.parents[1] / 'thread_store/common/scripts/run_one.py'
# SM-to-GPC maps measured in thread_store plan 07; the harness refuses a map whose UUID is not the GPU's.
GPC_MAPS = COMMON.parents[1] / 'thread_store/observation/data_store_congest/configs/gpc_map'
PEAK_GBPS = 3352.32                   # 2 x 2619 MHz x 5120 bits / 8
MAPPED_SMS = 124                      # SMs 124-131 are not placed by cluster launches (plan 07 E7.0)
# One shared-memory reservation for every run, so L1 size is the same in every arm: 3 x 64 KiB + 8
# barrier bytes each (the read co-runner), as in plan 07's TMA runs. It also fits a copy of up to
# 3 x 64 KiB slots, or 2 x 64 KiB slots plus a colocated 64 KiB TMA store writer.
SHARED_BYTES = 196864
ROUNDS = 3
TARGET_PERCENTS = [20, 40, 60, 80]
COPY_CTAS = [1, 2, 4, 8, 12, 16, 24, 32, 48]
COPY_SLOTS = [1, 2, 3]
COPY_CHUNK_KIB = [16, 32, 64]
COPY_FILL_THREADS = [128, 256, 512, 1024]   # at copy_chunk_sweep ctas (4, below the plateau)
WRITER_CTAS = [4, 8, 16, 24, 32, 48, 64]
WRITER_CHUNKS = [1, 2]
TMA_STORE, TMA_READ, THREAD_STORE = 0, 1, 2


def target_gbps(percent):
    return round(PEAK_GBPS * percent / 100)


def rate_levels():
    return [(f'p{percent}', target_gbps(percent)) for percent in TARGET_PERCENTS] + [('unlimited', 0)]


def rotated(items, repetition):
    offset = (repetition - 1) * max(1, len(items) // ROUNDS) % len(items)
    return items[offset:] + items[:offset]


def summarize(path):
    """One row per retained launch: copy throughput and phases (A1-A6, A15-A17), co-runner rate (A9), coverage (A8)."""
    data = json.loads(path.read_text())
    rows = []
    idle_list = data['ranks'][0]['writer_idle_by_iteration']
    writer_rank = data['writer_rank']
    for index, records in enumerate(data['ranks'][0]['iterations']):
        copy = [record for record in records if record['role'] == 'copy']
        writer_records = data['ranks'][writer_rank]['iterations'][index]
        writers = [record for record in writer_records if record['role'] == 'writer']
        row = dict(run=path.stem, iteration=index, writer_idle=idle_list[index])
        if data['communication']:
            started = min(record['start_ns'] for record in copy)
            completed = max(record['end_ns'] for record in copy)
            row.update(completed_us=(completed - started) / 1000, throughput_gbps=data['bytes'] / (completed - started))
            if data['detailed']:
                row.update(issue_us=statistics.median(record['issue_end_ns'] - record['start_ns'] for record in copy) / 1000,
                           bulk_wait_us=statistics.median(record['bulk_wait_end_ns'] - record['issue_end_ns'] for record in copy) / 1000,
                           ordering_us=statistics.median(record['fence_end_ns'] - record['bulk_wait_end_ns'] for record in copy) / 1000,
                           ack_wait_us=statistics.median(record['end_ns'] - record['publish_end_ns'] for record in copy) / 1000,
                           slot_wait_fraction=statistics.median(record['slot_wait_cycles'] / record['issue_loop_cycles'] for record in copy),
                           fill_fraction=statistics.median(record['fill_cycles'] / record['issue_loop_cycles'] for record in copy))
            window_start, window_end = started, completed
        else:
            timer_rank_records = writer_records if writer_rank else records
            timer = [record for record in timer_rank_records if record['role'] == ('receiver' if writer_rank else 'copy') and record['role_index'] == 0 and not record['colocated']][0]
            window_start, window_end = timer['publish_end_ns'], timer['end_ns']
        if writer_rank and data['communication']:
            # A8 on the destination: from just before "destination ready" until after the acknowledgement.
            receiver = [record for record in writer_records if record['role'] == 'receiver'][0]
            window_start, window_end = receiver['publish_end_ns'], receiver['end_ns']
        if writers and not row['writer_idle']:
            writer_start = min(record['start_ns'] for record in writers)
            writer_end = max(record['end_ns'] for record in writers)
            per_cta = [record['work_units'] * data['writer_block_bytes'] / (record['end_ns'] - record['start_ns']) for record in writers]
            # Requested bytes over the union of co-runner active intervals; includes start and tail chunks.
            row.update(writer_gbps=sum(record['work_units'] for record in writers) * data['writer_block_bytes'] / (writer_end - writer_start),
                       writer_slowest_to_fastest=min(per_cta) / max(per_cta),
                       minimum_overlap_fraction=min(max(0, min(record['end_ns'], window_end) - max(record['start_ns'], window_start)) / (window_end - window_start) for record in writers))
        rows.append(row)
    return data, rows


class Campaign:
    def __init__(self, gpus, date):
        self.gpus = gpus
        source, destination = gpus.split(',')
        self.gpc_map = GPC_MAPS / f'gpu{source}.json'
        self.destination_gpc_map = GPC_MAPS / f'gpu{destination}.json'
        for gpc_map in (self.gpc_map, self.destination_gpc_map):
            if not gpc_map.exists():
                raise SystemExit(f'No SM-to-GPC map {gpc_map}: run thread_store plan 07 E7.0 (bin/gpc_probe) on that GPU first')
        self.results = EXPERIMENT / 'results' / date

    def run(self, group, label, **overrides):
        """Write or confirm one config, run it once under the guard, validate the record."""
        values = dict(copy_ctas=16, copy_threads=512, copy_chunk_kib=64, copy_chunks_in_flight=2, copy_generate=1, copy_gpcs=255, copy_placement=0,
                      writer_kind=TMA_STORE, writer_ctas=0, writer_threads=1024, chunks_in_flight=1, writer_gpcs=255, writer_placement=0,
                      writer_colocated=0, writer_rank=0, receiver_gpcs=255, target_gbps=0, writer_idle=0, writer_alternate=0, use_unmapped=0,
                      block_threads=1024, shared_bytes=SHARED_BYTES, warmup=3, iterations=12, detailed=1, communication=1, payload_mib=2048, solo_ns=20000000)
        values.update(overrides)
        source_writers = 0 if values['writer_colocated'] or values['writer_rank'] else values['writer_ctas']
        values['use_unmapped'] = int(values['copy_ctas'] + source_writers > MAPPED_SMS)
        config = CONFIGS / group / f'{label}.cfg'
        config.parent.mkdir(parents=True, exist_ok=True)
        content = ''.join(f'{key}={value}\n' for key, value in values.items())
        if config.exists() and config.read_text() != content:
            raise RuntimeError(f'Configuration changed: {config}')
        config.write_text(content)
        output = self.results / group / 'raw' / f'{label}.json'
        if not output.exists():
            subprocess.run([sys.executable, str(RUN_ONE), str(config), str(output), '--binary', str(BINARY), '--gpus', self.gpus,
                            '--gpc-map', str(self.gpc_map), '--destination-gpc-map', str(self.destination_gpc_map)], check=True)
        data, rows = summarize(output)
        assert data['verified'] and len(rows) == values['iterations']
        writer_table = data['destination_role_of_sm'] if values['writer_rank'] else data['role_of_sm']
        assert data['role_of_sm'].count(1) == values['copy_ctas']
        assert values['writer_colocated'] or writer_table.count(2) == values['writer_ctas']
        if values['writer_ctas']:
            assert all(row['minimum_overlap_fraction'] == 1 for row in rows if not row['writer_idle']), 'co-runner did not cover the window'
        message = f'{group} {label}:'
        if values['communication']:
            message += f" copy {statistics.median(row['throughput_gbps'] for row in rows):.2f} GB/s"
        active = [row['writer_gbps'] for row in rows if not row['writer_idle'] and 'writer_gbps' in row]
        if active:
            message += f" co-runner {statistics.median(active):.0f} GB/s"
        print(message, flush=True)
        return rows

    def selection(self):
        return json.loads((self.results / 'selection.json').read_text())


def writer_active_ns(path):
    """Median co-runner active interval over the active transfers of one mixed run."""
    data = json.loads(path.read_text())
    spans = []
    for idle, records in zip(data['ranks'][0]['writer_idle_by_iteration'], data['ranks'][data['writer_rank']]['iterations']):
        writers = [record for record in records if record['role'] == 'writer']
        if not idle:
            spans.append(max(record['end_ns'] for record in writers) - min(record['start_ns'] for record in writers))
    return int(statistics.median(spans))


def copy_settings(chosen):
    copy = chosen['copy']
    return dict(copy_ctas=copy['ctas'], copy_threads=copy['threads'], copy_chunk_kib=copy['chunk_kib'], copy_chunks_in_flight=copy['slots'])


def tma_writer(chosen):
    return dict(writer_kind=TMA_STORE, writer_ctas=chosen['writer']['ctas'], chunks_in_flight=chosen['writer']['chunks'])


def reader():
    """Read co-runner from plan 04 / plan 07 E7.5: 48 CTAs of TMA bulk reads, 3 x 64 KiB in flight."""
    return dict(writer_kind=TMA_READ, writer_ctas=48, chunks_in_flight=3)


# The one placement of this plan. GPC masks: bit g = GPC g of the plan-07 map. Copy packed into the
# largest GPCs, G6 (18 SMs) first; co-runners spread over the seven GPCs other than G6, so they never
# share a GPC with up to 18 copy CTAs. Calibration runs use the same layout (COPY_LAYOUT, WRITER_LAYOUT).
COPY_LAYOUT = dict(copy_gpcs=255, copy_placement=1)
WRITER_LAYOUT = dict(writer_gpcs=255 & ~(1 << 6), writer_placement=0)
PLACEMENTS = {'separate': dict(COPY_LAYOUT, **WRITER_LAYOUT)}


def mixed(chosen, placement, target, round_number, co_runner, **overrides):
    """One run of 6 idle + 6 active transfers; odd rounds start idle, even rounds start active."""
    values = dict(copy_settings(chosen), target_gbps=target, writer_alternate=1 if round_number % 2 else 2)
    values.update(PLACEMENTS[placement])
    values.update(co_runner)
    values.update(overrides)
    return values


def alone(values, label_path):
    """The same SMs with the copy off; the window is matched to the co-runner's active time in the mixed run."""
    return dict(values, writer_alternate=0, communication=0, solo_ns=writer_active_ns(label_path), warmup=2, iterations=6)


def interleaved(campaign, group, chosen, placement_of_kind, co_runner_of_kind, targets_of_kind):
    """For each level and round, run each kind back to back (rotated) plus its co-runner-alone run."""
    kinds = list(placement_of_kind)
    for repetition in range(1, ROUNDS + 1):
        for level, _ in rotated(rate_levels(), repetition):
            for kind in rotated(kinds, repetition):
                label = f'mixed_{kind}_{level}_R{repetition}'
                values = mixed(chosen, placement_of_kind[kind], targets_of_kind[kind][level], repetition, co_runner_of_kind[kind])
                campaign.run(group, label, **values)
                campaign.run(group, f'alone_{kind}_{level}_R{repetition}', **alone(values, campaign.results / group / 'raw' / f'{label}.json'))


def references(campaign, group, chosen, placement_of_kind, co_runner_of_kind, when):
    for kind, placement in placement_of_kind.items():
        campaign.run(group, f'reference_{kind}_{when}', **mixed(chosen, placement, 0, 1, co_runner_of_kind[kind], writer_alternate=0, writer_idle=1))


def phase_smoke(campaign):
    """Every role combination once, short payload, to check the build; not used in any result."""
    short = dict(payload_mib=256, warmup=1, iterations=2)
    campaign.run('smoke', 'copy_alone', **short)
    campaign.run('smoke', 'copy_fill_once', copy_generate=0, **short)
    campaign.run('smoke', 'tma_writer_alone', copy_ctas=1, writer_ctas=8, communication=0, solo_ns=5000000, **short)
    campaign.run('smoke', 'tma_writer_mixed', writer_ctas=8, writer_alternate=1, **short)
    campaign.run('smoke', 'reader_mixed', writer_alternate=1, **dict(short, **reader()))
    campaign.run('smoke', 'thread_writer_mixed', writer_kind=THREAD_STORE, writer_ctas=8, writer_alternate=1, **short)
    campaign.run('smoke', 'colocated_mixed', writer_colocated=1, writer_ctas=16, writer_alternate=1, **short)
    campaign.run('smoke', 'colocated_alone', writer_colocated=1, writer_ctas=16, communication=0, solo_ns=5000000, **short)
    campaign.run('smoke', 'destination_mixed', writer_rank=1, writer_ctas=8, writer_gpcs=255 & ~(1 << 6), receiver_gpcs=1 << 6, writer_alternate=1, **short)
    campaign.run('smoke', 'destination_alone', writer_rank=1, writer_ctas=8, writer_gpcs=255 & ~(1 << 6), receiver_gpcs=1 << 6, communication=0, solo_ns=5000000, **short)


def phase_copy_sweep(campaign):
    grid = [(ctas, slots) for ctas in COPY_CTAS for slots in COPY_SLOTS]
    for repetition in range(1, ROUNDS + 1):
        for ctas, slots in rotated(grid, repetition):
            campaign.run('copy_alone', f'copy_c{ctas}_t512_k64_s{slots}_packed_R{repetition}', copy_ctas=ctas, copy_chunks_in_flight=slots, **COPY_LAYOUT)


def phase_copy_chunk(campaign):
    chosen = campaign.selection()['copy_chunk_sweep']
    grid = [(chunk, threads, 1) for chunk in COPY_CHUNK_KIB for threads in COPY_FILL_THREADS] + [(chunk, 512, 0) for chunk in COPY_CHUNK_KIB]
    for repetition in range(1, ROUNDS + 1):
        for chunk, threads, generate in rotated(grid, repetition):
            label = f"copy_c{chosen['ctas']}_t{threads}_k{chunk}_s{chosen['slots']}_{'fill' if generate else 'once'}_packed_R{repetition}"
            campaign.run('copy_alone', label, copy_ctas=chosen['ctas'], copy_threads=threads, copy_chunk_kib=chunk, copy_chunks_in_flight=chosen['slots'], copy_generate=generate, **COPY_LAYOUT)


def co_runner_alone(**overrides):
    # Copy off: copy CTA 0 (in G6) only times six 20 ms windows; every other SM holds an idle CTA.
    return dict(dict(copy_ctas=1, communication=0, warmup=2, iterations=6, **COPY_LAYOUT, **WRITER_LAYOUT), **overrides)


def phase_writer_sweep(campaign):
    grid = [(ctas, chunks) for ctas in WRITER_CTAS for chunks in WRITER_CHUNKS]
    for repetition in range(1, ROUNDS + 1):
        for ctas, chunks in rotated(grid, repetition):
            campaign.run('co_runner_alone', f'tma_writer_w{ctas}_k{chunks}_unlimited_R{repetition}', **co_runner_alone(writer_ctas=ctas, chunks_in_flight=chunks))


def phase_writer_rate(campaign):
    writer = tma_writer(campaign.selection())
    for repetition in range(1, ROUNDS + 1):
        for level, target in rotated(rate_levels(), repetition):
            campaign.run('co_runner_alone', f"tma_writer_w{writer['writer_ctas']}_k{writer['chunks_in_flight']}_{level}_R{repetition}",
                         **co_runner_alone(target_gbps=target, **writer))


def phase_reader_rate(campaign):
    for repetition in range(1, ROUNDS + 1):
        for level, target in rotated(rate_levels(), repetition):
            campaign.run('co_runner_alone', f'reader_w48_k3_{level}_R{repetition}',
                         **co_runner_alone(target_gbps=target, **reader()))


def colocated_levels(chosen):
    """E1.5 targets: fractions of the colocated writer's ceiling alone on the copy's SMs (selection.json)."""
    ceiling = chosen['colocated_ceiling_gbps']
    return [(f'c{percent}', round(ceiling * percent / 100)) for percent in (25, 50, 75)] + [('unlimited', 0)]


def phase_colocated_rate(campaign):
    chosen = campaign.selection()
    copy = copy_settings(chosen)
    for repetition in range(1, ROUNDS + 1):
        campaign.run('co_runner_alone', f"colocated_writer_w{copy['copy_ctas']}_unlimited_R{repetition}",
                     **dict(copy, **PLACEMENTS['separate'], writer_colocated=1, writer_ctas=copy['copy_ctas'], communication=0, warmup=2, iterations=6))


def phase_contention(campaign):
    group = 'contention'
    chosen = campaign.selection()
    placements = {placement: placement for placement in PLACEMENTS}
    co_runners = {placement: tma_writer(chosen) for placement in PLACEMENTS}
    targets = {placement: dict(rate_levels()) for placement in PLACEMENTS}
    references(campaign, group, chosen, placements, co_runners, 'open')
    interleaved(campaign, group, chosen, placements, co_runners, targets)
    references(campaign, group, chosen, placements, co_runners, 'close')
    # Probe cost: the same mixed runs with the per-CTA phase timers, cycle counters and checkpoints off.
    for repetition in range(1, ROUNDS + 1):
        for detailed in (1, 0):
            campaign.run(group, f'probe_separate_unlimited_d{detailed}_R{repetition}', **mixed(chosen, 'separate', 0, repetition, tma_writer(chosen), detailed=detailed))


def phase_colocated(campaign):
    group = 'colocated'
    chosen = campaign.selection()
    copy_ctas = chosen['copy']['ctas']
    same_sm = dict(writer_kind=TMA_STORE, writer_colocated=1, writer_ctas=copy_ctas, chunks_in_flight=chosen['writer']['chunks'])
    other_sms = dict(writer_kind=TMA_STORE, writer_ctas=copy_ctas, chunks_in_flight=chosen['writer']['chunks'])
    placements = {'same_sm': 'separate', 'other_sms': 'separate'}
    co_runners = {'same_sm': same_sm, 'other_sms': other_sms}
    levels = dict(colocated_levels(chosen))
    targets = {'same_sm': levels, 'other_sms': levels}
    references(campaign, group, chosen, placements, co_runners, 'open')
    for repetition in range(1, ROUNDS + 1):
        for level, _ in rotated(colocated_levels(chosen), repetition):
            for kind in rotated(list(placements), repetition):
                label = f'mixed_{kind}_{level}_R{repetition}'
                values = mixed(chosen, placements[kind], targets[kind][level], repetition, co_runners[kind])
                campaign.run(group, label, **values)
                campaign.run(group, f'alone_{kind}_{level}_R{repetition}', **alone(values, campaign.results / group / 'raw' / f'{label}.json'))
    references(campaign, group, chosen, placements, co_runners, 'close')


def followup_levels(chosen):
    """Same-SM follow-up: onset levels below E1.5's lowest (25%), plus 25% and no limit to join E1.5."""
    ceiling = chosen['colocated_ceiling_gbps']
    return [(f'c{percent}', round(ceiling * percent / 100)) for percent in (1, 5, 10, 25)] + [('unlimited', 0)]


def phase_colocated_followup(campaign):
    """Is the same-SM loss present with the first chunks, and does it need the writer's bytes to use TMA?
    Both writers run in the last two warps of each copy CTA; the thread-store writer stores with those
    64 threads (16-byte stores), so its bytes leave the SM without the TMA unit."""
    group = 'colocated_followup'
    chosen = campaign.selection()
    copy_ctas = chosen['copy']['ctas']
    placements = {'same_sm_tma': 'separate', 'same_sm_thread': 'separate'}
    co_runners = {'same_sm_tma': dict(writer_kind=TMA_STORE, writer_colocated=1, writer_ctas=copy_ctas, chunks_in_flight=chosen['writer']['chunks']),
                  'same_sm_thread': dict(writer_kind=THREAD_STORE, writer_colocated=1, writer_ctas=copy_ctas, writer_threads=64)}
    levels = dict(followup_levels(chosen))
    references(campaign, group, chosen, placements, co_runners, 'open')
    for repetition in range(1, ROUNDS + 1):
        for level, _ in rotated(followup_levels(chosen), repetition):
            for kind in rotated(list(placements), repetition):
                label = f'mixed_{kind}_{level}_R{repetition}'
                values = mixed(chosen, placements[kind], levels[level], repetition, co_runners[kind])
                campaign.run(group, label, **values)
                campaign.run(group, f'alone_{kind}_{level}_R{repetition}', **alone(values, campaign.results / group / 'raw' / f'{label}.json'))
    references(campaign, group, chosen, placements, co_runners, 'close')


# E1.7b: requests outstanding on each side, at the only E1.7 level where the destination HBM is full.
INFLIGHT_WRITERS = [(64, 1), (64, 2), (64, 4), (105, 2)]   # destination writer CTAs x chunks in flight; 105 of the 106 mapped non-G6 SMs
INFLIGHT_COPIES = [(16, 2), (16, 3), (48, 2)]               # copy CTAs x staging slots (64 KiB chunks, 512 fill threads)


def inflight_kind(writer, copy):
    return f'w{writer[0]}k{writer[1]}_c{copy[0]}s{copy[1]}'


def phase_destination_inflight(campaign):
    """Does the destination's split of a full HBM between the incoming push and a local TMA writer depend on how
    many requests each side keeps outstanding? Writer on the destination (E1.7 layout), no rate limit only.
    Each cell is one mixed run per round. The writer-alone run does not depend on the copy (the copy is off),
    so it runs once per writer setting per round, with the window of that round's 16 x 2 cell."""
    group = 'destination_inflight'
    chosen = campaign.selection()
    placement = dict(PLACEMENTS['separate'], writer_rank=1, receiver_gpcs=1 << 6, writer_kind=TMA_STORE)

    def cell(writer, copy, repetition, **overrides):
        values = dict(copy_settings(chosen), copy_ctas=copy[0], copy_chunks_in_flight=copy[1], target_gbps=0,
                      writer_alternate=1 if repetition % 2 else 2, writer_ctas=writer[0], chunks_in_flight=writer[1])
        values.update(placement)
        values.update(overrides)
        return values

    for copy in INFLIGHT_COPIES:
        campaign.run(group, f'reference_c{copy[0]}s{copy[1]}_open', **cell(INFLIGHT_WRITERS[0], copy, 1, writer_alternate=0, writer_idle=1))
    for repetition in range(1, ROUNDS + 1):
        for writer in rotated(INFLIGHT_WRITERS, repetition):
            for copy in rotated(INFLIGHT_COPIES, repetition):
                campaign.run(group, f'mixed_{inflight_kind(writer, copy)}_unlimited_R{repetition}', **cell(writer, copy, repetition))
            matched = campaign.results / group / 'raw' / f'mixed_{inflight_kind(writer, INFLIGHT_COPIES[0])}_unlimited_R{repetition}.json'
            campaign.run(group, f'alone_w{writer[0]}k{writer[1]}_unlimited_R{repetition}', **alone(cell(writer, INFLIGHT_COPIES[0], repetition), matched))
    for copy in INFLIGHT_COPIES:
        campaign.run(group, f'reference_c{copy[0]}s{copy[1]}_close', **cell(INFLIGHT_WRITERS[0], copy, 1, writer_alternate=0, writer_idle=1))


def matched_levels(campaign):
    """E1.6/E1.8 targets: the write rates the TMA store writer achieved during the copy in E1.3 (separate GPCs)."""
    import csv
    rows = csv.DictReader((campaign.results / 'contention_summary.csv').open())
    return {row['level']: round(float(row['writer_with_copy_gbps'])) for row in rows if row['placement'] == 'separate'}


def phase_read_contrast(campaign):
    group = 'direction'
    chosen = campaign.selection()
    placements = {'writes': 'separate', 'reads': 'separate'}
    co_runners = {'writes': tma_writer(chosen), 'reads': reader()}
    targets = {'writes': dict(rate_levels()), 'reads': matched_levels(campaign)}
    references(campaign, group, chosen, placements, co_runners, 'open')
    interleaved(campaign, group, chosen, placements, co_runners, targets)
    references(campaign, group, chosen, placements, co_runners, 'close')


def phase_destination(campaign):
    group = 'destination'
    chosen = campaign.selection()
    # Receiver in G6 on both arms; the co-runner's mask excludes G6, so it uses the same SM IDs on either GPU.
    placements = {'destination': 'separate', 'source': 'separate'}
    co_runners = {'destination': dict(tma_writer(chosen), writer_rank=1, receiver_gpcs=1 << 6), 'source': dict(tma_writer(chosen), receiver_gpcs=1 << 6)}
    targets = {kind: dict(rate_levels()) for kind in placements}
    references(campaign, group, chosen, placements, co_runners, 'open')
    interleaved(campaign, group, chosen, placements, co_runners, targets)
    references(campaign, group, chosen, placements, co_runners, 'close')


def phase_thread_writer(campaign):
    group = 'mechanism'
    chosen = campaign.selection()
    placements = {'tma': 'separate', 'thread': 'separate'}
    co_runners = {'tma': tma_writer(chosen), 'thread': dict(writer_kind=THREAD_STORE, writer_ctas=64, writer_threads=1024)}
    targets = {kind: dict(rate_levels()) for kind in placements}
    references(campaign, group, chosen, placements, co_runners, 'open')
    interleaved(campaign, group, chosen, placements, co_runners, targets)
    references(campaign, group, chosen, placements, co_runners, 'close')


PHASES = dict(smoke=phase_smoke, copy_sweep=phase_copy_sweep, copy_chunk=phase_copy_chunk,
              writer_sweep=phase_writer_sweep, writer_rate=phase_writer_rate, reader_rate=phase_reader_rate, colocated_rate=phase_colocated_rate,
              contention=phase_contention, colocated=phase_colocated, read_contrast=phase_read_contrast, destination=phase_destination,
              thread_writer=phase_thread_writer, colocated_followup=phase_colocated_followup, destination_inflight=phase_destination_inflight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=PHASES)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--date', default=datetime.date.today().isoformat(), help='dated results directory; keep one per session')
    arguments = parser.parse_args()
    PHASES[arguments.phase](Campaign(arguments.gpus, arguments.date))


if __name__ == '__main__':
    main()
