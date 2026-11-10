#!/usr/bin/env python3
"""Turns raw mm_mix records into one row per run: role byte rates, busBW, overlap, link bytes and state.

Role byte rate (N1): bytes finished by one role on one GPU, over the time from the role's first start
to its last end on that GPU, summarized as the mean over GPUs of one window, then the median over the
kept windows of one run. busBW (N2) follows the nccl-tests formulas for n = 4:
  multicast or reduce at x GB/s per rank -> 3x; allreduce at x per rank -> 6x; plain roles -> x.
"""
import json
import re
import statistics
import sys
from pathlib import Path

RANKS = 4
BUS_FACTOR = {'multicast': 3, 'reduce': 3, 'allreduce': 6, 'plain_load': 1, 'plain_store': 1}
LINK_LINE = re.compile(r'Link \d+: Data (Tx|Rx): (\d+) KiB')


def window_role_rates(window, chunk_bytes):
    """{role: (GB/s, start, end)} of one rank in one window."""
    roles = {}
    for cta in window['ctas']:
        if cta['role'] != 'controller':
            roles.setdefault(cta['role'], []).append(cta)
    rates = {}
    for role, ctas in roles.items():
        start = min(cta['start_ns'] for cta in ctas)
        end = max(cta['end_ns'] for cta in ctas)
        rates[role] = (sum(cta['chunks'] for cta in ctas) * chunk_bytes / (end - start), start, end)
    return rates


def link_totals(snapshot_path):
    """{gpu_position: (tx KiB, rx KiB)} from `nvidia-smi nvlink -gt d`, summed over links."""
    text = json.loads(snapshot_path.read_text())['link']['stdout']
    totals = []
    for block in re.split(r'(?=GPU \d+:)', text):
        if not block.strip():
            continue
        tx_total = 0
        rx_total = 0
        for direction, value in LINK_LINE.findall(block):
            if direction == 'Tx':
                tx_total += int(value)
            else:
                rx_total += int(value)
        totals.append((tx_total, rx_total))
    return totals


def telemetry_summary(path):
    """Median SM clock, largest power and the set of clock-event reasons while the GPUs were busy."""
    clocks = []
    powers = []
    reasons = set()
    for sample in json.loads(path.read_text()):
        for line in sample.splitlines():
            fields = [field.strip() for field in line.split(',')]
            if len(fields) < 8 or fields[6] in ('0', '[N/A]'):
                continue
            clocks.append(float(fields[2]))
            powers.append(float(fields[4]))
            reasons.add(fields[7])
    return {'sm_clock_mhz': statistics.median(clocks) if clocks else None, 'power_max_w': max(powers) if powers else None, 'clock_reasons': sorted(reasons)}


def summarize_run(json_path):
    record = json.loads(json_path.read_text())
    chunk_bytes = record['chunk_bytes']
    warmup = record['warmup']
    window_count = len(record['ranks'][0]['windows'])
    per_window = []
    for window_index in range(warmup, window_count):
        role_rates = {}
        overlaps = []
        spreads = {}
        for rank in record['ranks']:
            window = rank['windows'][window_index]
            rates = window_role_rates(window, chunk_bytes)
            for role, (rate, _, _) in rates.items():
                role_rates.setdefault(role, []).append(rate)
            if len(rates) == 2:
                values = list(rates.values())
                shared = min(value[2] for value in values) - max(value[1] for value in values)
                whole = max(value[2] for value in values) - min(value[1] for value in values)
                overlaps.append(shared / whole)
            for role in rates:
                chunks = [cta['chunks'] for cta in window['ctas'] if cta['role'] == role]
                spreads.setdefault(role, []).append(max(chunks) / min(chunks))
        per_window.append({'rates': role_rates, 'overlap': min(overlaps) if overlaps else None, 'spreads': spreads})
    summary = {
        'run': json_path.stem,
        'threads': record['threads'],
        'role_a': record['role_a'], 'ctas_a': record['ctas_a'], 'rank_mask_a': record['rank_mask_a'],
        'role_b': record['role_b'], 'ctas_b': record['ctas_b'], 'rank_mask_b': record['rank_mask_b'],
        'rank_offset': record['rank_offset'],
        'mismatches': sum(window['mismatches'] for rank in record['ranks'] for window in rank['windows']),
        'roles': {},
    }
    roles = per_window[0]['rates'].keys()
    for role in roles:
        means = [statistics.fmean(window['rates'][role]) for window in per_window]
        rate = statistics.median(means)
        summary['roles'][role] = {
            'rate_per_issuing_gpu_gbps': rate,
            'ranks_issuing': len(per_window[0]['rates'][role]),
            'bus_bw_gbps': rate * BUS_FACTOR[role],
            'window_min_gbps': min(means),
            'window_max_gbps': max(means),
            'cta_chunk_ratio_max_over_min': statistics.median(max(window['spreads'][role]) for window in per_window),
            'gpu_rates_gbps': [statistics.median(window['rates'][role][position] for window in per_window) for position in range(len(per_window[0]['rates'][role]))],
        }
    overlaps = [window['overlap'] for window in per_window if window['overlap'] is not None]
    summary['overlap_min'] = min(overlaps) if overlaps else None
    before = json_path.with_suffix('.before.json')
    after = json_path.with_suffix('.after.json')
    if before.exists() and after.exists():
        first = link_totals(before)
        last = link_totals(after)
        if first and len(first) == len(last):
            summary['link_gib_tx'] = [(last[gpu][0] - first[gpu][0]) / 2**20 for gpu in range(len(first))]
            summary['link_gib_rx'] = [(last[gpu][1] - first[gpu][1]) / 2**20 for gpu in range(len(first))]
            active_seconds = window_count * record['window_ns'] / 1e9
            summary['link_tx_gbps_estimate'] = [value * 2**30 / 1e9 / active_seconds for value in summary['link_gib_tx']]
            summary['link_rx_gbps_estimate'] = [value * 2**30 / 1e9 / active_seconds for value in summary['link_gib_rx']]
    telemetry = json_path.with_suffix('.telemetry.json')
    if telemetry.exists():
        summary.update(telemetry_summary(telemetry))
    return summary


def main():
    raw = Path(sys.argv[1])
    rows = [summarize_run(path) for path in sorted(raw.glob('*.json')) if not re.search(r'\.(before|after|telemetry)\.json$', path.name)]
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, indent=1) + '\n')
    for row in rows:
        roles = ' '.join(f"{role}:{values['rate_per_issuing_gpu_gbps']:.1f}x{values['ranks_issuing']} bus={values['bus_bw_gbps']:.1f}" for role, values in row['roles'].items())
        links = ''
        if 'link_rx_gbps_estimate' in row:
            links = f" linkRx={statistics.fmean(row['link_rx_gbps_estimate']):.0f} linkTx={statistics.fmean(row['link_tx_gbps_estimate']):.0f}"
        overlap = f" overlap={row['overlap_min']:.4f}" if row['overlap_min'] is not None else ''
        print(f"{row['run']}: {roles}{overlap}{links} clock={row.get('sm_clock_mhz')} bad={row['mismatches']}")


if __name__ == '__main__':
    main()
