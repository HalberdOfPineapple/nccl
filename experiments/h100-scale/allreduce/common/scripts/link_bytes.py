#!/usr/bin/env python3
"""Plan 04: NVLink data bytes per payload byte (metric S4) from long and short run pairs.

Each run's snapshots hold `nvidia-smi nvlink -gt d` before and after (KiB per link, Tx and Rx). For a pair of
runs with the same cell, S4 = (bytes_long - bytes_short) / (2 * (iters_long - iters_short) * payload), per GPU and
direction: the factor 2 is because nccl-tests times out-of-place and in-place passes. Fixed calls cancel.
Output CSV column names are read by make_figures.py: do not rename.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

from summarize import bus_factor, link_model

LINK_LINE = re.compile(r'Link (\d+): (?:Data|Raw) (Tx|Rx): (\d+) KiB')
# Snapshot key per counter type: 'link' holds data payload (-gt d), 'link_raw' raw bytes (-gt r, recorded since 2026-11-18).
SNAPSHOT_KEYS = {'data': 'link', 'raw': 'link_raw'}
LINK_RATING_GBPS = 478.0


def per_gpu_totals(text):
    """{gpu index in the -i list: {'Tx': bytes, 'Rx': bytes}} summed over links."""
    totals = {}
    gpu = -1
    for line in text.splitlines():
        if line.startswith('GPU '):
            gpu += 1
            totals[gpu] = {'Tx': 0, 'Rx': 0}
            continue
        match = LINK_LINE.search(line)
        if match:
            totals[gpu][match[2]] += int(match[3]) * 1024
    return totals


def run_bytes(raw_dir, run_name, counter='data'):
    snapshots = json.loads((raw_dir / f'{run_name}.snapshots.json').read_text())
    before = per_gpu_totals(snapshots['before'][SNAPSHOT_KEYS[counter]])
    after = per_gpu_totals(snapshots['after'][SNAPSHOT_KEYS[counter]])
    return {gpu: {direction: after[gpu][direction] - before[gpu][direction] for direction in ('Tx', 'Rx')} for gpu in before}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir', type=Path)
    parser.add_argument('jsonl', type=Path)
    parser.add_argument('output_csv', type=Path)
    parser.add_argument('--long-iters', type=int, default=200)
    parser.add_argument('--short-iters', type=int, default=20)
    parser.add_argument('--counter', choices=['data', 'raw'], default='data', help='data payload (default) or raw link bytes')
    arguments = parser.parse_args()
    records = [json.loads(line) for line in arguments.jsonl.read_text().splitlines() if line.strip()]
    by_key = {(record['cell_id'], record['round']): record for record in records if record['valid']}
    rows = []
    for (cell_id, round_index), record in sorted(by_key.items()):
        if not cell_id.endswith('_long'):
            continue
        short = by_key.get((cell_id[:-5] + '_short', round_index))
        if short is None:
            continue
        payload = record['rows'][0]['size_bytes']
        extra_calls = 2 * (arguments.long_iters - arguments.short_iters)
        long_bytes = run_bytes(arguments.raw_dir, record['run'], arguments.counter)
        short_bytes = run_bytes(arguments.raw_dir, short['run'], arguments.counter)
        per_gpu = {direction: [(long_bytes[gpu][direction] - short_bytes[gpu][direction]) / (extra_calls * payload) for gpu in long_bytes] for direction in ('Tx', 'Rx')}
        algbw = record['rows'][0]['out_of_place_algbw_gbps']
        collective = record.get('collective', 'allreduce')
        for direction in ('Tx', 'Rx'):
            values = per_gpu[direction]
            row_out = {'cell_id': cell_id[:-5], 'arm': record['arm'], 'gpu_count': record['gpu_count'], 'devices': record['devices'],
                         'round': round_index, 'direction': direction, 'payload_bytes': payload,
                         'link_bytes_per_payload_byte_median': round(statistics.median(values), 4),
                         'link_bytes_per_payload_byte_min': round(min(values), 4), 'link_bytes_per_payload_byte_max': round(max(values), 4),
                         'model_bytes_per_payload_byte': round(link_model(record['arm'], record['gpu_count'], collective, direction), 4), 'algbw_long_run_gbps': algbw,
                         'link_rate_gbps': round(algbw * statistics.median(values), 1),
                         'share_of_link_rating': round(algbw * statistics.median(values) / LINK_RATING_GBPS, 3)}
            # AllReduce rows keep their original columns; other collectives add one.
            if collective != 'allreduce':
                row_out['collective'] = collective
            rows.append(row_out)
    with arguments.output_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} rows -> {arguments.output_csv}')


if __name__ == '__main__':
    main()
