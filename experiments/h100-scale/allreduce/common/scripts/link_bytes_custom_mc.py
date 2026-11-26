#!/usr/bin/env python3
"""Plan 06, E6.3: NVLink bytes per payload byte for the example kernels, from long and short run pairs.

Same method as link_bytes.py (plan 04): counters before and after each run; a long run (100 timed calls) minus a
short run (10) of the same cell and round leaves only the extra timed calls, so start-up, the built-in test,
warm-up and the result check cancel. Here each timed call is one AllReduce (no in-place pass), so
bytes per payload byte = (bytes_long - bytes_short) / ((iters_long - iters_short) * payload), per GPU and direction.
Output CSV column names are read by make_figures_custom_mc.py: do not rename.

Usage: link_bytes_custom_mc.py RAW_DIR LINK_JSONL OUTPUT_CSV
"""
import csv
import json
from pathlib import Path
import statistics
import sys

from link_bytes import run_bytes

LINK_RATING_GBPS = 478.0


def model(kernel, gpu_count):
    """(sent, received) per payload byte: one-shot answers every GPU's loads for the whole buffer."""
    if kernel == 'oneshot':
        return float(gpu_count), 1.0
    return 1 + 1 / gpu_count, 1 + 1 / gpu_count


def main():
    raw_dir, jsonl, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    records = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    by_key = {(record['cell_id'], record['round']): record for record in records if record['valid']}
    pairs = {}
    for (cell_id, round_index), record in by_key.items():
        if not cell_id.endswith('_long'):
            continue
        short = by_key.get((cell_id[:-5] + '_short', round_index))
        if short is None:
            continue
        kernel = record['rows'][0]['kernel']
        payload = record['rows'][0]['bytes']
        extra_calls = record['rows'][0]['iters'] - short['rows'][0]['iters']
        long_bytes = run_bytes(raw_dir, record['run'])
        short_bytes = run_bytes(raw_dir, short['run'])
        group = pairs.setdefault((record['gpu_count'], kernel, record['rows'][0]['ctas']), {'Tx': [], 'Rx': [], 'algbw': [], 'devices': record['devices']})
        for gpu in long_bytes:
            for direction in ('Tx', 'Rx'):
                group[direction].append((long_bytes[gpu][direction] - short_bytes[gpu][direction]) / (extra_calls * payload))
        group['algbw'].append(record['rows'][0]['algbw_gbps'])
    rows = []
    for (gpu_count, kernel, ctas), group in sorted(pairs.items()):
        model_sent, model_received = model(kernel, gpu_count)
        sent = statistics.median(group['Tx'])
        algbw = statistics.median(group['algbw'])
        rows.append({'gpu_count': gpu_count, 'devices': group['devices'], 'kernel': kernel, 'ctas_x_threads': f'{ctas}x1024',
                     'pairs': len(group['algbw']), 'sent_per_payload_byte_median': round(sent, 4),
                     'sent_min': round(min(group['Tx']), 4), 'sent_max': round(max(group['Tx']), 4),
                     'received_per_payload_byte_median': round(statistics.median(group['Rx']), 4),
                     'received_min': round(min(group['Rx']), 4), 'received_max': round(max(group['Rx']), 4),
                     'model_sent': round(model_sent, 4), 'model_received': round(model_received, 4),
                     'algbw_long_run_gbps': round(algbw, 2), 'sent_link_rate_gbps': round(algbw * sent, 1),
                     'sent_share_of_link_rating': round(algbw * sent / LINK_RATING_GBPS, 3)})
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} rows -> {output}')


if __name__ == '__main__':
    main()
