#!/usr/bin/env python3
"""Turns run summaries (JSONL from run_allreduce.py) into one row per cell and payload size.

algBW (metric S1) is the out-of-place algbw printed by nccl-tests. busBW (metric S6) uses the factor of
the arm: 2(N - 1)/N for Ring and plain load/store, (1 + 1/N) for the two multimem arms. Only valid runs count.
Output CSV column names are read by make_figures.py: do not rename.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

MULTIMEM_ARMS = {'nvls', 'sym_multicast'}


def bus_factor(arm, gpu_count, collective='allreduce'):
    """Bytes in the busiest link direction per payload byte (see each series' 00_common_setup.md).

    AllReduce: (1 + 1/N) multimem arms, 2(N - 1)/N otherwise. AllGather and ReduceScatter (payload = nccl-tests size,
    the whole buffer of one GPU): 1 for the multimem arms (the GPU's own piece also crosses the switch), (N - 1)/N otherwise."""
    if collective == 'allreduce':
        if arm in MULTIMEM_ARMS:
            return 1 + 1 / gpu_count
        return 2 * (gpu_count - 1) / gpu_count
    if arm in MULTIMEM_ARMS:
        return 1.0
    return (gpu_count - 1) / gpu_count


def link_model(arm, gpu_count, collective, direction):
    """Expected link bytes per payload byte for one direction ('Tx' sent, 'Rx' received), from the code."""
    if collective == 'allreduce' or arm not in MULTIMEM_ARMS:
        return bus_factor(arm, gpu_count, collective)
    small_direction = 'Tx' if collective == 'allgather' else 'Rx'
    return 1 / gpu_count if direction == small_direction else 1.0


def load_records(paths):
    records = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def channels_of(record):
    if record['sym_channel_values']:
        return ' '.join(str(value) for value in record['sym_channel_values'])
    return ' '.join(str(int(text.split('..')[1]) - int(text.split('..')[0]) + 1) for text in record['algo_channel_ranges'])


def threads_of(record):
    if record['sym_thread_values']:
        return ' '.join(str(value) for value in record['sym_thread_values'])
    return '640 (code)' if record['arm'] == 'nvls' else '512 (code)'


def summarize(records):
    groups = {}
    rejected = 0
    for record in records:
        if not record['valid']:
            rejected += 1
            continue
        for row in record['rows']:
            key = (record['cell_id'], record['arm'], record['gpu_count'], record['devices'], record['requested_ctas'], row['size_bytes'], record.get('collective', 'allreduce'))
            groups.setdefault(key, []).append((record, row))
    output = []
    for (cell_id, arm, gpu_count, devices, requested_ctas, size_bytes, collective), items in sorted(groups.items(), key=lambda item: (item[0][6], item[0][2], item[0][1], item[0][5], item[0][4] or 0)):
        algbw = [row['out_of_place_algbw_gbps'] for record, row in items]
        in_place = [row['in_place_algbw_gbps'] for record, row in items]
        factor = bus_factor(arm, gpu_count, collective)
        median_algbw = statistics.median(algbw)
        row_out = {
            'cell_id': cell_id, 'arm': arm, 'gpu_count': gpu_count, 'devices': devices,
            'requested_ctas': requested_ctas if requested_ctas else 'auto',
            'ctas_used': ' / '.join(sorted({channels_of(record) for record, row in items})),
            'threads_per_cta': ' / '.join(sorted({threads_of(record) for record, row in items})),
            'size_bytes': size_bytes, 'runs': len(items),
            'algbw_median_gbps': round(median_algbw, 2), 'algbw_min_gbps': round(min(algbw), 2), 'algbw_max_gbps': round(max(algbw), 2),
            'in_place_algbw_median_gbps': round(statistics.median(in_place), 2),
            'bus_factor': round(factor, 4), 'busbw_median_gbps': round(median_algbw * factor, 2),
        }
        # AllReduce rows keep their original columns; other collectives add one.
        if collective != 'allreduce':
            row_out['collective'] = collective
        output.append(row_out)
    return output, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_csv', type=Path)
    parser.add_argument('jsonl', nargs='+')
    arguments = parser.parse_args()
    rows, rejected = summarize(load_records(arguments.jsonl))
    with arguments.output_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} rows, {rejected} rejected runs -> {arguments.output_csv}')


if __name__ == '__main__':
    main()
