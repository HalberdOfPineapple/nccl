#!/usr/bin/env python3
"""Derive per-transfer and per-invocation tables without modifying raw observations."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower) if upper != lower else ordered[lower]


def summarize(path):
    data = json.loads(path.read_text())
    if 'ranks' not in data:
        return []
    rows = []
    for iteration_index, source_records in enumerate(data['ranks'][0]['iterations']):
        destination_records = data['ranks'][1]['iterations'][iteration_index]
        communication = [record for record in source_records if record['role'] == 0]
        started = min(record['start_ns'] for record in communication)
        completed = max(record['end_ns'] for record in communication)
        receiver = destination_records[0]
        # Alternating runs list idle (1) or reading (0) per retained launch; older records do not.
        pressure_rank_index = 1 if data['kind'] in (3, 5) else 0
        idle_list = data['ranks'][pressure_rank_index].get('pressure_idle_by_iteration')
        launch_idle = idle_list[iteration_index] if idle_list else data.get('pressure_idle', 0)
        row = dict(run=path.stem, iteration=iteration_index, config=Path(data['config_path']).stem,
                   kind=data['kind'], co_ctas=data['co_ctas'], ctas=data['ctas'], threads=data['threads'],
                   pacing=data['pacing'], detailed=data['detailed'], communication=data['communication'], vector_reads=data.get('vector_reads',0),
                   copy_batch=data.get('copy_batch',1), payload_bytes=data['bytes'],
                   memory_threads=data.get('memory_threads', data['threads']), block_threads=data.get('block_threads', data['threads']),
                   read_unroll=data.get('read_unroll', 4), target_gbps=data.get('target_gbps', 0), pressure_idle=launch_idle, pressure_alternate=data.get('pressure_alternate', 0),
                   completed_us=(completed-started)/1000,
                   throughput_gbps=data['bytes']/(completed-started) if data['communication'] else 0,
                   issue_us=statistics.median(record['issue_end_ns']-record['start_ns'] for record in communication)/1000 if data['detailed'] and data['communication'] else None,
                   ordering_us=statistics.median(record['fence_end_ns']-record['issue_end_ns'] for record in communication)/1000 if data['detailed'] and data['communication'] else None,
                   notification_us=statistics.median(record['publish_end_ns']-record['fence_end_ns'] for record in communication)/1000 if data['detailed'] and data['communication'] else None,
                   ack_wait_us=statistics.median(record['end_ns']-record['publish_end_ns'] for record in communication)/1000 if data['detailed'] and data['communication'] else None,
                   issue_spread_us=(max(record['issue_end_ns'] for record in communication)-min(record['issue_end_ns'] for record in communication))/1000 if data['detailed'] and data['communication'] else None,
                   receiver_wait_us=(receiver['end_ns']-receiver['start_ns'])/1000,
                   co_rate=0, minimum_overlap_fraction=0,
                   co_active_us=0, co_work_units=0)
        pressure_rank = 1 if data['kind'] in (3, 5) else 0
        pressure_records = destination_records if pressure_rank else source_records
        co_records = [record for record in pressure_records if record['role'] == 1]
        if co_records:
            pressure_start = min(record['start_ns'] for record in co_records)
            pressure_end = max(record['end_ns'] for record in co_records)
            work = sum(record['work_units'] for record in co_records)
            # Logical GB/s for reads, TFLOP/s for arithmetic. Includes pre-start/tail work.
            # Older records have no memory_threads key: their co-runner used the shared thread count.
            # Kinds 4/5 count chunks of chunk_kib KiB; plain-load reads count 256 KiB bursts.
            if data['kind'] == 1:
                per_unit = 256 * 8 * data.get('memory_threads', data['threads']) * 2 / 1000
            elif data['kind'] in (4, 5):
                per_unit = data['chunk_kib'] * 1024
            else:
                per_unit = 262144
            row['co_rate'] = work * per_unit / (pressure_end-pressure_start)
            row['co_active_us'] = (pressure_end-pressure_start)/1000
            row['co_work_units'] = work
            window_start = receiver['start_ns'] if pressure_rank else started
            window_end = receiver['end_ns'] if pressure_rank else completed
            row['minimum_overlap_fraction'] = min(max(0,min(record['end_ns'],window_end)-max(record['start_ns'],window_start))/(window_end-window_start) for record in co_records)
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    arguments = parser.parse_args()
    rows = []
    summaries = []
    for path in sorted(arguments.directory.glob('*.json')):
        observations = summarize(path)
        rows.extend(observations)
        if not observations:
            continue
        summary = {key: value for key, value in observations[0].items() if key not in ['iteration']}
        for key,value in list(summary.items()):
            if isinstance(value, (int,float)):
                summary[key] = statistics.median(row[key] for row in observations)
        summary['throughput_iqr'] = percentile([row['throughput_gbps'] for row in observations],.75)-percentile([row['throughput_gbps'] for row in observations],.25)
        summaries.append(summary)
    write_csv(arguments.directory/'transfers.csv', rows)
    write_csv(arguments.directory/'runs.csv', summaries)
    print(f'{len(summaries)} runs, {len(rows)} transfers')

if __name__ == '__main__':
    main()
