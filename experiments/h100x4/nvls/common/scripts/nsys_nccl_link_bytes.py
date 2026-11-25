#!/usr/bin/env python3
"""NVLink packet bytes per payload byte for NCCL collectives run by nccl-tests under Nsight Systems
(RCA plan 01 E1.9, metrics R1-R4), with the runner of experiments/h100-scale/allreduce/common/scripts.

Every NCCL call is one collective kernel per GPU (ncclSymkDevKernel_* for the symmetric kernels,
ncclDevKernel_* for Ring) and moves one buffer of the run's size, so the payload of a GPU is the number
of its collective kernels x the size. Only GPU-metric samples inside those kernels are kept, so input
preparation and result checks of nccl-tests are excluded. Payload bytes are buffer bytes as in the
nccl-tests tables (AllGather: output size; ReduceScatter: input size; AllReduce: buffer size).

Usage: nsys_nccl_link_bytes.py <raw directory> <summary.jsonl of the runner> <output directory>
Writes nccl_rows.csv and nccl_summary.csv. Column names are an interface for the figure script.
"""
import csv
import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nsys_link_bytes import DRIVER_AGREEMENT_TOLERANCE, gpu_of_metric_type, metric_types  # noqa: E402

COLLECTIVE_KERNEL = re.compile(r'^(ncclSymkDevKernel_|ncclDevKernel_)')
RUN_NAME = re.compile(r'^(?P<cell>.+)_r(?P<round>\d+)$')


def data_model(collective, arm, ranks):
    """Data bytes per payload byte on (sent, received) of each GPU; measured exactly at 8 GPUs before."""
    if arm == 'ring':
        factor = {'allreduce': 2 * (ranks - 1) / ranks, 'allgather': (ranks - 1) / ranks, 'reducescatter': (ranks - 1) / ranks}[collective]
        return factor, factor
    return {'allreduce': (1 + 1 / ranks, 1 + 1 / ranks), 'allgather': (1 / ranks, 1.0), 'reducescatter': (1.0, 1 / ranks)}[collective]


def driver_data_bytes(snapshots, gpu_index):
    """Data counter difference of one GPU (sent, received), whole run, from the runner's snapshots."""
    def totals(text):
        result = {}
        for block in re.split(r'(?=^GPU \d+:)', text, flags=re.MULTILINE):
            header = re.match(r'GPU (\d+):', block)
            if not header:
                continue
            sent = sum(int(value) for value in re.findall(r'Data Tx: (\d+) KiB', block)) * 1024
            received = sum(int(value) for value in re.findall(r'Data Rx: (\d+) KiB', block)) * 1024
            result[int(header[1])] = (sent, received)
        return result
    before = totals(snapshots['before']['link'])
    after = totals(snapshots['after']['link'])
    return after[gpu_index][0] - before[gpu_index][0], after[gpu_index][1] - before[gpu_index][1]


def run_rows(record, raw_directory):
    stem = raw_directory / record['run']
    connection = sqlite3.connect(str(stem) + '_nsys.sqlite')
    connection.execute('CREATE INDEX IF NOT EXISTS gpu_metrics_by_metric ON GPU_METRICS(typeId, metricId, timestamp)')
    types = metric_types(connection)
    gpu_of_type = gpu_of_metric_type(connection)
    index_of_uuid = {uuid: cuda_device for cuda_device, uuid in connection.execute('SELECT cuDevice, uuid FROM TARGET_INFO_GPU')}
    uuid_of_device = {cuda_device: uuid for uuid, cuda_device in index_of_uuid.items()}
    intervals = {}
    for name, device_id, start, end in connection.execute('SELECT s.value, k.deviceId, k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS s ON s.id = k.shortName'):
        if COLLECTIVE_KERNEL.match(name):
            intervals.setdefault(uuid_of_device[device_id], []).append((start, end))
    sums = {}
    for (type_id, metric_id), (direction, key) in types.items():
        uuid = gpu_of_type[type_id]
        total = 0
        for start, end in intervals.get(uuid, []):
            total += connection.execute('SELECT COALESCE(SUM(value), 0) FROM GPU_METRICS WHERE typeId = ? AND metricId = ? AND timestamp > ? AND timestamp <= ?', (type_id, metric_id, start, end)).fetchone()[0]
        sums.setdefault(uuid, {}).setdefault(direction, {})[key] = total
    connection.close()
    snapshots = json.loads(Path(str(stem) + '.snapshots.json').read_text())
    size_bytes = record['rows'][0]['size_bytes']
    ranks = record['gpu_count']
    model_sent, model_received = data_model(record['collective'], record['arm'], ranks)
    rows = []
    for uuid, per_direction in sums.items():
        # CUDA_VISIBLE_DEVICES lists 0-7 in order, so the CUDA device index is the nvidia-smi index here.
        gpu = index_of_uuid[uuid]
        kernels = len(intervals.get(uuid, []))
        payload_bytes = kernels * size_bytes
        kernel_seconds = sum(end - start for start, end in intervals.get(uuid, [])) / 1e9
        driver_sent, driver_received = driver_data_bytes(snapshots, gpu)
        for direction, values in per_direction.items():
            user = values.get('request_user', 0) + values.get('response_user', 0)
            protocol = values.get('request_protocol', 0) + values.get('response_protocol', 0)
            model = model_sent if direction == 'sent' else model_received
            driver = driver_sent if direction == 'sent' else driver_received
            rows.append({
                'run': record['run'], 'cell': RUN_NAME.match(record['run'])['cell'], 'round': record['round'],
                'collective': record['collective'], 'arm': record['arm'], 'gpu_count': ranks,
                'ctas': record['sym_channel_values'][0] if record['sym_channel_values'] else record['algo_channel_ranges'][0],
                'threads_per_cta': record['sym_thread_values'][0] if record['sym_thread_values'] else 512,
                'gpu': gpu, 'direction': direction, 'kernels': kernels, 'size_bytes': size_bytes,
                'algbw_gbps': max(row['out_of_place_algbw_gbps'] for row in record['rows']),
                'request_user_per_payload_byte': values.get('request_user', 0) / payload_bytes,  # R1
                'request_protocol_per_payload_byte': values.get('request_protocol', 0) / payload_bytes,  # R2
                'response_user_per_payload_byte': values.get('response_user', 0) / payload_bytes,  # R3
                'response_protocol_per_payload_byte': values.get('response_protocol', 0) / payload_bytes,  # R4
                'packets_per_payload_byte': (user + protocol) / payload_bytes,
                'protocol_over_user': protocol / user if user else float('nan'),
                'user_over_model': user / payload_bytes / model,
                # Whole-run driver counter includes nccl-tests set-up and checks, so it is a little larger.
                'user_over_driver_data': user / driver if driver else float('nan'),
                'user_rate_gbps': user / kernel_seconds / 1e9 if kernel_seconds else float('nan'),
                'packets_rate_gbps': (user + protocol) / kernel_seconds / 1e9 if kernel_seconds else float('nan'),
            })
    return rows


def summarize(rows):
    groups = {}
    excluded = {}
    for row in rows:
        key = (row['cell'], row['direction'])
        # Same rule as the 4-GPU harness: a GPU whose profiler lost link counters shows user/model at
        # 17/18, 16/18, ...; it is left out of the medians.
        if abs(row['user_over_model'] - 1) > DRIVER_AGREEMENT_TOLERANCE:
            excluded[key] = excluded.get(key, 0) + 1
            continue
        groups.setdefault(key, []).append(row)
    summary = []
    for (cell, direction), members in groups.items():
        entry = {key: members[0][key] for key in ('cell', 'collective', 'arm', 'gpu_count', 'ctas', 'threads_per_cta', 'direction')}
        entry['samples'] = len(members)
        entry['excluded_gpu_rows'] = excluded.get((cell, direction), 0)
        for key in ('algbw_gbps', 'request_user_per_payload_byte', 'request_protocol_per_payload_byte', 'response_user_per_payload_byte', 'response_protocol_per_payload_byte', 'packets_per_payload_byte', 'protocol_over_user', 'user_over_model', 'user_over_driver_data', 'user_rate_gbps', 'packets_rate_gbps'):
            values = [member[key] for member in members]
            entry[key] = statistics.median(values)
            entry[key + '_min'] = min(values)
            entry[key + '_max'] = max(values)
        summary.append(entry)
    return summary


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    raw_directory = Path(sys.argv[1])
    records = [json.loads(line) for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()]
    output_directory = Path(sys.argv[3])
    rows = []
    for record in records:
        if record['valid'] and Path(str(raw_directory / record['run']) + '_nsys.sqlite').exists():
            rows.extend(run_rows(record, raw_directory))
    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(output_directory / 'nccl_rows.csv', rows)
    summary = summarize(rows)
    write_csv(output_directory / 'nccl_summary.csv', summary)
    for entry in summary:
        print(f"{entry['cell']:<36} {entry['direction']:<9} R1={entry['request_user_per_payload_byte']:.3f} R2={entry['request_protocol_per_payload_byte']:.3f} R3={entry['response_user_per_payload_byte']:.3f} R4={entry['response_protocol_per_payload_byte']:.3f} packets={entry['packets_per_payload_byte']:.3f} prot/user={entry['protocol_over_user']:.3f} user/model={entry['user_over_model']:.4f} algbw={entry['algbw_gbps']:.1f} n={entry['samples']} ex={entry['excluded_gpu_rows']}")


if __name__ == '__main__':
    main()
