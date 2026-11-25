#!/usr/bin/env python3
"""NVLink packet bytes per payload byte from Nsight Systems GPU metrics (RCA plan 01 metrics R1-R5).

The runner wraps a run in `nsys profile` with common/nsys/nvlink_packets.config (byte sums per sample)
and exports SQLite beside the run record. This script keeps only the samples whose timestamps fall
inside the mmMix kernel intervals of the same GPU (CUDA kernel trace, same clock as the samples), so
start-up, fill and check traffic are excluded without a long-minus-short pair. Payload counts all
windows, warm-up included, because every window is an mmMix kernel. Payload is in collective buffer
bytes (N x mean role bytes of one GPU), as in raw_link_bytes.py.

Usage: nsys_link_bytes.py <raw directory> <output directory> [run-name regex]
Writes nsys_rows.csv (one row per run, GPU and direction) and nsys_summary.csv (median over GPUs and
rounds). Column names are an interface for the figure scripts.
"""
import csv
import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw_link_bytes import RANKS, counter_differences, model_data_bytes, role_bytes_per_gpu  # noqa: E402

# Metric names of nvlink_packets.config -> (direction, packet kind, byte kind); "NVLink RX ..." is received.
PACKET_METRIC = re.compile(r'^NVLink (RX|TX) (Requests|Responses) (User|Protocol) Data')
CYCLE_METRIC = re.compile(r'^NVLink (RX|TX) Cycles (Active|Elapsed)')
RUN_SUFFIX = re.compile(r'_R(\d+)$')


def metric_types(connection):
    """{(typeId, metricId): ('received'|'sent', key)} for the packet and cycle metrics."""
    result = {}
    for type_id, metric_id, metric_name in connection.execute('SELECT typeId, metricId, metricName FROM TARGET_INFO_GPU_METRICS'):
        packet = PACKET_METRIC.match(metric_name)
        cycles = CYCLE_METRIC.match(metric_name)
        if packet:
            direction = 'received' if packet[1] == 'RX' else 'sent'
            result[(type_id, metric_id)] = (direction, f'{packet[2][:-1].lower()}_{packet[3].lower()}')
        elif cycles:
            direction = 'received' if cycles[1] == 'RX' else 'sent'
            result[(type_id, metric_id)] = (direction, f'cycles_{cycles[2].lower()}')
    return result


def gpu_of_metric_type(connection):
    """{typeId: GPU uuid}. GPU metrics are reported per GPU type; TARGET_INFO_GPU_METRICS names it.

    Nsight Systems 2026.4.1 leaves typeName empty. The low 32 bits of typeId are the GPU's id in
    TARGET_INFO_GPU (seen 2026-11-18: typeId = 2^48 + 2^32 + id for ids 4-7). The uuid then ties it to
    the harness rank, so CUDA ordering and nvidia-smi ordering never have to agree.
    """
    uuid_of_id = {gpu_id: uuid for gpu_id, uuid in connection.execute('SELECT id, uuid FROM TARGET_INFO_GPU')}
    result = {}
    for type_id, type_name in connection.execute('SELECT DISTINCT typeId, typeName FROM TARGET_INFO_GPU_METRICS'):
        match = re.search(r'GPU (\d+)', type_name or '')
        gpu_id = int(match[1]) if match else type_id & 0xFFFFFFFF
        if gpu_id not in uuid_of_id:
            raise RuntimeError(f'Cannot map GPU metrics type {type_id} "{type_name}" to a GPU; known ids {sorted(uuid_of_id)}')
        result[type_id] = uuid_of_id[gpu_id]
    return result


def kernel_intervals(connection):
    """{GPU uuid: [(start, end)]} of mmMix kernels from the CUDA trace."""
    uuid_of_device = {cuda_device: uuid for cuda_device, uuid in connection.execute('SELECT cuDevice, uuid FROM TARGET_INFO_GPU')}
    query = ('SELECT k.deviceId, k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL AS k '
             'JOIN StringIds AS s ON s.id = k.shortName WHERE s.value LIKE "%mmMix%" ORDER BY k.start')
    result = {}
    for device_id, start, end in connection.execute(query):
        result.setdefault(uuid_of_device[device_id], []).append((start, end))
    return result


def metric_sums(connection, intervals_of_gpu, types, gpu_of_type):
    """{GPU uuid: {direction: {key: sum}}} over samples inside that GPU's kernel intervals."""
    sums = {}
    samples = {}
    for (type_id, metric_id), (direction, key) in types.items():
        uuid = gpu_of_type[type_id]
        total = 0
        count = 0
        for start, end in intervals_of_gpu.get(uuid, []):
            value, rows = connection.execute('SELECT COALESCE(SUM(value), 0), COUNT(*) FROM GPU_METRICS WHERE typeId = ? AND metricId = ? AND timestamp > ? AND timestamp <= ?', (type_id, metric_id, start, end)).fetchone()
            total += value
            count += rows
        sums.setdefault(uuid, {}).setdefault(direction, {})[key] = total
        samples[uuid] = max(samples.get(uuid, 0), count)
    return sums, samples


def run_rows(record_path):
    database = Path(str(record_path.with_suffix('')) + '_nsys.sqlite')
    record = json.loads(record_path.read_text())
    connection = sqlite3.connect(database)
    # One range query per kernel interval; without this index each query scans every sample.
    connection.execute('CREATE INDEX IF NOT EXISTS gpu_metrics_by_metric ON GPU_METRICS(typeId, metricId, timestamp)')
    types = metric_types(connection)
    gpu_of_type = gpu_of_metric_type(connection)
    intervals = kernel_intervals(connection)
    sums, samples = metric_sums(connection, intervals, types, gpu_of_type)
    connection.close()
    role_bytes = role_bytes_per_gpu(record)
    rank_of_gpu = {rank['gpu']: rank['rank'] for rank in record['ranks']}
    # The harness writes 'GPU-<uuid>'; Nsight Systems stores the bare uuid.
    gpu_of_uuid = {rank['uuid'].removeprefix('GPU-'): rank['gpu'] for rank in record['ranks']}
    role_bytes_by_rank = {rank_of_gpu[gpu]: roles for gpu, roles in role_bytes.items()}
    mean_role_bytes = statistics.fmean(statistics.fmean(roles.values()) for roles in role_bytes.values() if roles)
    payload_bytes = RANKS * mean_role_bytes
    driver_data = counter_differences(record_path, 'link')
    driver_raw = counter_differences(record_path, 'link_raw') if 'link_raw' in json.loads(record_path.with_suffix('.before.json').read_text()) else None
    kernel_seconds = {uuid: sum(end - start for start, end in spans) / 1e9 for uuid, spans in intervals.items()}
    rows = []
    for uuid, per_direction in sorted(sums.items(), key=lambda item: gpu_of_uuid.get(item[0], 99)):
        if uuid not in gpu_of_uuid:
            continue
        gpu = gpu_of_uuid[uuid]
        model = model_data_bytes(role_bytes_by_rank, rank_of_gpu[gpu])
        for direction, values in per_direction.items():
            counter_key = 'tx' if direction == 'sent' else 'rx'
            user = values.get('request_user', 0) + values.get('response_user', 0)
            protocol = values.get('request_protocol', 0) + values.get('response_protocol', 0)
            row = {
                'run': record_path.stem,
                'arm': RUN_SUFFIX.sub('', record_path.stem),
                'round': int(RUN_SUFFIX.search(record_path.stem)[1]),
                'gpu': gpu,
                'direction': direction,
                'role_a': record['role_a'], 'ctas_a': record['ctas_a'], 'role_b': record['role_b'], 'ctas_b': record['ctas_b'],
                'threads_per_cta': record['threads'], 'access_bytes': record['pack_bytes'], 'pattern': record.get('pattern', 'contiguous'),
                'samples': samples.get(uuid, 0),
                'kernel_seconds': kernel_seconds.get(uuid, 0),
                'payload_bytes': payload_bytes,
                'request_user_per_payload_byte': values.get('request_user', 0) / payload_bytes,  # R1
                'request_protocol_per_payload_byte': values.get('request_protocol', 0) / payload_bytes,  # R2
                'response_user_per_payload_byte': values.get('response_user', 0) / payload_bytes,  # R3
                'response_protocol_per_payload_byte': values.get('response_protocol', 0) / payload_bytes,  # R4
                'user_over_model': user / model[counter_key] if model[counter_key] else float('nan'),
                'user_rate_gbps': user / kernel_seconds[uuid] / 1e9 if kernel_seconds.get(uuid) else float('nan'),
                'user_plus_protocol_rate_gbps': (user + protocol) / kernel_seconds[uuid] / 1e9 if kernel_seconds.get(uuid) else float('nan'),
                'user_over_driver_data': user / driver_data[gpu][counter_key] if driver_data[gpu][counter_key] else float('nan'),
                'total_over_driver_raw': (user + protocol) / driver_raw[gpu][counter_key] if driver_raw and driver_raw[gpu][counter_key] else float('nan'),
                # Whole-run driver counters; set-up traffic is below 0.01% of the window data (E1.3 test run).
                'driver_raw_per_payload_byte': driver_raw[gpu][counter_key] / payload_bytes if driver_raw else float('nan'),
                'raw_outside_packets_per_payload_byte': (driver_raw[gpu][counter_key] - user - protocol) / payload_bytes if driver_raw else float('nan'),
            }
            if 'cycles_elapsed' in values and values['cycles_elapsed']:
                row['busy_share'] = values['cycles_active'] / values['cycles_elapsed']  # R5
            rows.append(row)
    return rows


# In some runs the profiler loses the counters of 1-4 of a GPU's 18 links (its data bytes are then 17/18,
# 16/18, ... of the driver data counter, with no missing samples in time). Such GPU rows are kept in
# nsys_rows.csv but left out of the medians.
DRIVER_AGREEMENT_TOLERANCE = 0.005


def summarize(rows):
    groups = {}
    excluded = {}
    for row in rows:
        key = (row['arm'], row['direction'])
        if abs(row['user_over_driver_data'] - 1) > DRIVER_AGREEMENT_TOLERANCE:
            excluded[key] = excluded.get(key, 0) + 1
            continue
        groups.setdefault(key, []).append(row)
    summary = []
    numeric = [key for key in rows[0] if key.endswith(('_per_payload_byte', '_over_model', '_rate_gbps', '_over_driver_data', '_over_driver_raw')) or key == 'busy_share']
    for (arm, direction), members in groups.items():
        entry = {key: members[0][key] for key in ('arm', 'direction', 'role_a', 'ctas_a', 'role_b', 'ctas_b', 'threads_per_cta', 'access_bytes', 'pattern')}
        entry['samples'] = len(members)
        entry['excluded_gpu_rows'] = excluded.get((arm, direction), 0)
        for key in numeric:
            values = [member[key] for member in members if key in member]
            if values:
                entry[key] = statistics.median(values)
                entry[key + '_min'] = min(values)
                entry[key + '_max'] = max(values)
        summary.append(entry)
    return summary


def write_csv(path, rows):
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    raw_directory = Path(sys.argv[1])
    output_directory = Path(sys.argv[2])
    name_filter = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
    rows = []
    for record_path in sorted(raw_directory.glob('*_R*.json')):
        if re.search(r'\.(before|after|telemetry)$', record_path.stem) or (name_filter and not name_filter.search(record_path.stem)):
            continue
        if Path(str(record_path.with_suffix('')) + '_nsys.sqlite').exists():
            rows.extend(run_rows(record_path))
    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(output_directory / 'nsys_rows.csv', rows)
    summary = summarize(rows)
    write_csv(output_directory / 'nsys_summary.csv', summary)
    for entry in summary:
        print(f"{entry['arm']:<44} {entry['direction']:<8} R1={entry['request_user_per_payload_byte']:.3f} R2={entry['request_protocol_per_payload_byte']:.3f} R3={entry['response_user_per_payload_byte']:.3f} R4={entry['response_protocol_per_payload_byte']:.3f} user/model={entry['user_over_model']:.3f} user/driver={entry['user_over_driver_data']:.3f} total/raw={entry.get('total_over_driver_raw', float('nan')):.3f}")


if __name__ == '__main__':
    main()
