#!/usr/bin/env python3
"""Data and raw NVLink bytes per payload byte from long-minus-short run pairs (RCA metrics R8, R9, R10).

Each setting runs as `<arm>_long_R<k>` and `<arm>_short_R<k>` in one session. The driver counters cover
the whole process, so start-up, set-up and checking traffic are in both runs; the difference keeps only
the traffic of the extra windows. Payload is counted as collective buffer bytes, N x the mean role bytes
of one GPU, so values compare with the 8-GPU nccl-tests tables (see rca/00_common_setup.md).

Usage: raw_link_bytes.py <raw directory> <output directory>
Writes rows.csv (one row per arm, round, GPU and direction) and summary.csv (median over GPUs and rounds).
Column names are an interface for the figure scripts; renaming a variable never renames a column.
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

RANKS = 4
GPU_HEADER = re.compile(r'^GPU (\d+):', re.MULTILINE)
COUNTER_LINE = re.compile(r'Link \d+: (Data|Raw) (Tx|Rx): (\d+) KiB')
RUN_NAME = re.compile(r'^(?P<arm>.+)_(?P<length>long|short)_R(?P<round>\d+)$')


def counter_totals(snapshot, key):
    """{physical gpu: {'tx': bytes, 'rx': bytes}} summed over links, from one nvidia-smi snapshot key."""
    text = snapshot[key]['stdout']
    totals = {}
    headers = list(GPU_HEADER.finditer(text))
    for position, header in enumerate(headers):
        block_end = headers[position + 1].start() if position + 1 < len(headers) else len(text)
        block = text[header.end():block_end]
        sums = {'tx': 0, 'rx': 0}
        for _kind, direction, kibibytes in COUNTER_LINE.findall(block):
            sums[direction.lower()] += int(kibibytes) * 1024
        totals[int(header.group(1))] = sums
    return totals


def counter_differences(run_path, key):
    before = counter_totals(json.loads(run_path.with_suffix('.before.json').read_text()), key)
    after = counter_totals(json.loads(run_path.with_suffix('.after.json').read_text()), key)
    return {gpu: {direction: after[gpu][direction] - before[gpu][direction] for direction in ('tx', 'rx')} for gpu in before}


def role_bytes_per_gpu(record):
    """{physical gpu: {role: bytes}} over all windows, warm-up included, because warm-up also moves data."""
    chunk_bytes = record['chunk_bytes']
    result = {}
    for rank in record['ranks']:
        per_role = {}
        for window in rank['windows']:
            for cta in window['ctas']:
                if cta['role'] != 'controller':
                    per_role[cta['role']] = per_role.get(cta['role'], 0) + cta['chunks'] * chunk_bytes
        result[rank['gpu']] = per_role
    return result


def model_data_bytes(role_bytes_by_rank, rank):
    """Data bytes the byte model puts on one rank's link directions, from the role bytes of every rank.

    multimem.st: the issuer sends its bytes once and the switch writes them into every GPU, so a GPU
    receives the multicast bytes of all issuers. ld_reduce: the issuer receives its sums once and every
    GPU serves the reads of all issuers. allreduce does both. Plain roles on the ring of plan 01: rank r
    stores to r + 1, and rank r loads from r + 1 (so r + 1 sends what r loads). Issue rates differ
    between GPUs in mixed arms, so the sums over issuers are taken, not N x this GPU's own bytes.
    """
    def role_of(other, role):
        return role_bytes_by_rank[other].get(role, 0)
    previous = (rank - 1) % RANKS
    total = {role: sum(role_of(other, role) for other in range(RANKS)) for role in ('multicast', 'reduce', 'allreduce')}
    known = {'multicast', 'reduce', 'allreduce', 'plain_load', 'plain_store'}
    for role in role_bytes_by_rank[rank]:
        if role not in known:
            raise ValueError(f'No byte model for role {role}')
    sent = role_of(rank, 'multicast') + total['reduce'] + role_of(rank, 'allreduce') + total['allreduce'] + role_of(rank, 'plain_store') + role_of(previous, 'plain_load')
    received = total['multicast'] + role_of(rank, 'reduce') + role_of(rank, 'allreduce') + total['allreduce'] + role_of(previous, 'plain_store') + role_of(rank, 'plain_load')
    return {'tx': sent, 'rx': received}


def kept_window_seconds(record):
    return len(record['ranks'][0]['windows']) * record['window_ns'] / 1e9


def pair_rows(arm, round_index, long_path, short_path):
    long_record = json.loads(long_path.read_text())
    short_record = json.loads(short_path.read_text())
    data_long = counter_differences(long_path, 'link')
    data_short = counter_differences(short_path, 'link')
    raw_long = counter_differences(long_path, 'link_raw')
    raw_short = counter_differences(short_path, 'link_raw')
    roles_long = role_bytes_per_gpu(long_record)
    roles_short = role_bytes_per_gpu(short_record)
    extra_seconds = kept_window_seconds(long_record) - kept_window_seconds(short_record)
    role_bytes_extra = {gpu: {role: roles_long[gpu][role] - roles_short[gpu].get(role, 0) for role in roles_long[gpu]} for gpu in roles_long}
    # Payload in buffer bytes: N x the mean over GPUs of one GPU's role bytes (mean over roles for two roles).
    mean_role_bytes = statistics.fmean(statistics.fmean(per_role.values()) for per_role in role_bytes_extra.values())
    payload_bytes = RANKS * mean_role_bytes
    rank_of_gpu = {rank['gpu']: rank['rank'] for rank in long_record['ranks']}
    role_bytes_by_rank = {rank_of_gpu[gpu]: roles for gpu, roles in role_bytes_extra.items()}
    rows = []
    for gpu in sorted(role_bytes_extra):
        model = model_data_bytes(role_bytes_by_rank, rank_of_gpu[gpu])
        for direction in ('tx', 'rx'):
            data_bytes = data_long[gpu][direction] - data_short[gpu][direction]
            raw_bytes = raw_long[gpu][direction] - raw_short[gpu][direction]
            rows.append({
                'arm': arm,
                'round': round_index,
                'gpu': gpu,
                'direction': 'sent' if direction == 'tx' else 'received',
                'role_a': long_record['role_a'],
                'ctas_a': long_record['ctas_a'],
                'role_b': long_record['role_b'],
                'ctas_b': long_record['ctas_b'],
                'threads_per_cta': long_record['threads'],
                'extra_window_seconds': extra_seconds,
                'payload_bytes': payload_bytes,
                'data_bytes_per_payload_byte': data_bytes / payload_bytes,  # R8
                'raw_bytes_per_payload_byte': raw_bytes / payload_bytes,  # R9
                'raw_over_data': raw_bytes / data_bytes if data_bytes else float('nan'),
                'data_over_model': data_bytes / model[direction] if model[direction] else float('nan'),
                'data_rate_gbps': data_bytes / extra_seconds / 1e9,
                'raw_rate_gbps': raw_bytes / extra_seconds / 1e9,  # R10
                'mismatches': sum(window['mismatches'] for record in (long_record, short_record) for rank in record['ranks'] for window in rank['windows']),
            })
    return rows


def find_pairs(raw_directory):
    runs = {}
    for path in sorted(raw_directory.glob('*.json')):
        match = RUN_NAME.match(path.stem)
        if match and not re.search(r'\.(before|after|telemetry)$', path.stem):
            runs.setdefault((match['arm'], int(match['round'])), {})[match['length']] = path
    return {key: value for key, value in runs.items() if 'long' in value and 'short' in value}


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['arm'], row['direction']), []).append(row)
    summary = []
    for (arm, direction), members in groups.items():
        first = members[0]
        entry = {key: first[key] for key in ('arm', 'direction', 'role_a', 'ctas_a', 'role_b', 'ctas_b', 'threads_per_cta')}
        entry['samples'] = len(members)
        for key in ('data_bytes_per_payload_byte', 'raw_bytes_per_payload_byte', 'raw_over_data', 'data_over_model', 'data_rate_gbps', 'raw_rate_gbps'):
            values = [member[key] for member in members]
            entry[key] = statistics.median(values)
            entry[key + '_min'] = min(values)
            entry[key + '_max'] = max(values)
        entry['mismatches'] = sum(member['mismatches'] for member in members)
        summary.append(entry)
    return summary


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    raw_directory = Path(sys.argv[1])
    output_directory = Path(sys.argv[2])
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for (arm, round_index), paths in sorted(find_pairs(raw_directory).items()):
        rows.extend(pair_rows(arm, round_index, paths['long'], paths['short']))
    write_csv(output_directory / 'rows.csv', rows)
    summary = summarize(rows)
    write_csv(output_directory / 'summary.csv', summary)
    for entry in summary:
        print(f"{entry['arm']:<40} {entry['direction']:<8} data/payload={entry['data_bytes_per_payload_byte']:.3f} raw/payload={entry['raw_bytes_per_payload_byte']:.3f} raw/data={entry['raw_over_data']:.3f} data/model={entry['data_over_model']:.4f} data={entry['data_rate_gbps']:.1f} raw={entry['raw_rate_gbps']:.1f} GB/s")


if __name__ == '__main__':
    main()
