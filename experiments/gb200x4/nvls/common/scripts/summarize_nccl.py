#!/usr/bin/env python3
"""Turn one or more nccl_runner.py session directories into tables (CSV). No GPU needed.

Usage: summarize_nccl.py <session_dir>... --output <dir>

Writes:
  runs.csv          one row per run: configuration, status, launched kernel and CTAs, timing,
                    busBW by the per-direction factor (S6), and link-counter bytes per GPU/direction
  configs.csv       per (session, block, arm, CTAs, size): median and IQR over rounds of out-of-place
                    algBW (S1) and busBW (S6); only runs with status 'ok'
  counters.csv      per long/short pair: data and raw bytes per payload byte (R8, R9), raw link rate
                    (R10), raw per data byte (G5) and ceiling share (G2), per GPU and direction

Payload and time for counters follow 00_common_setup.md: long minus short leaves
(n_long - n_short) timed calls; each call is one out-of-place and one in-place collective in
nccl-tests, so payload bytes = 2 x (n_long - n_short) x buffer size, and time = the same calls x the
long run's mean times. Warm-up calls are equal in both runs and cancel.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

GPU_COUNT = 4
NOMINAL_LINK_GBPS = 18 * 53.125  # per direction per GPU, nvidia-smi nvlink -s

# Data bytes per payload byte on the busier direction (00_common_setup.md, N = 4).
# nccl-tests 'size' is the full buffer for AllGather (output) and ReduceScatter (input).
BUSBW_FACTOR = {
    # AG-CE copies each GPU's part to the 3 peers (unicast), so each direction carries (N - 1)/N, like
    # Ring; measured 0.750 with the data counter on 2026-11-26.
    'AG-MC': 1.0, 'AG-CE': (GPU_COUNT - 1) / GPU_COUNT, 'RS-MC': 1.0,
    'AR-MC': 1 + 1 / GPU_COUNT, 'AR-NVLS': 1 + 1 / GPU_COUNT,
    'AR-Ring': 2 * (GPU_COUNT - 1) / GPU_COUNT, 'AR-TMA': 2 * (GPU_COUNT - 1) / GPU_COUNT,
    'AR-LDST': 2 * (GPU_COUNT - 1) / GPU_COUNT, 'AR-Tuner': None,
    'AG-Ring': (GPU_COUNT - 1) / GPU_COUNT, 'RS-Ring': (GPU_COUNT - 1) / GPU_COUNT,
}


def loadRecords(sessionDirectory):
    """All result.json records of a session, in run order."""
    records = []
    for resultPath in sorted((sessionDirectory / 'raw').glob('*/result.json')):
        record = json.loads(resultPath.read_text())
        record['session'] = sessionDirectory.name
        records.append(record)
    return records


def counterDelta(before, after, counterType):
    """KiB moved per GPU and direction between two snapshots, summed over the 18 links."""
    delta = {}
    for gpu, directions in after[counterType].items():
        for direction, links in directions.items():
            previous = before[counterType][gpu][direction]
            delta[(gpu, direction)] = 1024 * (sum(links) - sum(previous))
    return delta


def launchedSummary(record):
    """Most frequent launch line: kernel/algorithm and CTA count NCCL actually used."""
    if not record['launches']:
        return ''
    return max(record['launches'].items(), key=lambda item: item[1])[0]


def runRow(record):
    run = record['run']
    row = {
        'session': record['session'], 'run_id': run['run_id'], 'block': run['block'], 'round': run['round'],
        'arm': run['arm'], 'ctas_requested': run['ctas'] if run['ctas'] is not None else 'default',
        'size': run['size'], 'length': run['length'], 'iterations': run['iterations'], 'warmup': run['warmup'],
        'status': record['status'], 'launched': launchedSummary(record),
        'wall_seconds': round(record['wall_seconds'], 2),
    }
    factor = BUSBW_FACTOR.get(run['arm'])
    if record['rows']:
        table = record['rows'][0]
        row.update({key: table[key] for key in ('size_bytes', 'out_of_place_time_us', 'out_of_place_algbw_gbps',
                                                'in_place_time_us', 'in_place_algbw_gbps',
                                                'out_of_place_wrong', 'in_place_wrong')})
        row['out_of_place_busbw_gbps'] = round(table['out_of_place_algbw_gbps'] * factor, 2) if factor else ''
    sm = [float(state['clocks.sm']) for state in record['gpu_state_after'] if state.get('clocks.sm', '').isdigit()]
    row['sm_clock_after_mhz_min'] = min(sm) if sm else ''
    return row


def median(values):
    return statistics.median(values) if values else ''


def interquartileRange(values):
    if len(values) < 2:
        return ''
    quartiles = statistics.quantiles(values, n=4, method='inclusive')
    return quartiles[2] - quartiles[0]


def configRows(rows):
    """Median and IQR over rounds for each configuration (ok runs, single or long runs)."""
    groups = {}
    for row in rows:
        if row['status'] != 'ok' or row['length'] == 'short' or 'out_of_place_algbw_gbps' not in row:
            continue
        key = (row['session'], row['block'], row['arm'], row['ctas_requested'], row['size'])
        groups.setdefault(key, []).append(row)
    output = []
    for (session, block, arm, ctas, size), members in sorted(groups.items(), key=lambda item: str(item[0])):
        algbw = [member['out_of_place_algbw_gbps'] for member in members]
        busbw = [member['out_of_place_busbw_gbps'] for member in members if member['out_of_place_busbw_gbps'] != '']
        output.append({
            'session': session, 'block': block, 'arm': arm, 'ctas_requested': ctas,
            'launched': members[0]['launched'], 'size': size, 'rounds': len(members),
            'algbw_median_gbps': round(median(algbw), 2), 'algbw_iqr_gbps': round(interquartileRange(algbw) or 0, 2),
            'busbw_median_gbps': round(median(busbw), 2) if busbw else '',
            'busbw_iqr_gbps': round(interquartileRange(busbw) or 0, 2) if busbw else '',
            'busbw_share_of_nominal': round(median(busbw) / NOMINAL_LINK_GBPS, 3) if busbw else '',
        })
    return output


def counterRows(records):
    """Long-minus-short accounting for every long/short pair of the same configuration and round."""
    pairs = {}
    for record in records:
        run = record['run']
        if run['length'] in ('long', 'short') and record['status'] == 'ok':
            key = (record['session'], run['block'], run['arm'], run['ctas'], run['size'], run['round'])
            pairs.setdefault(key, {})[run['length']] = record
    output = []
    for (session, block, arm, ctas, size, roundIndex), pair in sorted(pairs.items(), key=lambda item: str(item[0])):
        if set(pair) != {'long', 'short'}:
            continue
        long, short = pair['long'], pair['short']
        extraCalls = long['run']['iterations'] - short['run']['iterations']
        table = long['rows'][0]
        payloadBytes = 2 * extraCalls * table['size_bytes']
        seconds = extraCalls * (table['out_of_place_time_us'] + table['in_place_time_us']) * 1e-6
        deltas = {}
        for counterType in ('data', 'raw'):
            longDelta = counterDelta(long['links_before'], long['links_after'], counterType)
            shortDelta = counterDelta(short['links_before'], short['links_after'], counterType)
            deltas[counterType] = {key: longDelta[key] - shortDelta[key] for key in longDelta}
        for (gpu, direction) in sorted(deltas['data']):
            dataBytes = deltas['data'][(gpu, direction)]
            rawBytes = deltas['raw'][(gpu, direction)]
            output.append({
                'session': session, 'block': block, 'arm': arm, 'ctas_requested': ctas if ctas is not None else 'default',
                'size': size, 'round': roundIndex, 'gpu': gpu, 'direction': direction,
                'data_bytes_per_payload_byte': round(dataBytes / payloadBytes, 4),
                'raw_bytes_per_payload_byte': round(rawBytes / payloadBytes, 4),
                'raw_per_data_byte': round(rawBytes / dataBytes, 4) if dataBytes else '',
                'data_rate_gbps': round(dataBytes / seconds / 1e9, 1),
                'raw_link_rate_gbps': round(rawBytes / seconds / 1e9, 1),
                'ceiling_share': round(rawBytes / seconds / 1e9 / NOMINAL_LINK_GBPS, 3),
            })
    return output


def writeCsv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sessions', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    records = [record for session in arguments.sessions for record in loadRecords(session)]
    rows = [runRow(record) for record in records]
    arguments.output.mkdir(parents=True, exist_ok=True)
    writeCsv(arguments.output / 'runs.csv', rows)
    writeCsv(arguments.output / 'configs.csv', configRows(rows))
    writeCsv(arguments.output / 'counters.csv', counterRows(records))
    statusCounts = {}
    for row in rows:
        statusCounts[row['status']] = statusCounts.get(row['status'], 0) + 1
    print(f'{len(rows)} runs {statusCounts}; tables in {arguments.output}')


if __name__ == '__main__':
    main()
