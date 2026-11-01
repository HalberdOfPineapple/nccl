#!/usr/bin/env python3
"""Plan 08: turn ncu tables of peer_store_path into one row per phase of each kept transfer.

Usage as a tool: peer_store_counters.py RAW_DIRECTORY [LABEL_FILTER]; prints medians per label and phase.
Counters are per launch and GPU-wide on the profiled GPU. Only the destination's launch sequence has
every phase; on the source GPU only phaseMain exists.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

SECTOR_BYTES = 32
KERNEL_COLUMN = 4
METRIC_COLUMN = 12
VALUE_COLUMN = 14


def launches_of_table(path):
    """Launches in order: (phase name, {metric: value})."""
    launches = {}
    for row in csv.reader(path.open()):
        if len(row) > VALUE_COLUMN and row[KERNEL_COLUMN].startswith('phase'):
            phase = row[KERNEL_COLUMN].split('(')[0]
            entry = launches.setdefault(int(row[0]), (phase, {}))
            entry[1][row[METRIC_COLUMN]] = float(row[VALUE_COLUMN].replace(',', '')) if row[VALUE_COLUMN] != 'n/a' else None
    return [launches[launch] for launch in sorted(launches)]


def transfers_of_table(path, kept_transfers):
    """Kept transfers only: a list of [(phase name with its order, metrics), ...].

    A transfer on the destination starts at phaseClean. The two quiet phases are told apart by their
    place: phaseQuietBefore comes before phaseMain and phaseQuietAfter comes after it.
    """
    launches = launches_of_table(path)
    starts_with_clean = any(phase == 'phaseClean' for phase, _ in launches)
    transfers = []
    for phase, metrics in launches:
        if phase == ('phaseClean' if starts_with_clean else 'phaseMain') or not transfers:
            transfers.append([])
        if phase == 'phaseQuiet':
            seen_main = any(name == 'phaseMain' for name, _ in transfers[-1])
            phase = 'phaseQuietAfter' if seen_main else 'phaseQuietBefore'
        transfers[-1].append((phase, metrics))
    return transfers[-kept_transfers:]


def run_rows(json_path):
    """One dict per phase of each kept transfer of one run."""
    record = json.loads(json_path.read_text())
    kept = len(record['ranks'][1]['transfers'])
    rows = []
    for transfer_index, transfer in enumerate(transfers_of_table(json_path.with_suffix('.ncu.csv'), kept)):
        for phase, metrics in transfer:
            rows.append({'run': json_path.stem, 'transfer': transfer_index, 'phase': phase, **metrics})
    return record, rows


def median_by_phase(rows):
    phases = {}
    for row in rows:
        phases.setdefault(row['phase'], []).append(row)
    medians = {}
    for phase, phase_rows in phases.items():
        metric_names = [name for name in phase_rows[0] if name not in ('run', 'transfer', 'phase')]
        medians[phase] = {name: statistics.median(row[name] for row in phase_rows if row.get(name) is not None) for name in metric_names if any(row.get(name) is not None for row in phase_rows)}
    return medians


def label_of_run(stem):
    return stem.rsplit('_R', 1)[0]


def rows_by_label(raw_directory):
    grouped = {}
    records = {}
    for json_path in sorted(raw_directory.glob('*.json')):
        if json_path.name.endswith(('.before.json', '.after.json', '.telemetry.json')) or not json_path.with_suffix('.ncu.csv').exists():
            continue
        record, rows = run_rows(json_path)
        label = label_of_run(json_path.stem)
        grouped.setdefault(label, []).extend(rows)
        records[label] = record
    return grouped, records


def main():
    raw_directory = Path(sys.argv[1])
    label_filter = sys.argv[2] if len(sys.argv) > 2 else ''
    grouped, records = rows_by_label(raw_directory)
    for label, rows in grouped.items():
        if label_filter not in label:
            continue
        stored_gib = records[label]['stored_bytes'] / 2**30
        print(f"\n{label}: stored {stored_gib:.3f} GiB; runs {len(set(row['run'] for row in rows))}")
        print(f"{'phase':<18}{'time us':>10}{'NVL rx user GiB':>16}{'L2 write GiB':>13}{'w hit':>10}{'w miss':>10}{'L2 read GiB':>12}{'r hit':>10}{'r miss':>10}{'gpc GiB':>9}{'fabric GiB':>11}{'HBM wr GiB':>11}{'HBM rd GiB':>11}")
        for phase, values in median_by_phase(rows).items():
            def gib_of_sectors(name):
                return values.get(name, 0) * SECTOR_BYTES / 2**30
            print(f"{phase:<18}{values.get('gpu__time_duration.sum', 0) / 1e3:>10.0f}{values.get('nvlrx__bytes_packet_request_data_user.sum', 0) / 2**30:>16.4f}"
                  f"{gib_of_sectors('lts__t_sectors_op_write.sum'):>13.4f}{gib_of_sectors('lts__t_sectors_op_write_lookup_hit.sum'):>10.3f}{gib_of_sectors('lts__t_sectors_op_write_lookup_miss.sum'):>10.3f}"
                  f"{gib_of_sectors('lts__t_sectors_op_read.sum'):>12.4f}{gib_of_sectors('lts__t_sectors_op_read_lookup_hit.sum'):>10.3f}{gib_of_sectors('lts__t_sectors_op_read_lookup_miss.sum'):>10.3f}"
                  f"{gib_of_sectors('lts__t_sectors_srcnode_gpc.sum'):>9.3f}{gib_of_sectors('lts__t_sectors_srcunit_ltcfabric.sum'):>11.3f}"
                  f"{values.get('dram__bytes_write.sum', 0) / 2**30:>11.4f}{values.get('dram__bytes_read.sum', 0) / 2**30:>11.4f}")


if __name__ == '__main__':
    main()
