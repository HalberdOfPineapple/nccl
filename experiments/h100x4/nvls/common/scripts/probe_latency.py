#!/usr/bin/env python3
"""Probe latency (R6) and bytes in flight (R7) per run, for RCA plan 01 E1.8.

Each run has one probe thread per GPU that times single outstanding 16-byte operations; the harness
writes per-window percentiles under the window key "probe". Background role rates (N1) come from the
same windows, so R7 = role rate x probe median latency is a Little's-law estimate for the background.

Usage: probe_latency.py <raw directory> <output csv>
Column names are an interface for the figure scripts.
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mm_summary import window_role_rates  # noqa: E402

RUN_NAME = re.compile(r'^e1_8_probe_(?P<probe>[a-z_]+?)_background_(?P<background>[a-z_]+)_R(?P<round>\d+)$')


def run_rows(path):
    match = RUN_NAME.match(path.stem)
    record = json.loads(path.read_text())
    rows = []
    for rank in record['ranks']:
        kept = rank['windows'][record['warmup']:]
        probes = [window['probe'] for window in kept if window['probe']['samples'] > 0]
        role_rates = {}
        for window in kept:
            for role, (rate, _start, _end) in window_role_rates(window, record['chunk_bytes']).items():
                role_rates.setdefault(role, []).append(rate)
        background_rate = sum(statistics.median(values) for values in role_rates.values()) if role_rates else 0.0
        median_ns = statistics.median(probe['median_ns'] for probe in probes)
        rows.append({
            'run': path.stem,
            'probe': match['probe'],
            'background': match['background'],
            'round': int(match['round']),
            'gpu': rank['gpu'],
            'background_role': record['role_a'] if record['ctas_a'] else 'none',
            'background_ctas': record['ctas_a'],
            'threads_per_cta': record['threads'],
            'kept_windows': len(probes),
            'samples_per_window': statistics.median(probe['samples'] for probe in probes),
            'probe_min_ns': statistics.median(probe['min_ns'] for probe in probes),
            'probe_median_ns': median_ns,  # R6
            'probe_p90_ns': statistics.median(probe['p90_ns'] for probe in probes),  # R6
            'probe_p99_ns': statistics.median(probe['p99_ns'] for probe in probes),
            'background_rate_gbps': background_rate,  # N1, role bytes per ns of this GPU's own part
            'background_bytes_in_flight_kib': background_rate * median_ns / 1024,  # R7: GB/s x ns = bytes
            'mismatches': sum(window['mismatches'] for window in rank['windows']),
        })
    return rows


def main():
    raw_directory = Path(sys.argv[1])
    output = Path(sys.argv[2])
    rows = []
    for path in sorted(raw_directory.glob('e1_8_*_R*.json')):
        if not re.search(r'\.(before|after|telemetry)$', path.stem):
            rows.extend(run_rows(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row['probe'], row['background']), []).append(row)
    for (probe, background), members in sorted(groups.items()):
        print(f"probe={probe:<9} background={background:<13} median={statistics.median(m['probe_median_ns'] for m in members):7.0f} ns p90={statistics.median(m['probe_p90_ns'] for m in members):7.0f} ns rate={statistics.median(m['background_rate_gbps'] for m in members):6.1f} GB/s in_flight={statistics.median(m['background_bytes_in_flight_kib'] for m in members):7.0f} KiB")


if __name__ == '__main__':
    main()
