#!/usr/bin/env python3
"""TMA plan 01 calibration tables and figures: the TMA push alone (E1.1) and the co-runners alone (E1.2).

Usage: analyze_tma_calibration.py RESULTS_DATE_DIRECTORY  (run with the plotting environment)

Unit of repetition: one run (one harness invocation). A run's value is the median of its retained
transfers (copy) or windows (co-runner). Tables give the median over runs with min-max; they never
pool transfers across runs. Settings are read from each record's keys, not from file names.
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'thread_store/common/scripts'))
from plot_style import set_zero_based_y, zero_based_upper  # noqa: E402
from run_tma_congest import PEAK_GBPS, summarize  # noqa: E402

KIND_NAMES = {0: 'TMA store writer', 1: 'TMA read co-runner', 2: 'thread-store writer'}


def run_values(path):
    """One run: its settings and the medians of its transfers or windows."""
    data, rows = summarize(path)
    pair = f"{data['source_gpu']}->{data['destination_gpu']}"
    value = dict(pair=pair, copy_ctas=data['copy_ctas'], copy_threads=data['copy_threads'], chunk_kib=data['copy_chunk_bytes'] // 1024,
                 slots=data['copy_chunks_in_flight'], generate=data['copy_generate'], writer_kind=data['writer_kind'], writer_ctas=data['writer_ctas'],
                 chunks=data['chunks_in_flight'], colocated=data['writer_colocated'], target_gbps=data['target_gbps'], communication=data['communication'])
    for metric in ('throughput_gbps', 'issue_us', 'bulk_wait_us', 'ordering_us', 'ack_wait_us', 'slot_wait_fraction', 'fill_fraction', 'writer_gbps', 'writer_slowest_to_fastest'):
        samples = [row[metric] for row in rows if metric in row]
        if samples:
            value[metric] = statistics.median(samples)
    return value


def load(group_directory):
    return [run_values(path) for path in sorted((group_directory / 'raw').glob('*.json')) if not path.name.endswith(('.before.json', '.after.json', '.telemetry.json'))]


def aggregate(runs, keys, metrics):
    """Median over runs of each metric, with min and max, per setting tuple."""
    grouped = defaultdict(list)
    for run in runs:
        grouped[tuple(run[key] for key in keys)].append(run)
    table = []
    for setting in sorted(grouped):
        members = grouped[setting]
        row = dict(zip(keys, setting), runs=len(members), pairs=' '.join(sorted({member['pair'] for member in members})))
        for metric in metrics:
            samples = [member[metric] for member in members if metric in member]
            if samples:
                row[metric] = statistics.median(samples)
                row[f'{metric}_min'] = min(samples)
                row[f'{metric}_max'] = max(samples)
        table.append(row)
    return table


def write_csv(path, table):
    if not table:
        return
    columns = list(dict.fromkeys(key for row in table for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in table:
            writer.writerow({key: (f'{value:.4f}' if isinstance(value, float) else value) for key, value in row.items()})
    print(f'wrote {path}')


def copy_tables(results):
    runs = [run for run in load(results / 'copy_alone')]
    metrics = ['throughput_gbps', 'issue_us', 'bulk_wait_us', 'ordering_us', 'ack_wait_us', 'slot_wait_fraction', 'fill_fraction']
    table = aggregate(runs, ['copy_ctas', 'copy_threads', 'chunk_kib', 'slots', 'generate'], metrics)
    write_csv(results / 'copy_sweep_summary.csv', table)
    return table


def plot_copy_sweep(table, figures):
    """CTA scaling at 512 fill threads/CTA and 64 KiB chunks, one line per slot count; slot wait beside it."""
    sweep = [row for row in table if row['copy_threads'] == 512 and row['chunk_kib'] == 64 and row['generate'] == 1]
    if not sweep:
        return
    figure, (throughput_axis, wait_axis) = plt.subplots(1, 2, figsize=(11, 4.2))
    for slots in sorted({row['slots'] for row in sweep}):
        points = sorted((row for row in sweep if row['slots'] == slots), key=lambda row: row['copy_ctas'])
        ctas = [row['copy_ctas'] for row in points]
        throughput_axis.errorbar(ctas, [row['throughput_gbps'] for row in points],
                                 yerr=[[row['throughput_gbps'] - row['throughput_gbps_min'] for row in points], [row['throughput_gbps_max'] - row['throughput_gbps'] for row in points]],
                                 marker='o', markersize=4, linewidth=1, capsize=2, label=f'{slots} slot(s) × 64 KiB')
        wait_axis.plot(ctas, [100 * row['slot_wait_fraction'] for row in points], marker='o', markersize=4, linewidth=1, label=f'{slots} slot(s) × 64 KiB')
    for axis in (throughput_axis, wait_axis):
        axis.set_xscale('log', base=2)
        axis.set_xticks(sorted({row['copy_ctas'] for row in sweep}))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_xlabel('copy CTAs (512 fill threads/CTA, packed from G6)')
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.3, linewidth=0.5)
    set_zero_based_y(throughput_axis, zero_based_upper([row['throughput_gbps_max'] for row in sweep]))
    throughput_axis.set_ylabel('copy throughput, GB/s')
    throughput_axis.set_title('TMA push alone, 2048 MiB')
    set_zero_based_y(wait_axis, 100)
    wait_axis.set_ylabel('slot wait, % of thread 0 issue loop')
    wait_axis.set_title('Time thread 0 waits for a free staging slot')
    figure.tight_layout()
    figure.savefig(figures / 'copy_sweep.png', dpi=150)
    plt.close(figure)
    print(f"wrote {figures / 'copy_sweep.png'}")


def co_runner_tables(results):
    runs = [run for run in load(results / 'co_runner_alone')]
    table = aggregate(runs, ['writer_kind', 'writer_ctas', 'chunks', 'colocated', 'copy_ctas', 'target_gbps'], ['writer_gbps', 'writer_slowest_to_fastest'])
    for row in table:
        row['percent_of_peak'] = 100 * row['writer_gbps'] / PEAK_GBPS
        if row['target_gbps']:
            row['achieved_over_target'] = row['writer_gbps'] / row['target_gbps']
    write_csv(results / 'co_runner_alone_summary.csv', table)
    return table


def plot_writer_sweep(table, figures):
    sweep = [row for row in table if row['writer_kind'] == 0 and not row['colocated'] and row['target_gbps'] == 0]
    if not sweep:
        return
    figure, axis = plt.subplots(figsize=(6, 4.2))
    for chunks in sorted({row['chunks'] for row in sweep}):
        points = sorted((row for row in sweep if row['chunks'] == chunks), key=lambda row: row['writer_ctas'])
        axis.errorbar([row['writer_ctas'] for row in points], [row['writer_gbps'] for row in points],
                      yerr=[[row['writer_gbps'] - row['writer_gbps_min'] for row in points], [row['writer_gbps_max'] - row['writer_gbps'] for row in points]],
                      marker='o', markersize=4, linewidth=1, capsize=2, label=f'{chunks} × 64 KiB in flight per CTA')
    axis.axhline(PEAK_GBPS, color='grey', linewidth=0.8, linestyle='--', label='HBM peak 3352 GB/s')
    set_zero_based_y(axis, zero_based_upper([PEAK_GBPS] + [row['writer_gbps_max'] for row in sweep]))
    axis.set_xlabel('TMA store writer CTAs (spread over the 7 GPCs other than G6)')
    axis.set_ylabel('requested write rate, GB/s')
    axis.set_title('TMA store writer alone, no rate limit')
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.3, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(figures / 'writer_sweep.png', dpi=150)
    plt.close(figure)
    print(f"wrote {figures / 'writer_sweep.png'}")


def plot_rate_tracking(table, figures):
    """Achieved against target for every co-runner calibrated at the 20-80% levels."""
    limited = [row for row in table if row['target_gbps']]
    if not limited:
        return
    figure, axis = plt.subplots(figsize=(6, 4.2))
    for (kind, ctas, chunks) in sorted({(row['writer_kind'], row['writer_ctas'], row['chunks']) for row in limited}):
        points = sorted((row for row in limited if (row['writer_kind'], row['writer_ctas'], row['chunks']) == (kind, ctas, chunks)), key=lambda row: row['target_gbps'])
        axis.plot([row['target_gbps'] for row in points], [row['writer_gbps'] for row in points], marker='o', markersize=4, linewidth=1,
                  label=f'{KIND_NAMES[kind]}: {ctas} CTAs × {chunks} × 64 KiB')
    upper = zero_based_upper([row['target_gbps'] for row in limited] + [row['writer_gbps'] for row in limited])
    axis.plot([0, upper], [0, upper], color='grey', linewidth=0.8, linestyle='--', label='achieved = target')
    axis.set_xlim(0, upper)
    set_zero_based_y(axis, upper)
    axis.set_xlabel('target rate, GB/s')
    axis.set_ylabel('achieved rate, GB/s')
    axis.set_title('Co-runners alone: achieved against target')
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.3, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(figures / 'rate_tracking.png', dpi=150)
    plt.close(figure)
    print(f"wrote {figures / 'rate_tracking.png'}")


def main():
    results = Path(sys.argv[1]).resolve()
    figures = results / 'figures'
    figures.mkdir(exist_ok=True)
    if (results / 'copy_alone').exists():
        plot_copy_sweep(copy_tables(results), figures)
    if (results / 'co_runner_alone').exists():
        table = co_runner_tables(results)
        plot_writer_sweep(table, figures)
        plot_rate_tracking(table, figures)


if __name__ == '__main__':
    main()
