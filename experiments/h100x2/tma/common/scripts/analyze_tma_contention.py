#!/usr/bin/env python3
"""TMA plan 01 tables and figures for the TMA push next to a co-runner (E1.3, E1.5, E1.6, E1.7, E1.8).

Usage: analyze_tma_contention.py RESULTS_DATE_DIRECTORY GROUP [GROUP ...]   (plotting environment)
GROUP is a results sub-directory: contention, colocated, direction, destination, mechanism.

Unit of repetition: one run. Each mixed run alternates 6 idle-co-runner and 6 active transfers; the
run's copy change is median(active) / median(idle) - 1 over its own transfers, so each run carries
its own reference. Tables give the median of the per-run changes and their min-max over the runs.
Reference variability: interquartile range (IQR) of the idle transfers of all mixed runs of one
kind; "change / idle IQR" divides the change of the pooled medians in GB/s by it. Transfers are
nested in runs, so this is a guide to size, not a test. Co-runner: its rate during the copy (active
transfers) against the co-runner alone on the same SMs, in windows matched to its active time.
Output: GROUP_summary.csv (keys read by run_tma_congest.matched_levels: placement, level,
writer_with_copy_gbps) and figures/GROUP.png.
"""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'thread_store/common/scripts'))
from plot_style import set_zero_based_y, zero_based_upper  # noqa: E402
from run_tma_congest import summarize  # noqa: E402

HEALTHY = '#1f77b4'      # idle co-runner (project colour for healthy)
PERTURBED = '#ff7f0e'    # active co-runner (project colour for perturbed)
LEVEL_ORDER = ['p20', 'p40', 'p60', 'p80', 'c1', 'c5', 'c10', 'c25', 'c50', 'c75', 'unlimited']
LEVEL_NAMES = {'p20': '20%', 'p40': '40%', 'p60': '60%', 'p80': '80%', 'c1': '1% of ceiling', 'c5': '5% of ceiling', 'c10': '10% of ceiling', 'c25': '25% of ceiling', 'c50': '50% of ceiling', 'c75': '75% of ceiling', 'unlimited': 'no limit'}
KIND_TITLES = {
    'separate': 'TMA store writer: 64 CTAs × 1 chunk, other 7 GPCs',
    'same_sm': 'TMA store writer: 16 CTAs × 1 chunk, on the copy\'s 16 SMs (G6)',
    'other_sms': 'TMA store writer: 16 CTAs × 1 chunk, other 7 GPCs',
    'writes': 'TMA store writer: 64 CTAs × 1 chunk, other 7 GPCs',
    'reads': 'TMA read co-runner: 48 CTAs × 3 chunks, other 7 GPCs',
    'source': 'TMA store writer on the source GPU (other 7 GPCs)',
    'destination': 'TMA store writer on the destination GPU (other 7 GPCs)',
    'tma': 'TMA store writer: 64 CTAs × 1 chunk',
    'same_sm_tma': 'TMA store writer: 1 chunk, last 2 warps of each of the 16 copy CTAs',
    'same_sm_thread': 'thread-store writer: 64 threads (2 warps) of each of the 16 copy CTAs',
    'thread': 'thread-store writer: 64 CTAs × 1024 threads/CTA',
}
COPY_LABEL = 'copy: 16 CTAs × 512 fill threads/CTA, 2 × 64 KiB slots, G6, 2048 MiB'
MIXED = re.compile(r'mixed_(?P<kind>.+)_(?P<level>p\d+|c\d+|unlimited)_R(?P<round>\d+)$')


def quartile_range(values):
    if len(values) < 2:
        return 0.0
    quartiles = statistics.quantiles(sorted(values), n=4, method='inclusive')
    return quartiles[2] - quartiles[0]


def median_of(rows, metric):
    samples = [row[metric] for row in rows if metric in row]
    return statistics.median(samples) if samples else None


def runs_of(raw):
    """Mixed runs grouped by (kind, level), each with its idle and active transfers and its co-runner-alone partner."""
    grouped = defaultdict(list)
    for path in sorted(raw.glob('mixed_*.json')):
        if path.name.endswith(('.before.json', '.after.json', '.telemetry.json')):
            continue
        match = MIXED.match(path.stem)
        _, rows = summarize(path)
        idle = [row for row in rows if row['writer_idle']]
        active = [row for row in rows if not row['writer_idle']]
        alone_path = raw / f"alone_{match['kind']}_{match['level']}_R{match['round']}.json"
        alone_rate = None
        if alone_path.exists():
            _, alone_rows = summarize(alone_path)
            alone_rate = median_of(alone_rows, 'writer_gbps')
        grouped[(match['kind'], match['level'])].append(dict(idle=idle, active=active, alone_rate=alone_rate))
    return grouped


def summary_rows(grouped):
    idle_by_kind = defaultdict(list)
    for (kind, _), runs in grouped.items():
        for run in runs:
            idle_by_kind[kind] += [row['throughput_gbps'] for row in run['idle']]
    table = []
    for (kind, level) in sorted(grouped, key=lambda key: (key[0], LEVEL_ORDER.index(key[1]))):
        runs = grouped[(kind, level)]
        changes = [median_of(run['active'], 'throughput_gbps') / median_of(run['idle'], 'throughput_gbps') - 1 for run in runs]
        idle_pooled = [row for run in runs for row in run['idle']]
        active_pooled = [row for run in runs for row in run['active']]
        idle_iqr = quartile_range(idle_by_kind[kind])
        idle_median = median_of(idle_pooled, 'throughput_gbps')
        active_median = median_of(active_pooled, 'throughput_gbps')
        with_copy = [median_of(run['active'], 'writer_gbps') for run in runs]
        alone = [run['alone_rate'] for run in runs if run['alone_rate'] is not None]
        row = dict(placement=kind, level=level, runs=len(runs),
                   copy_idle_gbps=idle_median, copy_active_gbps=active_median,
                   copy_change_percent=100 * statistics.median(changes), copy_change_min_percent=100 * min(changes), copy_change_max_percent=100 * max(changes),
                   idle_iqr_gbps=idle_iqr, change_over_idle_iqr=(active_median - idle_median) / idle_iqr if idle_iqr else None,
                   writer_with_copy_gbps=statistics.median(with_copy),
                   writer_alone_gbps=statistics.median(alone) if alone else None)
        if alone:
            row['writer_change_percent'] = 100 * (row['writer_with_copy_gbps'] / row['writer_alone_gbps'] - 1)
        for metric in ('slot_wait_fraction', 'fill_fraction', 'issue_us', 'bulk_wait_us', 'ordering_us', 'ack_wait_us'):
            row[f'{metric}_idle'] = median_of(idle_pooled, metric)
            row[f'{metric}_active'] = median_of(active_pooled, metric)
        table.append(row)
    return table


def write_csv(path, table):
    columns = list(dict.fromkeys(key for row in table for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in table:
            writer.writerow({key: (f'{value:.4f}' if isinstance(value, float) else value) for key, value in row.items()})
    print(f'wrote {path}')


def plot(table, path, group):
    """Rows: co-runner kinds. Left: copy throughput, idle vs active halves. Right: co-runner rate, alone vs during the copy."""
    kinds = list(dict.fromkeys(row['placement'] for row in table))
    figure, axes = plt.subplots(len(kinds), 2, figsize=(11, 3.6 * len(kinds)), squeeze=False)
    copy_upper = zero_based_upper([row['copy_idle_gbps'] for row in table] + [row['copy_active_gbps'] for row in table])
    rate_upper = zero_based_upper([value for row in table for value in (row['writer_with_copy_gbps'], row['writer_alone_gbps']) if value])
    for index, kind in enumerate(kinds):
        rows = [row for row in table if row['placement'] == kind]
        positions = range(len(rows))
        names = [LEVEL_NAMES[row['level']] for row in rows]
        copy_axis, rate_axis = axes[index]
        copy_axis.bar([position - 0.2 for position in positions], [row['copy_idle_gbps'] for row in rows], width=0.4, color=HEALTHY, label='co-runner idle')
        copy_axis.bar([position + 0.2 for position in positions], [row['copy_active_gbps'] for row in rows], width=0.4, color=PERTURBED, label='co-runner active')
        rate_axis.bar([position - 0.2 for position in positions], [row['writer_alone_gbps'] or 0 for row in rows], width=0.4, color=HEALTHY, label='co-runner alone (copy off)')
        rate_axis.bar([position + 0.2 for position in positions], [row['writer_with_copy_gbps'] for row in rows], width=0.4, color=PERTURBED, label='co-runner during the copy')
        set_zero_based_y(copy_axis, copy_upper)
        set_zero_based_y(rate_axis, rate_upper)
        copy_axis.set_ylabel('copy throughput, GB/s')
        rate_axis.set_ylabel('co-runner rate, GB/s')
        copy_axis.set_title(f'Copy next to {KIND_TITLES.get(kind, kind)}', fontsize=9)
        rate_axis.set_title(f'{KIND_TITLES.get(kind, kind)}', fontsize=9)
        # Legends go where bars leave space: above the copy bars, at the low-rate end of the rate panel.
        for axis, location in ((copy_axis, 'upper right'), (rate_axis, 'upper left')):
            axis.set_xticks(list(positions))
            axis.set_xticklabels(names, fontsize=8)
            axis.set_xlabel('co-runner target rate (% of HBM peak 3352 GB/s unless marked)')
            axis.legend(frameon=False, fontsize=8, loc=location)
    figure.suptitle(f'{group}: {COPY_LABEL}; median over runs', fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f'wrote {path}')


def main():
    results = Path(sys.argv[1]).resolve()
    (results / 'figures').mkdir(exist_ok=True)
    for group in sys.argv[2:]:
        table = summary_rows(runs_of(results / group / 'raw'))
        write_csv(results / f'{group}_summary.csv', table)
        plot(table, results / 'figures' / f'{group}.png', group)


if __name__ == '__main__':
    main()
