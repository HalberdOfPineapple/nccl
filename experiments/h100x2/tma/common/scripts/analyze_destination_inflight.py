#!/usr/bin/env python3
"""TMA plan 01 E1.7b: TMA store writer on the destination at no limit, with more requests outstanding on
either side. Tables and figure.

Usage: analyze_destination_inflight.py RESULTS_DATE_DIRECTORY   (run with the plotting environment)

Unit of repetition: one run. Each mixed run alternates 6 idle-writer and 6 active transfers; the run's
copy change is median(active) / median(idle) - 1. Tables give the median of the 3 runs' changes with
min and max. Copy timers (slot wait, bulk completion wait) are medians over the copy CTAs per transfer,
then over the idle or active transfers of all runs of the cell. The writer-alone run does not depend
on the copy, so each writer setting has one alone run per round (copy off, same destination SMs).
Output: destination_inflight_summary.csv and figures/destination_inflight.png.
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
from run_tma_congest import INFLIGHT_COPIES, INFLIGHT_WRITERS, summarize  # noqa: E402

HEALTHY = '#1f77b4'
PERTURBED = '#ff7f0e'
MIXED = re.compile(r'mixed_w(?P<writer_ctas>\d+)k(?P<chunks>\d+)_c(?P<copy_ctas>\d+)s(?P<slots>\d+)_unlimited_R(?P<round>\d+)$')
ALONE = re.compile(r'alone_w(?P<writer_ctas>\d+)k(?P<chunks>\d+)_unlimited_R(?P<round>\d+)$')


def median_of(rows, metric):
    samples = [row[metric] for row in rows if metric in row]
    return statistics.median(samples) if samples else None


def quartile_range(values):
    quartiles = statistics.quantiles(sorted(values), n=4, method='inclusive')
    return quartiles[2] - quartiles[0]


def raw_records(raw, pattern):
    for path in sorted(raw.glob('*.json')):
        match = pattern.match(path.stem)
        if match:
            yield path, match


def summary(results):
    raw = results / 'destination_inflight' / 'raw'
    alone_rates = defaultdict(list)
    alone_evenness = defaultdict(list)
    for path, match in raw_records(raw, ALONE):
        _, rows = summarize(path)
        writer = (int(match['writer_ctas']), int(match['chunks']))
        alone_rates[writer].append(median_of(rows, 'writer_gbps'))
        alone_evenness[writer].append(median_of(rows, 'writer_slowest_to_fastest'))
    cells = defaultdict(list)
    for path, match in raw_records(raw, MIXED):
        _, rows = summarize(path)
        writer = (int(match['writer_ctas']), int(match['chunks']))
        copy = (int(match['copy_ctas']), int(match['slots']))
        cells[(writer, copy)].append(dict(idle=[row for row in rows if row['writer_idle']], active=[row for row in rows if not row['writer_idle']]))
    table = []
    for writer in INFLIGHT_WRITERS:
        for copy in INFLIGHT_COPIES:
            runs = cells.get((writer, copy), [])
            if not runs:
                continue
            changes = [median_of(run['active'], 'throughput_gbps') / median_of(run['idle'], 'throughput_gbps') - 1 for run in runs]
            idle = [row for run in runs for row in run['idle']]
            active = [row for run in runs for row in run['active']]
            idle_iqr = quartile_range([row['throughput_gbps'] for row in idle])
            during = statistics.median(median_of(run['active'], 'writer_gbps') for run in runs)
            alone = statistics.median(alone_rates[writer]) if alone_rates[writer] else None
            table.append(dict(
                writer=f'{writer[0]} CTAs x {writer[1]} chunks', copy=f'{copy[0]} CTAs x {copy[1]} slots', runs=len(runs),
                copy_idle_gbps=median_of(idle, 'throughput_gbps'), copy_active_gbps=median_of(active, 'throughput_gbps'),
                copy_change_percent=100 * statistics.median(changes), copy_change_min_percent=100 * min(changes), copy_change_max_percent=100 * max(changes),
                idle_iqr_gbps=idle_iqr, change_over_idle_iqr=(median_of(active, 'throughput_gbps') - median_of(idle, 'throughput_gbps')) / idle_iqr if idle_iqr else None,
                writer_alone_gbps=alone, writer_with_copy_gbps=during, writer_change_percent=100 * (during / alone - 1) if alone else None,
                writer_alone_slowest_to_fastest=statistics.median(alone_evenness[writer]) if alone_evenness[writer] else None,
                writer_with_copy_slowest_to_fastest=median_of(active, 'writer_slowest_to_fastest'),
                slot_wait_idle=median_of(idle, 'slot_wait_fraction'), slot_wait_active=median_of(active, 'slot_wait_fraction'),
                bulk_wait_us_idle=median_of(idle, 'bulk_wait_us'), bulk_wait_us_active=median_of(active, 'bulk_wait_us'),
                ack_wait_us_idle=median_of(idle, 'ack_wait_us'), ack_wait_us_active=median_of(active, 'ack_wait_us')))
    return table


def write_csv(path, table):
    columns = list(dict.fromkeys(key for row in table for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in table:
            writer.writerow({key: (f'{value:.4f}' if isinstance(value, float) else value) for key, value in row.items()})
    print(f'wrote {path}')


def plot(table, path):
    """Left: copy throughput, idle vs active, one group per writer setting, one bar pair per copy setting.
    Right: writer rate alone vs during the copy, per writer setting and copy setting."""
    writers = list(dict.fromkeys(row['writer'] for row in table))
    copies = list(dict.fromkeys(row['copy'] for row in table))
    figure, (copy_axis, writer_axis) = plt.subplots(1, 2, figsize=(13, 4.8))
    pair_width = 0.8 / len(copies)
    for writer_index, writer in enumerate(writers):
        for copy_index, copy in enumerate(copies):
            row = next((row for row in table if row['writer'] == writer and row['copy'] == copy), None)
            if row is None:
                continue
            left = writer_index - 0.4 + copy_index * pair_width
            copy_axis.bar(left + pair_width * 0.25, row['copy_idle_gbps'], width=pair_width * 0.45, color=HEALTHY, label='writer idle' if writer_index == copy_index == 0 else None)
            copy_axis.bar(left + pair_width * 0.72, row['copy_active_gbps'], width=pair_width * 0.45, color=PERTURBED, label='writer active' if writer_index == copy_index == 0 else None)
            writer_axis.bar(left + pair_width * 0.72, row['writer_with_copy_gbps'], width=pair_width * 0.45, color=PERTURBED, label='writer during the copy' if writer_index == copy_index == 0 else None)
            writer_axis.bar(left + pair_width * 0.25, row['writer_alone_gbps'] or 0, width=pair_width * 0.45, color=HEALTHY, label='writer alone (copy off)' if writer_index == copy_index == 0 else None)
    for axis in (copy_axis, writer_axis):
        axis.set_xticks(range(len(writers)))
        axis.set_xticklabels([writer.replace(' x ', ' × ') for writer in writers], fontsize=8)
        axis.set_xlabel('TMA store writer on the destination (7 GPCs other than G6), no rate limit')
        axis.legend(frameon=False, fontsize=8, loc='upper left' if axis is writer_axis else 'upper right')
    set_zero_based_y(copy_axis, zero_based_upper([row['copy_idle_gbps'] for row in table] + [row['copy_active_gbps'] for row in table]))
    set_zero_based_y(writer_axis, zero_based_upper([row['writer_alone_gbps'] or 0 for row in table] + [row['writer_with_copy_gbps'] for row in table]))
    copy_axis.set_ylabel('copy throughput, GB/s')
    writer_axis.set_ylabel('writer rate, GB/s')
    copy_axis.set_title('Copy; bars within a group: copy CTAs × slots (16×2, 16×3, 48×2)', fontsize=9)
    writer_axis.set_title('Destination writer; same grouping', fontsize=9)
    figure.suptitle('E1.7b: writer on the destination at no limit; copy 2048 MiB, 64 KiB chunks, 512 fill threads/CTA, G6; median over 3 runs', fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f'wrote {path}')


def main():
    results = Path(sys.argv[1]).resolve()
    (results / 'figures').mkdir(exist_ok=True)
    table = summary(results)
    write_csv(results / 'destination_inflight_summary.csv', table)
    plot(table, results / 'figures' / 'destination_inflight.png')


if __name__ == '__main__':
    main()
