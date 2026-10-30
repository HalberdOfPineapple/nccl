#!/usr/bin/env python3
"""Tables and figures for the HBM read-rate calibration. Stages without data are skipped.

Unit of repetition: one invocation (five 20 ms windows; their median is the invocation value).
Each setting has five invocations; tables give the median of those five and their min–max.
"""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from run_hbm_read_rate import (RESULTS, SELECTION, PEAK_GBPS, ROUNDS, SCREEN_CTAS, CTA_COUNTS,
                               TARGET_PERCENTS, chunk_grid, window_rate)

FIGURES = RESULTS / 'figures'
# Blue and orange mean healthy and perturbed elsewhere in this project; settings here are neither.
COLORS = ['#3a923a', '#9372b2', '#c03d3e', '#845b53', '#2a9d9a', '#b8860b']


def invocation(label):
    path = RESULTS / 'raw' / f'{label}.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    rates = [window_rate(data, window) for window in data['windows']]
    # Spread between the slowest and fastest pressure CTA, per window, as a balance check.
    balance = []
    for window in data['windows']:
        per_cta = [record['chunks_read'] / (record['end_ns'] - record['start_ns']) for record in window if record['role'] == 1]
        balance.append(min(per_cta) / max(per_cta))
    control = [record for window in data['windows'] for record in window if record['role'] == 0]
    return dict(rate=statistics.median(rates), balance=min(balance),
                timer_step_min_ns=min(record['timer_step_min_ns'] for record in control),
                timer_step_max_ns=max(record['timer_step_max_ns'] for record in control),
                registers=data['registers'], shared_bytes=data['shared_bytes'])


def summarize(prefix, **fields):
    runs = [invocation(f'{prefix}_R{repetition}') for repetition in range(1, ROUNDS + 1)]
    if any(run is None for run in runs):
        return None
    rates = [run['rate'] for run in runs]
    median = statistics.median(rates)
    return dict(**fields, rate_gbps=median, rate_min_gbps=min(rates), rate_max_gbps=max(rates),
                percent_of_peak=100 * median / PEAK_GBPS, percent_min=100 * min(rates) / PEAK_GBPS, percent_max=100 * max(rates) / PEAK_GBPS,
                slowest_to_fastest_cta=min(run['balance'] for run in runs),
                timer_step_min_ns=min(run['timer_step_min_ns'] for run in runs), timer_step_max_ns=max(run['timer_step_max_ns'] for run in runs),
                shared_bytes=runs[0]['shared_bytes'])


def write_csv(path, rows):
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def error_bars(rows):
    return [[row['percent_of_peak'] - row['percent_min'] for row in rows], [row['percent_max'] - row['percent_of_peak'] for row in rows]]


def style(axis):
    axis.set_ylim(0, 100)
    axis.yaxis.set_major_locator(MultipleLocator(20))
    axis.grid(axis='y', linewidth=.4, alpha=.5)
    axis.spines[['top', 'right']].set_visible(False)


def chunk_stage():
    rows = [summarize(f'chunks_c{SCREEN_CTAS}_k{size}_n{count}', pressure_ctas=SCREEN_CTAS, chunk_kib=size, chunks_in_flight=count, in_flight_kib=size * count)
            for size, count in chunk_grid()]
    if any(row is None for row in rows):
        return None
    for row in rows:
        row['rate_per_cta_gbps'] = row['rate_gbps'] / SCREEN_CTAS
    figure, axis = plt.subplots(figsize=(7.5, 4.5), layout='constrained')
    for color, size in zip(COLORS, sorted({row['chunk_kib'] for row in rows})):
        series = [row for row in rows if row['chunk_kib'] == size]
        axis.errorbar([row['in_flight_kib'] for row in series], [row['percent_of_peak'] for row in series], yerr=error_bars(series),
                      marker='o', markersize=3, linewidth=1, capsize=2, color=color, label=f'{size} KiB per chunk')
    axis.axvline(114, color='grey', linestyle=':', linewidth=1)
    axis.text(116, 92, 'shared memory the fused\nharness reserves today (114 KiB)', fontsize=7, color='grey', va='top')
    style(axis)
    axis.set(xlabel='Bytes in flight per pressure CTA (KiB) = chunk size × chunks in flight', ylabel='Read rate (% of HBM peak 3352 GB/s)',
             title=f'Bytes in flight per CTA, with {SCREEN_CTAS} pressure CTAs × 1024 threads/CTA (1 copy thread each)')
    axis.title.set_fontsize(9)
    axis.legend(fontsize=7, title='Line = one chunk size;\npoints = 1, 2, 3, 4, 6 chunks', title_fontsize=7)
    figure.savefig(FIGURES / 'bytes_in_flight.png', dpi=150)
    plt.close(figure)
    return rows


def cta_stage(selection):
    rows = []
    for name, setting in selection.get('per_cta_settings', {}).items():
        for ctas in CTA_COUNTS:
            row = summarize(f'ctas_{name}_c{ctas}', setting=name, pressure_ctas=ctas, chunk_kib=setting['chunk_kib'], chunks_in_flight=setting['chunks_in_flight'])
            if row is None:
                return None
            rows.append(row)
    if not rows:
        return None
    figure, axis = plt.subplots(figsize=(7.5, 4.5), layout='constrained')
    for color, (name, setting) in zip(COLORS, selection['per_cta_settings'].items()):
        series = [row for row in rows if row['setting'] == name]
        axis.errorbar([row['pressure_ctas'] for row in series], [row['percent_of_peak'] for row in series], yerr=error_bars(series),
                      marker='o', markersize=3, linewidth=1, capsize=2, color=color,
                      label=f"{setting['chunks_in_flight']} × {setting['chunk_kib']} KiB in flight per CTA ({setting['label']})")
    style(axis)
    axis.set(xlabel='Number of pressure CTAs (one per SM; 1024 threads/CTA, 1 copy thread)', ylabel='Read rate (% of HBM peak 3352 GB/s)',
             title='Number of pressure CTAs, no rate limit')
    axis.legend(fontsize=7)
    figure.savefig(FIGURES / 'pressure_ctas.png', dpi=150)
    plt.close(figure)
    return rows


def rate_stage(selection):
    rows = []
    for name, setting in selection.get('rate_settings', {}).items():
        for percent in TARGET_PERCENTS + [0]:
            row = summarize(f'rate_{name}_p{percent}', setting=name, pressure_ctas=setting['pressure_ctas'], chunk_kib=setting['chunk_kib'],
                            chunks_in_flight=setting['chunks_in_flight'], target_percent=percent or None, target_gbps=round(PEAK_GBPS * percent / 100) or None)
            if row is None:
                return None
            rows.append(row)
    if not rows:
        return None
    figure, axis = plt.subplots(figsize=(7.5, 5), layout='constrained')
    axis.plot([0, 100], [0, 100], color='grey', linewidth=.8, linestyle='--', label='achieved = target')
    # The settings overlap exactly below 90%; large hollow circles keep the first one visible.
    markers = [dict(marker='o', markersize=8, markerfacecolor='none'), dict(marker='s', markersize=3)]
    for color, marker, (name, setting) in zip(COLORS, markers, selection['rate_settings'].items()):
        series = [row for row in rows if row['setting'] == name and row['target_percent']]
        unlimited = next(row for row in rows if row['setting'] == name and not row['target_percent'])
        label = f"{setting['pressure_ctas']} CTAs × {setting['chunks_in_flight']} × {setting['chunk_kib']} KiB in flight ({setting['label']})"
        axis.errorbar([row['target_percent'] for row in series], [row['percent_of_peak'] for row in series], yerr=error_bars(series),
                      linewidth=1, capsize=2, color=color, label=label, **marker)
        axis.axhline(unlimited['percent_of_peak'], color=color, linewidth=.8, linestyle=':')
    axis.set_xlim(0, 105)
    axis.xaxis.set_major_locator(MultipleLocator(10))
    style(axis)
    axis.set(xlabel='Target read rate, the knob (% of HBM peak 3352 GB/s)', ylabel='Achieved read rate (% of HBM peak 3352 GB/s)',
             title='Target read rate versus achieved read rate\nDotted line: same CTAs with no rate limit. Points: median of 5 runs; bars: min–max')
    axis.title.set_fontsize(9)
    axis.legend(fontsize=7, loc='upper left')
    figure.savefig(FIGURES / 'target_rate.png', dpi=150)
    plt.close(figure)
    return rows


def check_records():
    intervals = []
    for path in sorted((RESULTS / 'raw').glob('*.telemetry.json')):
        telemetry = json.loads(path.read_text())
        intervals.append((telemetry['host_start_ns'], telemetry['host_end_ns']))
    intervals.sort()
    assert all(previous[1] <= following[0] for previous, following in zip(intervals, intervals[1:])), 'Overlapping invocations'
    clocks = set()
    for path in (RESULTS / 'raw').glob('*.telemetry.json'):
        for sample in json.loads(path.read_text())['samples']:
            if sample['returncode'] == 0 and sample['stdout'].strip():
                clocks.add(int(sample['stdout'].split(',')[3]))
    return dict(invocations=len(intervals), sampled_memory_clocks_mhz=sorted(clocks))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    selection = json.loads(SELECTION.read_text()) if SELECTION.exists() else {}
    summary = dict(peak_gbps=PEAK_GBPS, records=check_records())
    for name, function, arguments in [('bytes_in_flight', chunk_stage, ()), ('pressure_ctas', cta_stage, (selection,)), ('target_rate', rate_stage, (selection,))]:
        rows = function(*arguments)
        if rows:
            summary[name] = rows
            write_csv(RESULTS / f'{name}.csv', rows)
    (RESULTS / 'analysis_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['records']))


if __name__ == '__main__':
    main()
