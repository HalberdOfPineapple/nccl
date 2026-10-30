#!/usr/bin/env python3
"""Plan 07 tables and figures for the store-only copy alone (E7.1) and the writer alone (E7.2).

Unit of repetition: one invocation. Copy: median of 12 measured transfers. Writer: median of five
20 ms windows. Each setting has 3 to 5 invocations (5 rounds before 2026-10-30, 3 after); tables give
the median over all invocations present, their min-max, and their count. Stages without data are skipped.
"""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from plot_style import set_zero_based_y, zero_based_upper
from run_store_congest import (COPY_CTAS, COPY_FLOOR_CTAS, COPY_FLOOR_THREADS, COPY_THREADS, MAXIMUM_ROUNDS, PEAK_GBPS, RESULTS, ROUNDS, WRITER_CTAS,
                               WRITER_THREADS, LEGACY_BATCHES, copy_label, legacy_copy_label, rate_levels, selection, summarize)

FIGURES = RESULTS / 'figures'
# Blue and orange mean healthy and perturbed elsewhere in this project; settings here are neither.
COLORS = ['#3a923a', '#9372b2', '#c03d3e', '#845b53', '#2a9d9a', '#b8860b']


def invocation(group, label):
    path = RESULTS / group / 'raw' / f'{label}.json'
    if not path.exists():
        return None
    data, rows = summarize(path)
    result = dict(gpus=f"{data['source_gpu']},{data['destination_gpu']}")
    for key in ('throughput_gbps', 'issue_us', 'ordering_us', 'ack_wait_us', 'completed_us', 'writer_gbps', 'writer_slowest_to_fastest'):
        values = [row[key] for row in rows if key in row]
        if values:
            result[key] = statistics.median(values) if key != 'writer_slowest_to_fastest' else min(values)
    copy_gpcs = sorted({data['gpc_of_sm'][sm] for sm, role in enumerate(data['role_of_sm']) if role == 1})
    writer_gpcs = sorted({data['gpc_of_sm'][sm] for sm, role in enumerate(data['role_of_sm']) if role == 2})
    result.update(copy_gpcs=' '.join(map(str, copy_gpcs)), writer_gpcs=' '.join(map(str, writer_gpcs)))
    return result


def summarize_setting(group, prefix, key, **fields):
    runs = [invocation(group, f'{prefix}_R{repetition}') for repetition in range(1, MAXIMUM_ROUNDS + 1)]
    if any(run is None for run in runs[:ROUNDS]):
        return None
    runs = [run for run in runs if run is not None]
    values = [run[key] for run in runs]
    row = dict(**fields, runs=len(runs), median=statistics.median(values), minimum=min(values), maximum=max(values))
    for extra in ('issue_us', 'ordering_us', 'ack_wait_us', 'completed_us', 'writer_slowest_to_fastest'):
        if extra in runs[0]:
            row[extra] = statistics.median(run[extra] for run in runs) if extra != 'writer_slowest_to_fastest' else min(run[extra] for run in runs)
    row['copy_gpcs'] = runs[0]['copy_gpcs']
    row['writer_gpcs'] = runs[0]['writer_gpcs']
    return row


def write_csv(path, rows):
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def style(axis):
    axis.grid(axis='y', linewidth=.4, alpha=.5)
    axis.spines[['top', 'right']].set_visible(False)


def error_bars(rows):
    return [[row['median'] - row['minimum'] for row in rows], [row['maximum'] - row['median'] for row in rows]]


def copy_prefix(ctas, threads, placement):
    """Records of one copy setting from the build without copy_batch (2026-10-30). The earlier
    build's records are not joined with them: the same 22-instruction loop in a different order ran
    7-8% slower per SM in the new build, so the two builds differ below the plateau."""
    return copy_label(ctas, threads, placement)


def copy_sweep_stage():
    """copy_sweep and copy_floor together: same loop, payload and spread placement."""
    rows = []
    for ctas in sorted(set(COPY_CTAS + COPY_FLOOR_CTAS)):
        for threads in sorted(set(COPY_THREADS + COPY_FLOOR_THREADS)):
            row = summarize_setting('copy_alone', copy_prefix(ctas, threads, 'spread'), 'throughput_gbps', copy_ctas=ctas, copy_threads=threads, placement='spread')
            if row:
                row['copy_threads_total'] = ctas * threads
                rows.append(row)
    if not rows:
        return None
    write_csv(RESULTS / 'copy_sweep_summary.csv', rows)
    write_plateau_table(rows)
    upper = zero_based_upper([row['maximum'] for row in rows])
    figure, (cta_axis, thread_axis) = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for color, threads in zip(COLORS, (128, 256, 512, 1024)):
        series = [row for row in rows if row['copy_threads'] == threads]
        cta_axis.errorbar([row['copy_ctas'] for row in series], [row['median'] for row in series], yerr=error_bars(series),
                          marker='o', markersize=3, linewidth=1, capsize=2, color=color, label=f'{threads} threads/CTA')
    cta_axis.set_xscale('log', base=2)
    cta_axis.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    cta_axis.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128'])
    cta_axis.set_xlabel('copy CTAs (log scale)')
    for color, ctas in zip(COLORS[4:] + COLORS[2:4], (16, 64, 128)):
        series = [row for row in rows if row['copy_ctas'] == ctas]
        thread_axis.errorbar([row['copy_threads'] for row in series], [row['median'] for row in series], yerr=error_bars(series),
                             marker='o', markersize=3, linewidth=1, capsize=2, color=color, label=f'{ctas} CTAs')
    thread_axis.set_xlabel('copy threads/CTA')
    thread_axis.set_xticks([128, 256, 512, 768, 1024])
    for axis in (cta_axis, thread_axis):
        set_zero_based_y(axis, upper)
        style(axis)
        axis.legend(fontsize=7, frameon=False)
    cta_axis.set_ylabel('copy throughput (GB/s)')
    figure.suptitle('Store-only copy alone, spread over all GPCs: CTA scaling (left) and thread scaling (right)\n'
                    '2 GiB payload, one 4-byte peer store per loop step, CTA-local stride; median of 3-5 runs, bars min-max', fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES / 'copy_sweep.png', dpi=150)
    plt.close(figure)
    return rows


def write_plateau_table(rows):
    """The plateau (median of all settings with 32 or more CTAs) and, for each threads/CTA value,
    the fewest CTAs whose median is within 1% of it."""
    plateau = statistics.median(row['median'] for row in rows if row['copy_ctas'] >= 32)
    table = []
    for threads in sorted({row['copy_threads'] for row in rows}):
        series = sorted((row for row in rows if row['copy_threads'] == threads), key=lambda row: row['copy_ctas'])
        reached = [row for row in series if row['median'] >= .99 * plateau]
        first = reached[0] if reached else None
        table.append(dict(copy_threads=threads, plateau_gbps=plateau, fewest_ctas_within_1_percent=first['copy_ctas'] if first else '',
                          median_gbps=first['median'] if first else '', copy_threads_total=first['copy_threads_total'] if first else ''))
    write_csv(RESULTS / 'copy_plateau_summary.csv', table)


def loop_step_stage():
    """The removed copy_batch knob (stores per loop step), kept as the evidence for removing it."""
    rows = []
    for ctas in sorted(set(COPY_CTAS + COPY_FLOOR_CTAS)):
        for threads in sorted(set(COPY_THREADS + COPY_FLOOR_THREADS)):
            for batch in LEGACY_BATCHES:
                row = summarize_setting('copy_alone', legacy_copy_label(ctas, threads, batch, 'spread'), 'throughput_gbps',
                                        copy_ctas=ctas, copy_threads=threads, stores_per_loop_step=batch, placement='spread')
                if row:
                    rows.append(row)
    if not rows:
        return None
    write_csv(RESULTS / 'copy_loop_step_summary.csv', rows)
    return rows


def copy_placement_stage():
    try:
        settings = selection()['copy_placement_settings']
    except (FileNotFoundError, KeyError):
        return None
    rows = []
    for setting in settings:
        for placement in ('spread', 'packed'):
            row = summarize_setting('copy_alone', copy_prefix(setting['ctas'], setting['threads'], placement), 'throughput_gbps',
                                    copy_ctas=setting['ctas'], copy_threads=setting['threads'], placement=placement)
            if row is None:
                return None
            rows.append(row)
    write_csv(RESULTS / 'copy_placement_summary.csv', rows)
    figure, axis = plt.subplots(figsize=(1.6 + 1.3 * len(settings), 3.6))
    width = .38
    for offset, (placement, color) in enumerate((('spread', COLORS[0]), ('packed', COLORS[1]))):
        series = [row for row in rows if row['placement'] == placement]
        positions = [index + (offset - .5) * width for index in range(len(series))]
        axis.bar(positions, [row['median'] for row in series], width, color=color, label='spread over all 8 GPCs' if placement == 'spread' else 'packed into the fewest GPCs')
        axis.errorbar(positions, [row['median'] for row in series], yerr=error_bars(series), fmt='none', ecolor='black', linewidth=.8, capsize=2)
    axis.set_xticks(range(len(settings)))
    packed_gpcs = [row['copy_gpcs'] for row in rows if row['placement'] == 'packed']
    axis.set_xticklabels([f"{setting['ctas']} CTAs × {setting['threads']} threads/CTA\npacked into G{', G'.join(gpcs.split())}"
                          for setting, gpcs in zip(settings, packed_gpcs)], fontsize=7)
    set_zero_based_y(axis, zero_based_upper([row['maximum'] for row in rows]))
    axis.set_ylabel('copy throughput (GB/s)')
    axis.legend(fontsize=7, frameon=False)
    style(axis)
    axis.set_title('Store-only copy alone: copy CTAs spread over all GPCs vs packed into the fewest GPCs', fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / 'copy_placement.png', dpi=150)
    plt.close(figure)
    return rows


def writer_sweep_stage():
    rows = []
    for ctas in WRITER_CTAS:
        for threads in WRITER_THREADS:
            row = summarize_setting('writer_alone', f'writer_w{ctas}_t{threads}_unlimited_spread', 'writer_gbps', writer_ctas=ctas, writer_threads=threads, placement='spread')
            if row:
                row['percent_of_peak'] = 100 * row['median'] / PEAK_GBPS
                row['per_cta_gbps'] = row['median'] / ctas
                rows.append(row)
    if not rows:
        return None
    write_csv(RESULTS / 'writer_sweep_summary.csv', rows)
    figure, (cta_axis, thread_axis) = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, gridspec_kw=dict(width_ratios=[2, 1]))
    series = [row for row in rows if row['writer_threads'] == 1024]
    percent = lambda value: 100 * value / PEAK_GBPS
    cta_axis.errorbar([row['writer_ctas'] for row in series], [percent(row['median']) for row in series],
                      yerr=[[percent(row['median'] - row['minimum']) for row in series], [percent(row['maximum'] - row['median']) for row in series]],
                      marker='o', markersize=3, linewidth=1, capsize=2, color=COLORS[0], label='1024 threads/CTA')
    cta_axis.set_xlabel('writer CTAs (one per SM)')
    cta_axis.set_ylabel('achieved write rate (% of HBM peak)')
    for color, ctas in zip(COLORS[2:], (32, 64)):
        series = [row for row in rows if row['writer_ctas'] == ctas]
        thread_axis.errorbar([row['writer_threads'] for row in series], [percent(row['median']) for row in series], marker='o', markersize=3, linewidth=1, color=color, label=f'{ctas} writer CTAs')
    thread_axis.set_xlabel('writer threads/CTA')
    thread_axis.set_xticks(WRITER_THREADS)
    for axis in (cta_axis, thread_axis):
        axis.set_ylim(0, 100)
        axis.yaxis.set_major_locator(MultipleLocator(20))
        axis.legend(fontsize=7, frameon=False)
        style(axis)
    figure.suptitle(f'Writer alone, no rate limit, spread over all GPCs (16-byte stores into a 1 GiB buffer; peak {PEAK_GBPS:.0f} GB/s)\n'
                    'median of 3-5 runs × five 20 ms windows; bars min-max', fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES / 'writer_sweep.png', dpi=150)
    plt.close(figure)
    return rows


def writer_rate_stage():
    try:
        chosen = selection()['writer']
    except (FileNotFoundError, KeyError):
        return None
    rows = []
    for placement in ('spread', 'packed'):
        for level, target in rate_levels():
            row = summarize_setting('writer_alone', f"writer_w{chosen['ctas']}_t{chosen['threads']}_{level}_{placement}", 'writer_gbps',
                                    writer_ctas=chosen['ctas'], writer_threads=chosen['threads'], placement=placement, level=level, target_gbps=target)
            if row:
                row['percent_of_peak'] = 100 * row['median'] / PEAK_GBPS
                row['achieved_over_target'] = row['median'] / target if target else None
                rows.append(row)
    if not rows:
        return None
    write_csv(RESULTS / 'writer_rate_summary.csv', rows)
    figure, axis = plt.subplots(figsize=(5.4, 4))
    for color, placement, marker in ((COLORS[0], 'spread', 'o'), (COLORS[1], 'packed', 's')):
        series = [row for row in rows if row['placement'] == placement and row['target_gbps']]
        unlimited = [row for row in rows if row['placement'] == placement and not row['target_gbps']]
        if not series + unlimited:
            continue
        gpcs = (series + unlimited)[0]['writer_gpcs'].split()
        name = f"{placement}: {len(gpcs)} GPCs (G{', G'.join(gpcs)})"
        if series:
            axis.plot([100 * row['target_gbps'] / PEAK_GBPS for row in series], [row['percent_of_peak'] for row in series], marker=marker, markersize=4, linewidth=1, color=color, label=name)
        for row in unlimited:
            axis.axhline(row['percent_of_peak'], color=color, linewidth=.8, linestyle=':', label=f'{placement}, no limit: {row["percent_of_peak"]:.1f}%')
    axis.plot([0, 100], [0, 100], color='#888888', linewidth=.6, linestyle='--', label='achieved = target')
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.xaxis.set_major_locator(MultipleLocator(20))
    axis.yaxis.set_major_locator(MultipleLocator(20))
    axis.set_xlabel('target write rate (% of HBM peak)')
    axis.set_ylabel('achieved write rate (% of HBM peak)')
    axis.legend(fontsize=7, frameon=False)
    style(axis)
    axis.set_title(f"Writer alone: {chosen['ctas']} CTAs × {chosen['threads']} threads/CTA, 64 KiB blocks\nmedian of 3-5 runs", fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES / 'writer_rate.png', dpi=150)
    plt.close(figure)
    return rows


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for stage in (copy_sweep_stage, loop_step_stage, copy_placement_stage, writer_sweep_stage, writer_rate_stage):
        rows = stage()
        print(stage.__name__, 'skipped' if rows is None else f'{len(rows)} settings')


if __name__ == '__main__':
    main()
