#!/usr/bin/env python3
"""Figure for 'Communication reference selection' in the shared-resource report.

Reads only saved per-run tables; adds no measurements.
Copy throughput against total issuing threads (CTAs × threads/CTA) for every
          calibration stage. A line joins only runs from one matched sweep (same GPU pair,
          build, payload, batch, traversal, and the same fixed launch dimension).
"""
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COMMON = Path(__file__).resolve().parents[1]
RESULTS = COMMON.parent / 'observation/shared_resources/results'
FIGURES = RESULTS / '2026-10-27-saturation-gpu6-7/figures'

# One colour per calibration stage. Blue and orange are kept for healthy/perturbed elsewhere.
STAGES = {
    'gpu45': ('#7f7f7f', 'GPUs 4 → 5, first build: 256 MiB, 1 element per thread'),
    'gpu67': ('#2a9d9a', 'GPUs 6 → 7, first build: 256 MiB, 1 element per thread'),
    'expanded': ('#9372b2', 'GPUs 6 → 7, larger limits: 256 MiB, 1 element per thread'),
    'final': ('#3a923a', 'GPUs 6 → 7, final build: 2048 MiB, 8 elements per thread'),
}


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def summarize(rows, field):
    """Median and min–max of invocation values, per (CTAs, threads/CTA)."""
    groups = {}
    for row in rows:
        groups.setdefault((int(float(row['ctas'])), int(float(row['threads']))), []).append(float(row[field]))
    return {key: (statistics.median(values), min(values), max(values), len(values)) for key, values in sorted(groups.items())}


def load_sweeps():
    """Each sweep: stage, which knob varies, its fixed value, and summarized points."""
    first = read_rows(RESULTS / '2026-10-27/runs.csv')
    gpu67 = read_rows(RESULTS / '2026-10-27-gpu6-7/runs.csv')
    calibration = read_rows(RESULTS / '2026-10-27-saturation-gpu6-7/runs.csv')
    gpu67_local = [row for row in gpu67 if row['run'].startswith('baseline_') and '_g0_' in row['run']]
    expanded = [row for row in calibration if row['phase'] == 'concurrency' and row['run'].startswith('confirm_')]
    final = [row for row in calibration if row['phase'] == 'batch' and row['run'].startswith('confirm_') and row['payload_mib'] == '2048' and row['batch'] == '8']
    sweeps = [
        ('gpu45', 'ctas', 256, summarize([row for row in first if row['config'].startswith('baseline_ctas_')], 'throughput_gbps')),
        ('gpu45', 'threads', 32, summarize([row for row in first if row['config'].startswith('baseline_threads_')], 'throughput_gbps')),
        ('gpu67', 'ctas', 256, summarize([row for row in gpu67_local if float(row['threads']) == 256], 'throughput_gbps')),
        ('gpu67', 'ctas', 512, summarize([row for row in gpu67_local if float(row['threads']) == 512], 'throughput_gbps')),
        ('gpu67', 'threads', 32, summarize([row for row in gpu67_local if float(row['ctas']) == 32], 'throughput_gbps')),
        ('expanded', 'threads', 128, summarize([row for row in expanded if row['ctas'] == '128' and row['traversal'] == '0'], 'gbps')),
        ('expanded', 'ctas', 1024, summarize([row for row in expanded if row['threads'] == '1024' and row['traversal'] == '0'], 'gbps')),
        ('final', 'threads', 128, summarize([row for row in final if row['ctas'] == '128'], 'gbps')),
        ('final', 'ctas', 1024, summarize([row for row in final if row['threads'] == '1024'], 'gbps')),
    ]
    # Single-invocation screens with CTA-local stride, shown without lines.
    screens = summarize([row for row in calibration if row['phase'] == 'concurrency' and row['run'].startswith('screen_') and row['traversal'] == '0'], 'gbps')
    # Global-stride confirmations at 96 and 132 CTAs, where CTA regions are not aligned.
    global_stride = summarize([row for row in expanded if row['traversal'] == '1'], 'gbps')
    return sweeps, screens, global_stride


def plot_scaling(sweeps, screens, global_stride):
    figure, axis = plt.subplots(figsize=(10, 6.2), layout='constrained')
    for stage, varied, fixed, points in sweeps:
        color = STAGES[stage][0]
        keys = list(points)
        totals = [ctas * threads for ctas, threads in keys]
        medians = [points[key][0] for key in keys]
        errors = [[points[key][0] - points[key][1] for key in keys], [points[key][2] - points[key][0] for key in keys]]
        marker = 'o' if varied == 'ctas' else 's'
        linestyle = '-' if varied == 'ctas' else '--'
        axis.errorbar(totals, medians, yerr=errors, color=color, marker=marker, markersize=5, linestyle=linestyle, linewidth=1.2, capsize=2)
    for (ctas, threads), (median, _, _, _) in screens.items():
        axis.plot(ctas * threads, median, marker='o', markersize=6, markerfacecolor='none', color=STAGES['expanded'][0], linestyle='none')
        if ctas in (96, 132):
            axis.annotate(f'{ctas} × {threads}\none run', (ctas * threads, median), xytext=(8, -4), textcoords='offset points', fontsize=7, color=STAGES['expanded'][0])
    for (ctas, threads), (median, _, _, _) in global_stride.items():
        axis.plot(ctas * threads, median, marker='^', markersize=6, color=STAGES['expanded'][0], linestyle='none')
    selected = next(points for stage, varied, fixed, points in sweeps if stage == 'final' and varied == 'threads')[(128, 512)][0]
    axis.annotate(f'Selected: 128 × 512\n{selected:.1f} GB/s', (128 * 512, selected), xytext=(-110, 22), textcoords='offset points', fontsize=8,
                  arrowprops=dict(arrowstyle='->', linewidth=.8))
    axis.set_xscale('log', base=2)
    axis.set_xlim(800, 2 ** 18)
    ticks = [2 ** power for power in range(10, 18)]
    axis.set_xticks(ticks, [f'{tick // 1024}K' for tick in ticks])
    axis.set(ylim=(0, 450), xlabel='Total issuing threads = communication CTAs × threads per CTA (log scale)', ylabel='Completed copy throughput (GB/s)',
             title='Copy throughput as issuing threads increase, communication alone')
    axis.grid(axis='y', linewidth=.4, alpha=.5)
    axis.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([], [], color=color, linewidth=2, label=label) for color, label in STAGES.values()]
    handles += [Line2D([], [], color='black', marker='o', linestyle='-', label='line: CTAs change, threads/CTA fixed'),
                Line2D([], [], color='black', marker='s', linestyle='--', label='line: threads/CTA change, CTAs fixed'),
                Line2D([], [], color='black', marker='o', markerfacecolor='none', linestyle='none', label='one run only (screen), CTA-local stride'),
                Line2D([], [], color='black', marker='^', linestyle='none', label='global stride, 5 runs')]
    axis.legend(handles=handles, fontsize=7.5, loc='upper left', frameon=False)
    # The caption in the report explains points, bars and labels.
    figure.savefig(FIGURES / 'reference_scaling.png', dpi=150)
    plt.close(figure)


def main():
    sweeps, screens, global_stride = load_sweeps()
    plot_scaling(sweeps, screens, global_stride)


if __name__ == '__main__':
    main()
