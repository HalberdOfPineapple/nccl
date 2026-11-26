#!/usr/bin/env python3
"""Figures for the AllReduce scalability series, made only from saved summary CSVs.

Usage (matplotlib through uv):
  uv run --no-project --with matplotlib python make_figures.py calibration SUMMARY_CSV OUTPUT_PNG
  uv run --no-project --with matplotlib python make_figures.py scaling SUMMARY_CSV OUTPUT_PNG [--exclude-gpus 6[,...]]
  uv run --no-project --with matplotlib python make_figures.py payload SUMMARY_CSV OUTPUT_PNG
  uv run --no-project --with matplotlib python make_figures.py link LINK_CSV OUTPUT_PNG

Each arm has one fixed colour in every figure. Rate axes start at zero and comparable panels share a scale.
"""
import csv
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARM_NAMES = {'ring': 'Ring', 'nvls': 'NVLS', 'sym_multicast': 'Symmetric multicast', 'sym_plain': 'Symmetric plain load/store'}
ARM_COLOURS = {'ring': '#1f77b4', 'nvls': '#ff7f0e', 'sym_multicast': '#2ca02c', 'sym_plain': '#9467bd'}
ARM_ORDER = ['ring', 'nvls', 'sym_multicast', 'sym_plain']


def read_rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def add_arm_legend(figure):
    """One legend for all panels, built from the arm colours, so an arm missing from one panel is still listed."""
    handles = [plt.Line2D([], [], color=ARM_COLOURS[arm], marker='o', markersize=4, linewidth=1.2, label=ARM_NAMES[arm]) for arm in ARM_ORDER]
    figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=4, fontsize=9, frameon=False)


def rounded_top(value):
    step = 50 if value > 200 else 25
    return step * math.ceil(value * 1.08 / step)


def figure_calibration(rows, output):
    """algBW against requested CTA count, one panel per GPU count; NCCL's own choice as a marked point."""
    gpu_counts = sorted({int(row['gpu_count']) for row in rows})
    top = rounded_top(max(float(row['algbw_max_gbps']) for row in rows))
    figure, axes = plt.subplots(1, len(gpu_counts), figsize=(5.2 * len(gpu_counts), 4.2), squeeze=False, sharey=True)
    for axis, gpu_count in zip(axes[0], gpu_counts):
        for arm in ARM_ORDER:
            arm_rows = [row for row in rows if row['arm'] == arm and int(row['gpu_count']) == gpu_count]
            if not arm_rows:
                continue
            fixed = sorted((int(row['requested_ctas']), float(row['algbw_median_gbps']), float(row['algbw_min_gbps']), float(row['algbw_max_gbps'])) for row in arm_rows if row['requested_ctas'] != 'auto')
            xs = [item[0] for item in fixed]
            ys = [item[1] for item in fixed]
            errors = [[item[1] - item[2] for item in fixed], [item[3] - item[1] for item in fixed]]
            axis.errorbar(xs, ys, yerr=errors, color=ARM_COLOURS[arm], marker='o', markersize=3.5, linewidth=1.2, capsize=2, label=ARM_NAMES[arm])
            for row in arm_rows:
                if row['requested_ctas'] == 'auto':
                    used = int(row['ctas_used'].split()[0])
                    axis.scatter([used], [float(row['algbw_median_gbps'])], marker='*', s=110, color=ARM_COLOURS[arm], edgecolor='black', linewidth=0.6, zorder=5)
        axis.set_xscale('log', base=2)
        axis.set_xticks([2, 4, 8, 16, 24, 32, 48, 64])
        axis.set_xticklabels(['2', '4', '8', '16', '24', '32', '48', '64'])
        axis.set_ylim(0, top)
        axis.set_xlabel('CTAs requested (log scale)')
        axis.set_title(f'{gpu_count} GPUs ({arm_rows[0]["devices"] if arm_rows else ""}), 2 GiB per GPU')
        axis.grid(alpha=0.3)
    axes[0][0].set_ylabel('algBW, GB/s (payload per GPU / time of one call)')
    add_arm_legend(figure)
    figure.text(0.5, -0.02, 'Lines: median of 3 runs, bars: min-max. Star: CTA count NCCL chose by itself. Threads/CTA: Ring 512, NVLS 640, symmetric kernels 512.', ha='center', fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=130, bbox_inches='tight')


GPU_SETS = {2: '{4,5}', 4: '{4-7}', 6: '{1-6}', 8: '{0-7}'}


def figure_scaling(rows, output, excluded=()):
    """algBW and busBW against GPU count; one line per arm and CTA setting. GPU counts in `excluded` are left out."""
    rows = [row for row in rows if int(row['gpu_count']) not in excluded]
    gpu_counts = sorted({int(row['gpu_count']) for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # Room for the 478 GB/s link rating line on both panels (shared scale).
    top = max(500, rounded_top(max(max(float(row['busbw_median_gbps']), float(row['algbw_median_gbps'])) for row in rows)))
    settings = (('auto', '-', 'o', 'CTAs chosen by NCCL'), ('c16', '--', 's', '16 CTAs fixed'), ('c48', ':', '^', '48 CTAs fixed'))
    for setting, style, marker, setting_label in settings:
        for arm in ARM_ORDER:
            series = sorted((int(row['gpu_count']), row) for row in rows if row['arm'] == arm and row['cell_id'].endswith('_' + setting))
            if not series:
                continue
            xs = [item[0] for item in series]
            for axis, column in zip(axes, ('algbw_median_gbps', 'busbw_median_gbps')):
                ys = [float(item[1][column]) for item in series]
                axis.plot(xs, ys, linestyle=style, color=ARM_COLOURS[arm], marker=marker, markersize=4, linewidth=1.3,
                          label=f'{ARM_NAMES[arm]}, {setting_label}')
    for axis, label in zip(axes, ('algBW, GB/s', 'busBW, GB/s (factor of each arm)')):
        axis.set_xticks(gpu_counts)
        axis.set_xlabel('GPU count')
        axis.set_ylabel(label)
        axis.set_ylim(0, top)
        axis.grid(alpha=0.3)
    axes[0].set_title('algBW = payload per GPU / time of one call')
    axes[1].set_title('busBW: Ring, plain x 2(N-1)/N; multimem x (1+1/N)')
    axes[1].axhline(478, color='grey', linewidth=0.8, linestyle=':')
    axes[1].text(2.05, 470, 'NVLink rating, 478 GB/s per direction', fontsize=7, color='grey', va='top')
    axes[0].legend(fontsize=7, loc='lower left')
    gpu_sets = ', '.join(f'{n} = {GPU_SETS[n]}' for n in gpu_counts)
    left_out = f' {", ".join(str(n) for n in sorted(excluded))} GPUs left out.' if excluded else ''
    figure.text(0.5, -0.02, f'Payload 2 GiB per GPU. Median of 5 runs. GPUs: {gpu_sets}.{left_out} Threads/CTA: Ring 512, NVLS 640, symmetric 512. CTA counts NCCL chose are in the report table.', ha='center', fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=130, bbox_inches='tight')


def figure_payload(rows, output):
    """algBW against payload, one panel per GPU count."""
    gpu_counts = sorted({int(row['gpu_count']) for row in rows})
    top = rounded_top(max(float(row['algbw_median_gbps']) for row in rows))
    figure, axes = plt.subplots(1, len(gpu_counts), figsize=(4.6 * len(gpu_counts), 4.2), squeeze=False, sharey=True)
    for axis, gpu_count in zip(axes[0], gpu_counts):
        for arm in ARM_ORDER:
            series = sorted((int(row['size_bytes']), float(row['algbw_median_gbps'])) for row in rows if row['arm'] == arm and int(row['gpu_count']) == gpu_count)
            if series:
                axis.plot([item[0] / 2**20 for item in series], [item[1] for item in series], color=ARM_COLOURS[arm], marker='o', markersize=3.5, linewidth=1.2, label=ARM_NAMES[arm])
        axis.set_xscale('log', base=2)
        ticks = [32, 128, 512, 2048, 8192]
        axis.set_xticks(ticks)
        axis.set_xticklabels(['32 MiB', '128 MiB', '512 MiB', '2 GiB', '8 GiB'])
        axis.set_ylim(0, top)
        axis.set_xlabel('Payload per GPU (log scale)')
        axis.set_title(f'{gpu_count} GPUs')
        axis.grid(alpha=0.3)
    axes[0][0].set_ylabel('algBW, GB/s')
    add_arm_legend(figure)
    figure.text(0.5, -0.02, 'CTAs chosen by NCCL; for each arm and GPU count NCCL used the same CTA count at every payload (Ring 24, NVLS 16, symmetric multicast 4 at 4 GPUs and 2 at 6-8, plain 8). Median of 5 runs. GPUs: 2 = {4,5}, 4 = {4-7}, 6 = {1-6}, 8 = {0-7}.', ha='center', fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=130, bbox_inches='tight')


def figure_link(rows, output):
    """Left: measured link bytes per payload byte (markers) against the two byte models (grey lines).
    Right: link rate in use as a share of the 478 GB/s rating. Arms are shifted a little sideways so
    arms with equal values (Ring and plain load/store share one model) stay visible."""
    figure, (bytes_axis, share_axis) = plt.subplots(1, 2, figsize=(11, 4.4))
    gpu_counts = [2, 4, 6, 8]
    bytes_axis.plot(gpu_counts, [2 * (n - 1) / n for n in gpu_counts], color='grey', linewidth=1, label='Model 2(N-1)/N (Ring, plain)')
    bytes_axis.plot(gpu_counts, [1 + 1 / n for n in gpu_counts], color='grey', linewidth=1, linestyle='--', label='Model 1+1/N (multimem)')
    for index, arm in enumerate(ARM_ORDER):
        byte_points = {}
        share_points = {}
        for row in rows:
            if row['arm'] != arm:
                continue
            gpu_count = int(row['gpu_count'])
            byte_points.setdefault(gpu_count, []).append(float(row['link_bytes_per_payload_byte_median']))
            share_points.setdefault(gpu_count, []).append(float(row['share_of_link_rating']))
        if not byte_points:
            continue
        xs = sorted(byte_points)
        shifted = [x + (index - 1.5) * 0.08 for x in xs]
        bytes_axis.scatter(shifted, [statistics.median(byte_points[x]) for x in xs], color=ARM_COLOURS[arm], s=22, zorder=3, label=f'{ARM_NAMES[arm]}, measured')
        share_axis.plot(xs, [statistics.median(share_points[x]) for x in xs], color=ARM_COLOURS[arm], marker='o', markersize=4, linewidth=1.3, label=ARM_NAMES[arm])
    for axis in (bytes_axis, share_axis):
        axis.set_xticks(gpu_counts)
        axis.set_xlabel('GPU count')
        axis.grid(alpha=0.3)
    bytes_axis.set_ylim(0, 2.0)
    bytes_axis.set_ylabel('NVLink data bytes per payload byte, per GPU and direction')
    bytes_axis.set_title('Measured bytes against the byte models')
    bytes_axis.legend(fontsize=7, loc='lower right')
    share_axis.set_ylim(0, 1.0)
    share_axis.set_ylabel('Link rate in use / 478 GB/s')
    share_axis.set_title('Share of the NVLink rating in use')
    share_axis.legend(fontsize=7, loc='lower right')
    figure.text(0.5, -0.02, 'Payload 2 GiB per GPU, CTAs chosen by NCCL. Long run (200 timed calls) minus short run (20). Median over GPUs, both directions and 3 rounds.', ha='center', fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=130, bbox_inches='tight')


def main():
    kind, source, output = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(source)
    if '--exclude-gpus' in sys.argv:
        if kind != 'scaling':
            sys.exit('--exclude-gpus is supported only for the scaling figure')
        excluded = {int(n) for n in sys.argv[sys.argv.index('--exclude-gpus') + 1].split(',')}
        figure_scaling(rows, output, excluded)
        print(output)
        return
    {'calibration': figure_calibration, 'scaling': figure_scaling, 'payload': figure_payload, 'link': figure_link}[kind](rows, output)
    print(output)


if __name__ == '__main__':
    main()
