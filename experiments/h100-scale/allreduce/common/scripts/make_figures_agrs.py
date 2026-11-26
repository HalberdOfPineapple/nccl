#!/usr/bin/env python3
"""Figures for the AllGather and ReduceScatter scalability series, made only from saved summary CSVs.

Usage (matplotlib through uv):
  uv run --no-project --with matplotlib python make_figures_agrs.py calibration SUMMARY_CSV OUTPUT_PNG
  uv run --no-project --with matplotlib python make_figures_agrs.py scaling SUMMARY_CSV OUTPUT_PNG
  uv run --no-project --with matplotlib python make_figures_agrs.py payload SUMMARY_CSV OUTPUT_PNG
  uv run --no-project --with matplotlib python make_figures_agrs.py link LINK_CSV OUTPUT_PNG

One row of panels per collective. Arm colours and names are those of the AllReduce figures (make_figures.py).
Rate axes start at zero; panels of one figure share a scale.
"""
import csv
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from make_figures import ARM_COLOURS, ARM_NAMES, ARM_ORDER, rounded_top

ARM_NAMES = {**ARM_NAMES, 'nvls': 'Classic NVLS'}
COLLECTIVE_NAMES = {'allgather': 'AllGather', 'reducescatter': 'ReduceScatter'}
COLLECTIVE_ORDER = ['allgather', 'reducescatter']
GPU_SETS = {2: '{4,5}', 4: '{4-7}', 6: '{1-6}', 8: '{0-7}'}
SIZE_2_GIB = 2 * 2**30
THREADS_TEXT = 'Threads/CTA: Ring 512, classic NVLS 640, symmetric kernels 512.'
# One line per arm and CTA setting; points of different settings are never joined.
SETTING_STYLES = {'auto': ('-', 'o', "NCCL's CTA choice"), '32': ('--', 's', '32 CTAs fixed'), '64': (':', '^', '64 CTAs fixed')}


def settings_of(rows, collective, arm):
    return [setting for setting in SETTING_STYLES if any(row['collective'] == collective and row['arm'] == arm and row['requested_ctas'] == setting for row in rows)]


def read_rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def caption_and_legend(figure, axes_list, text):
    """One arm legend under all panels, then the caption under the legend."""
    entries = {}
    for axis in axes_list:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            entries.setdefault(label, handle)
    legend = figure.legend(list(entries.values()), list(entries.keys()), loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8, frameon=False)
    figure.canvas.draw()
    bottom = legend.get_window_extent().transformed(figure.transFigure.inverted()).y0 - 0.01
    width = max(80, int(figure.get_figwidth() * 15))
    figure.text(0.5, bottom, '\n'.join(textwrap.wrap(text, width)), ha='center', va='top', fontsize=8)


def figure_calibration(rows, output):
    """algBW against requested CTAs; star = NCCL's own choice. Rows: collective; columns: GPU count."""
    rows = [row for row in rows if row['cell_id'].startswith('calibration_')]
    gpu_counts = sorted({int(row['gpu_count']) for row in rows})
    top = rounded_top(max(float(row['algbw_max_gbps']) for row in rows))
    figure, axes = plt.subplots(2, len(gpu_counts), figsize=(4.6 * len(gpu_counts), 7.6), squeeze=False, sharey=True)
    for row_index, collective in enumerate(COLLECTIVE_ORDER):
        for axis, gpu_count in zip(axes[row_index], gpu_counts):
            for arm in ARM_ORDER:
                arm_rows = [row for row in rows if row['collective'] == collective and row['arm'] == arm and int(row['gpu_count']) == gpu_count]
                if not arm_rows:
                    continue
                fixed = sorted((int(row['requested_ctas']), float(row['algbw_median_gbps']), float(row['algbw_min_gbps']), float(row['algbw_max_gbps']))
                               for row in arm_rows if row['requested_ctas'] != 'auto')
                if fixed:
                    errors = [[item[1] - item[2] for item in fixed], [item[3] - item[1] for item in fixed]]
                    axis.errorbar([item[0] for item in fixed], [item[1] for item in fixed], yerr=errors, color=ARM_COLOURS[arm], marker='o',
                                  markersize=3.5, linewidth=1.2, capsize=2, label=ARM_NAMES[arm])
                for row in arm_rows:
                    if row['requested_ctas'] == 'auto':
                        axis.scatter([int(row['ctas_used'].split()[0])], [float(row['algbw_median_gbps'])], marker='*', s=110,
                                     color=ARM_COLOURS[arm], edgecolor='black', linewidth=0.6, zorder=5)
            axis.set_xscale('log', base=2)
            axis.set_xticks([2, 4, 8, 16, 32, 64])
            axis.set_xticklabels(['2', '4', '8', '16', '32', '64'])
            axis.set_ylim(0, top)
            axis.grid(alpha=0.3)
            axis.set_title(f'{COLLECTIVE_NAMES[collective]}, {gpu_count} GPUs {GPU_SETS[gpu_count]}', fontsize=10)
            if row_index == 1:
                axis.set_xlabel('CTAs requested (log scale)')
        axes[row_index][0].set_ylabel('algBW, GB/s')
    figure.tight_layout()
    caption_and_legend(figure, axes.flatten(), 'Payload 2 GiB (nccl-tests size: AllGather output, ReduceScatter input, per GPU). Lines: median of 3 runs, '
                       'bars: min-max. Star: CTA count NCCL chose by itself (at 2 and 6 GPUs only NCCL\'s choice and twice it were run). ' + THREADS_TEXT)
    figure.savefig(output, dpi=130, bbox_inches='tight')


def figure_scaling(rows, output):
    """algBW and busBW against GPU count at 2 GiB, NCCL's CTA choice. Rows: collective."""
    rows = [row for row in rows if row['cell_id'].startswith('scaling_')]
    gpu_counts = sorted({int(row['gpu_count']) for row in rows})
    top = rounded_top(max(max(float(row['algbw_median_gbps']), float(row['busbw_median_gbps'])) for row in rows))
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for row_index, collective in enumerate(COLLECTIVE_ORDER):
        for axis, column, title in zip(axes[row_index], ('algbw_median_gbps', 'busbw_median_gbps'),
                                       ('algBW = size / time of one call', 'busBW: Ring, plain x (N-1)/N; multicast arms x 1')):
            for arm in ARM_ORDER:
                for setting in settings_of(rows, collective, arm):
                    style, marker, setting_label = SETTING_STYLES[setting]
                    series = sorted((int(row['gpu_count']), float(row[column])) for row in rows
                                    if row['collective'] == collective and row['arm'] == arm and row['requested_ctas'] == setting)
                    axis.plot([item[0] for item in series], [item[1] for item in series], color=ARM_COLOURS[arm], linestyle=style, marker=marker,
                              markersize=4, linewidth=1.3, label=f'{ARM_NAMES[arm]}, {setting_label}')
            axis.set_xticks(gpu_counts)
            axis.set_ylim(0, top)
            axis.grid(alpha=0.3)
            axis.set_title(f'{COLLECTIVE_NAMES[collective]}: {title}', fontsize=10)
            if row_index == 1:
                axis.set_xlabel('GPU count')
        axes[row_index][0].set_ylabel('GB/s')
    for axis in axes[:, 1]:
        axis.axhline(478, color='grey', linewidth=0.8, linestyle=':')
    axes[0][1].text(gpu_counts[0], 470, 'NVLink rating, 478 GB/s per direction', fontsize=7, color='grey', va='top')
    figure.tight_layout()
    sets = ', '.join(f'{n} = {GPU_SETS[n]}' for n in gpu_counts)
    caption_and_legend(figure, axes.flatten(), f'Payload 2 GiB per GPU. Solid: CTAs chosen by NCCL (in the report table); dashed and dotted: fixed CTA counts. Median of 5 runs. GPUs: {sets}. '
                       'Symmetric multicast is not available at 2 GPUs. ' + THREADS_TEXT)
    figure.savefig(output, dpi=130, bbox_inches='tight')


def figure_payload(rows, output):
    """algBW against payload, NCCL's CTA choice. Rows: collective; columns: GPU count."""
    rows = [row for row in rows if row['cell_id'].startswith('payload_')]
    gpu_counts = sorted({int(row['gpu_count']) for row in rows})
    top = rounded_top(max(float(row['algbw_median_gbps']) for row in rows))
    figure, axes = plt.subplots(2, len(gpu_counts), figsize=(4.6 * len(gpu_counts), 7.6), squeeze=False, sharey=True)
    for row_index, collective in enumerate(COLLECTIVE_ORDER):
        for axis, gpu_count in zip(axes[row_index], gpu_counts):
            for arm in ARM_ORDER:
                for setting in settings_of(rows, collective, arm):
                    style, marker, setting_label = SETTING_STYLES[setting]
                    series = sorted((int(row['size_bytes']), float(row['algbw_median_gbps'])) for row in rows if row['collective'] == collective
                                    and row['arm'] == arm and int(row['gpu_count']) == gpu_count and row['requested_ctas'] == setting)
                    if series:
                        axis.plot([item[0] / 2**20 for item in series], [item[1] for item in series], color=ARM_COLOURS[arm], linestyle=style,
                                  marker=marker, markersize=3, linewidth=1.2, label=f'{ARM_NAMES[arm]}, {setting_label}')
            axis.set_xscale('log', base=2)
            axis.set_xticks([32, 128, 512, 2048, 8192])
            axis.set_xticklabels(['32M', '128M', '512M', '2G', '8G'])
            axis.set_ylim(0, top)
            axis.grid(alpha=0.3)
            axis.set_title(f'{COLLECTIVE_NAMES[collective]}, {gpu_count} GPUs {GPU_SETS[gpu_count]}', fontsize=10)
            if row_index == 1:
                axis.set_xlabel('Size per GPU (nccl-tests), bytes')
        axes[row_index][0].set_ylabel('algBW, GB/s')
    figure.tight_layout()
    caption_and_legend(figure, axes.flatten(), 'Solid: CTAs chosen by NCCL (recorded per size in the summary); dashed and dotted: fixed CTA counts. Median of 5 runs. ' + THREADS_TEXT)
    figure.savefig(output, dpi=130, bbox_inches='tight')


def figure_link(rows, output):
    """Link bytes per payload byte, sent and received, against the model. One panel per collective."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    top = max(max(float(row['link_bytes_per_payload_byte_max']), float(row['model_bytes_per_payload_byte'])) for row in rows) * 1.15
    for axis, collective in zip(axes, COLLECTIVE_ORDER):
        selected = [row for row in rows if row.get('collective') == collective]
        cells = sorted({(int(row['gpu_count']), row['arm']) for row in selected}, key=lambda item: (-item[0], ARM_ORDER.index(item[1])))
        for position, (gpu_count, arm) in enumerate(cells):
            for offset, direction, hatch in ((-0.2, 'Tx', ''), (0.2, 'Rx', '//')):
                values = sorted(float(row['link_bytes_per_payload_byte_median']) for row in selected
                                if int(row['gpu_count']) == gpu_count and row['arm'] == arm and row['direction'] == direction)
                model = [float(row['model_bytes_per_payload_byte']) for row in selected
                         if int(row['gpu_count']) == gpu_count and row['arm'] == arm and row['direction'] == direction][0]
                axis.bar(position + offset, values[len(values) // 2], width=0.4, color=ARM_COLOURS[arm], hatch=hatch, edgecolor='white',
                         label='Sent (solid)' if direction == 'Tx' else 'Received (hatched)')
                axis.hlines(model, position + offset - 0.2, position + offset + 0.2, color='black', linewidth=1.3)
        axis.set_xticks(range(len(cells)))
        short_names = {'ring': 'Ring', 'nvls': 'Classic\nNVLS', 'sym_multicast': 'Symmetric\nmulticast', 'sym_plain': 'Symmetric\nplain'}
        axis.set_xticklabels([f'{short_names[arm]}\n{gpu_count} GPUs' for gpu_count, arm in cells], fontsize=8)
        axis.set_ylim(0, top)
        axis.grid(alpha=0.3, axis='y')
        axis.set_title(COLLECTIVE_NAMES[collective], fontsize=10)
    axes[0].set_ylabel('NVLink bytes per payload byte, per GPU')
    figure.tight_layout()
    figure.text(0.5, -0.02, '\n'.join(textwrap.wrap('Solid: sent, hatched: received; median over GPUs of the median over 3 long-minus-short pairs. '
                'Black lines: model from the code (00_common_setup.md). Payload 2 GiB per GPU, CTAs chosen by NCCL.', 160)), ha='center', va='top', fontsize=8)
    figure.savefig(output, dpi=130, bbox_inches='tight')


def main():
    kind, source, output = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(source)
    {'calibration': figure_calibration, 'scaling': figure_scaling, 'payload': figure_payload, 'link': figure_link}[kind](rows, output)
    print(output)


if __name__ == '__main__':
    main()
