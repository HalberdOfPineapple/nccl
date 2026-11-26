#!/usr/bin/env python3
"""Plan 06 figures, made only from saved summary CSVs (summarize_custom_mc.py, link_bytes_custom_mc.py).

Usage (matplotlib through uv):
  uv run --no-project --with matplotlib python make_figures_custom_mc.py cta SUMMARY_CSV OUTPUT_PNG
  uv run --no-project --with matplotlib python make_figures_custom_mc.py payload SUMMARY_CSV OUTPUT_PNG
  uv run --no-project --with matplotlib python make_figures_custom_mc.py gpus SUMMARY_CSV OUTPUT_PNG
  uv run --no-project --with matplotlib python make_figures_custom_mc.py link LINK_CSV OUTPUT_PNG

Each kernel has one fixed colour in every figure; NCCL's library kernel keeps the green of symmetric multicast in
the other plans. Rate axes start at zero and panels of one figure share a scale.
"""
import csv
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KERNEL_NAMES = {'oneshot': 'One-shot (example)', 'twoshot': 'Two-shot, end barrier (example)',
                'poison': 'Two-shot, poison (example)', 'nccl': 'NCCL library, RSxLDMC_AGxSTMC'}
KERNEL_COLOURS = {'oneshot': '#d62728', 'twoshot': '#8c564b', 'poison': '#e377c2', 'nccl': '#2ca02c'}
KERNEL_ORDER = ['oneshot', 'twoshot', 'poison', 'nccl']
GPU_SETS = {2: '{4,5}', 4: '{4-7}', 6: '{1-6}', 8: '{0-7}'}
SIZE_2_GIB = 2 * 2**30
SIZE_32_MIB = 32 * 2**20


def caption(figure, text, y=-0.03):
    """Caption under the figure, wrapped so it is never wider than the panels."""
    width = max(60, int(figure.get_figwidth() * 17))
    lines = textwrap.wrap(text, width)
    figure.text(0.5, y, '\n'.join(lines), ha='center', va='top', fontsize=7.5)


def bottom_legend(figure, axis, panel_count):
    """One legend under all panels, so it never covers data or squeezes a panel."""
    handles, labels = axis.get_legend_handles_labels()
    legend = figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2 if panel_count == 1 else 4, fontsize=7, frameon=False)
    figure.canvas.draw()
    # Figure-coordinate y of the legend's lower edge, where the caption starts.
    return legend.get_window_extent().transformed(figure.transFigure.inverted()).y0 - 0.02


def read_rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def rounded_top(value):
    step = 50 if value < 400 else 100
    return (int(value / step) + 1) * step


def nccl_shape_text(rows, gpu_count):
    shapes = sorted({row['ctas_x_threads'] for row in rows if row['kernel'] == 'nccl' and int(row['gpu_count']) == gpu_count})
    return '/'.join(shapes) if shapes else 'n/a'


def figure_cta(rows, output):
    """E6.1: algBW against CTAs for the example kernels; NCCL's own choice as a star."""
    rows = [row for row in rows if row['step'] == 'cta']
    gpu_counts = sorted({int(row['gpu_count']) for row in rows})
    top = rounded_top(max(float(row['algbw_median_gbps']) for row in rows))
    figure, axes = plt.subplots(1, len(gpu_counts), figsize=(5.0 * len(gpu_counts), 4.2), squeeze=False, sharey=True)
    for axis, gpu_count in zip(axes[0], gpu_counts):
        for kernel in ('oneshot', 'twoshot', 'poison'):
            for size, style in ((SIZE_2_GIB, '-'), (SIZE_32_MIB, '--')):
                points = sorted((int(row['ctas']), row) for row in rows if row['kernel'] == kernel and int(row['gpu_count']) == gpu_count and int(row['size_bytes']) == size)
                if not points:
                    continue
                xs = [point[0] for point in points]
                ys = [float(point[1]['algbw_median_gbps']) for point in points]
                lows = [y - float(point[1]['algbw_min_gbps']) for y, point in zip(ys, points)]
                highs = [float(point[1]['algbw_max_gbps']) - y for y, point in zip(ys, points)]
                label = f"{KERNEL_NAMES[kernel]}, {'2 GiB' if size == SIZE_2_GIB else '32 MiB'}"
                axis.errorbar(xs, ys, yerr=[lows, highs], linestyle=style, color=KERNEL_COLOURS[kernel], marker='o', markersize=3, linewidth=1.2, capsize=2, label=label)
        for size, marker in ((SIZE_2_GIB, '*'), (SIZE_32_MIB, 'P')):
            nccl = [row for row in rows if row['kernel'] == 'nccl' and int(row['gpu_count']) == gpu_count and int(row['size_bytes']) == size]
            if nccl:
                ctas = int(nccl[0]['ctas_x_threads'].split('x')[0])
                values = sorted(float(row['algbw_median_gbps']) for row in nccl)
                median = values[len(values) // 2]
                axis.scatter([ctas], [median], marker=marker, s=120, color=KERNEL_COLOURS['nccl'], edgecolor='black', linewidth=0.6, zorder=5,
                             label=f"NCCL library, its own CTA choice, {'2 GiB' if size == SIZE_2_GIB else '32 MiB'}")
                if size == SIZE_2_GIB:
                    axis.annotate(f"NCCL: {nccl[0]['nccl_kernel'].replace('AllReduce_', '')}\n{nccl[0]['ctas_x_threads'].replace('x', ' x ')}",
                                  (ctas, median), textcoords='offset points', xytext=(8, 6), fontsize=7, color='#1b5e20')
        axis.set_xscale('log', base=2)
        axis.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
        axis.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128', '256'])
        axis.set_xlabel('CTAs (example kernels: x 1024 threads/CTA)')
        axis.set_ylim(0, top)
        axis.grid(alpha=0.3)
        axis.set_title(f'{gpu_count} GPUs {GPU_SETS[gpu_count]}')
    axes[0][0].set_ylabel('algBW, GB/s')
    legend_bottom = bottom_legend(figure, axes[0][-1], len(axes[0]))
    caption(figure, 'Lines: median of 3 runs, bars: min-max. Solid 2 GiB, dashed 32 MiB per GPU. NCCL markers: its own CTA choice, '
                'median over the runs of the sweep. Timing: CUDA graph of 10 calls, largest time over ranks.', legend_bottom)
    figure.tight_layout()
    figure.savefig(output, dpi=130, bbox_inches='tight')


def figure_payload(rows, output):
    """E6.2: algBW against payload per GPU, each kernel at its flat CTA count; CTA shapes written inside each panel."""
    rows = [row for row in rows if row['step'] == 'flat']
    gpu_counts = sorted({int(row['gpu_count']) for row in rows})
    top = rounded_top(max(float(row['algbw_median_gbps']) for row in rows))
    figure, axes = plt.subplots(1, len(gpu_counts), figsize=(4.8 * len(gpu_counts), 4.2), squeeze=False, sharey=True)
    legend_entries = {}
    for axis, gpu_count in zip(axes[0], gpu_counts):
        shape_lines = []
        for kernel in KERNEL_ORDER:
            selected = [row for row in rows if row['kernel'] == kernel and int(row['gpu_count']) == gpu_count]
            if not selected:
                continue
            # The NCCL column ran in every E6.2 run of this GPU count; pool its rows per size (one curve).
            sizes = sorted({int(row['size_bytes']) for row in selected})
            ys = []
            for size in sizes:
                values = sorted(float(row['algbw_median_gbps']) for row in selected if int(row['size_bytes']) == size)
                ys.append(values[len(values) // 2])
            xs = [size / 2**20 for size in sizes]
            if kernel == 'nccl' and 'RSxLDMC' not in selected[0]['nccl_kernel']:
                name = f"NCCL library, {selected[0]['nccl_kernel'].replace('AllReduce_', '')} (NCCL's choice at 2 GPUs)"
                handle, = axis.plot(xs, ys, color=KERNEL_COLOURS[kernel], marker='o', markersize=4, markerfacecolor='white', linestyle='--', linewidth=1.2, label=name)
            else:
                name = KERNEL_NAMES[kernel]
                handle, = axis.plot(xs, ys, color=KERNEL_COLOURS[kernel], marker='o', markersize=3, linewidth=1.3, label=name)
            legend_entries.setdefault(name, handle)
            short_name = {'oneshot': 'one-shot', 'twoshot': 'two-shot', 'poison': 'poison', 'nccl': 'NCCL'}[kernel]
            shape_lines.append(f"{short_name} {selected[0]['ctas_x_threads'].replace('x', ' x ')}")
        axis.text(0.97, 0.34, 'CTAs x threads/CTA:\n' + '\n'.join(shape_lines), transform=axis.transAxes, ha='right', va='bottom', fontsize=7,
                  bbox={'facecolor': 'white', 'edgecolor': '#cccccc', 'boxstyle': 'round,pad=0.3'})
        axis.set_xscale('log', base=2)
        axis.set_xticks([32, 64, 128, 256, 512, 1024, 2048])
        axis.set_xticklabels(['32M', '64M', '128M', '256M', '512M', '1G', '2G'])
        axis.set_xlabel('Payload per GPU, bytes')
        axis.set_ylim(0, top)
        axis.grid(alpha=0.3)
        axis.set_title(f'{gpu_count} GPUs {GPU_SETS[gpu_count]}')
    axes[0][0].set_ylabel('algBW, GB/s')
    legend = figure.legend(list(legend_entries.values()), list(legend_entries.keys()), loc='upper center', bbox_to_anchor=(0.5, 0.0),
                           ncol=3, fontsize=7, frameon=False)
    figure.canvas.draw()
    legend_bottom = legend.get_window_extent().transformed(figure.transFigure.inverted()).y0 - 0.02
    caption(figure, 'Median of 5 runs. Example kernels at their flat CTA count from the CTA sweep. '
                    'Timing: CUDA graph of 10 calls, largest time over ranks.', legend_bottom)
    figure.tight_layout()
    figure.savefig(output, dpi=130, bbox_inches='tight')


def figure_gpus(rows, output):
    """E6.2: algBW against GPU count at 32 MiB and 2 GiB."""
    rows = [row for row in rows if row['step'] == 'flat']
    gpu_counts = sorted({int(row['gpu_count']) for row in rows})
    top = rounded_top(max(float(row['algbw_median_gbps']) for row in rows))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for axis, size in zip(axes, (SIZE_32_MIB, SIZE_2_GIB)):
        for kernel in KERNEL_ORDER:
            xs = []
            ys = []
            for gpu_count in gpu_counts:
                matching = [row for row in rows if row['kernel'] == kernel and int(row['gpu_count']) == gpu_count and int(row['size_bytes']) == size]
                values = sorted(float(row['algbw_median_gbps']) for row in matching)
                if kernel == 'nccl' and values and 'RSxLDMC' not in matching[0]['nccl_kernel']:
                    # NCCL ran another kernel here (plain load/store at 2 GPUs): its own marker, not joined to the multicast line.
                    axis.scatter([gpu_count], [values[len(values) // 2]], marker='o', s=36, facecolor='white', edgecolor=KERNEL_COLOURS['nccl'], linewidth=1.3, zorder=5,
                                 label=f"NCCL library, {matching[0]['nccl_kernel'].replace('AllReduce_', '')} (NCCL's choice at {gpu_count} GPUs)")
                    continue
                if values:
                    xs.append(gpu_count)
                    ys.append(values[len(values) // 2])
            if xs:
                axis.plot(xs, ys, color=KERNEL_COLOURS[kernel], marker='o', markersize=4, linewidth=1.3, label=KERNEL_NAMES[kernel])
        axis.set_xticks(gpu_counts)
        axis.set_xlabel('GPU count')
        axis.set_ylim(0, top)
        axis.grid(alpha=0.3)
        axis.set_title(f"Payload {'32 MiB' if size == SIZE_32_MIB else '2 GiB'} per GPU")
    axes[0].set_ylabel('algBW, GB/s')
    axes[0].legend(fontsize=7, loc='lower left')
    shapes = '; '.join(f'{n}: ' + ', '.join(f"{kernel} {sorted({row['ctas_x_threads'] for row in rows if row['kernel'] == kernel and int(row['gpu_count']) == n})[0]}"
                                           for kernel in KERNEL_ORDER if any(row['kernel'] == kernel and int(row['gpu_count']) == n for row in rows))
                       for n in gpu_counts)
    sets = ', '.join(f'{n} = {GPU_SETS[n]}' for n in gpu_counts)
    caption(figure, f'Median of 5 runs. GPUs: {sets}. CTAs x threads/CTA by GPU count: {shapes}.', -0.06)
    figure.tight_layout()
    figure.savefig(output, dpi=130, bbox_inches='tight')


def figure_link(rows, output):
    """E6.3: link bytes per payload byte, sent and received, against the byte-count models."""
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    labels = []
    sent = []
    received = []
    models = []
    for row in rows:
        labels.append(f"{KERNEL_NAMES[row['kernel']].split(' (')[0].split(',')[0]}\n{row['gpu_count']} GPUs, {row['ctas_x_threads']}")
        sent.append(float(row['sent_per_payload_byte_median']))
        received.append(float(row['received_per_payload_byte_median']))
        models.append((float(row['model_sent']), float(row['model_received'])))
    positions = range(len(labels))
    axis.bar([p - 0.2 for p in positions], sent, width=0.4, color='#4c72b0', label='Sent (Tx), median over GPUs and rounds')
    axis.bar([p + 0.2 for p in positions], received, width=0.4, color='#dd8452', label='Received (Rx)')
    for position, (model_sent, model_received) in zip(positions, models):
        axis.hlines(model_sent, position - 0.4, position, color='black', linewidth=1.2)
        axis.hlines(model_received, position, position + 0.4, color='black', linewidth=1.2, linestyle='--')
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel('NVLink bytes per payload byte, per GPU')
    axis.set_ylim(0, max(max(sent), max(received), max(max(m) for m in models)) * 1.15)
    axis.grid(alpha=0.3, axis='y')
    axis.legend(fontsize=7, loc='upper left')
    caption(figure, 'Black lines: model (one-shot sends N, receives 1; two-shot 1 + 1/N each way). Payload 2 GiB per GPU; '
                'long (100 timed calls) minus short (10) runs, 3 rounds.', -0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=130, bbox_inches='tight')


def main():
    kind, source, output = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(source)
    {'cta': figure_cta, 'payload': figure_payload, 'gpus': figure_gpus, 'link': figure_link}[kind](rows, output)
    print(output)


if __name__ == '__main__':
    main()
