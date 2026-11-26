#!/usr/bin/env python3
"""busBW and algBW of AllReduce, AllGather and ReduceScatter for Ring and symmetric multicast, from saved summary CSVs only.

The algBW figure adds "ReduceScatter + AllGather": the rate of a ReduceScatter followed by an AllGather of the same size S,
1 / (1/algBW_RS + 1/algBW_AG), which does the work of one AllReduce of size S.

busBW = algBW x bytes in the busiest link direction per payload byte (summarize.bus_factor), which the link counters
confirmed for every cell used here (AllReduce plan 04; AllGather/ReduceScatter plan 01). It is therefore the link rate
in the busiest direction, comparable across collectives.

Sources (all relative to experiments/h100-scale/):
  AllReduce, 2 GiB, NCCL's CTA choice:      allreduce/scalability/gpu_count_scaling/results/2026-11-16/scaling_summary.csv (5 runs)
  AllReduce, 2 GiB, Ring 64 CTAs:           allreduce/scalability/selection_calibration/results/2026-11-16/calibration_summary.csv (3 runs)
  AllReduce, 32 MiB, NCCL's CTA choice:     allreduce/scalability/payload_size/results/2026-11-16/sweep_summary.csv (5 runs)
  AllGather, ReduceScatter, 2 GiB:          allgather_reducescatter/scalability/symmetric_multicast/results/2026-11-18/scaling_summary.csv (5 runs)
  AllGather, ReduceScatter, 32 MiB:         .../payload_summary.csv (5 runs)

Usage (matplotlib through uv), from experiments/h100-scale/:
  uv run --no-project --with matplotlib python allreduce/common/scripts/compare_busbw.py OUTPUT_CSV BUSBW_PNG ALGBW_PNG [EXPECTATION_PNG]
Output CSV column names are read by nothing else yet; keep them stable once a report cites them.
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from summarize import bus_factor

ALLREDUCE = Path('allreduce/scalability')
AGRS = Path('allgather_reducescatter/scalability/symmetric_multicast/results/2026-11-18')
SIZE_2_GIB = 2 * 2**30
SIZE_32_MIB = 32 * 2**20
COLLECTIVES = ['allreduce', 'allgather', 'reducescatter']
COLLECTIVE_NAMES = {'allreduce': 'AllReduce', 'allgather': 'AllGather', 'reducescatter': 'ReduceScatter'}
COLLECTIVE_COLOURS = {'allreduce': '#1f77b4', 'allgather': '#d95f02', 'reducescatter': '#7570b3'}
SETTINGS = [('ring', 'auto'), ('ring', '64'), ('sym_multicast', 'auto')]
SETTING_NAMES = {('ring', 'auto'): "Ring, NCCL's CTA choice (24 x 512)", ('ring', '64'): 'Ring, 64 x 512 fixed',
                 ('sym_multicast', 'auto'): "Symmetric multicast, NCCL's CTA choice"}
GPU_SETS = {2: '{4,5}', 4: '{4-7}', 6: '{1-6}', 8: '{0-7}'}


def read(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def nearest_power_of_two(size):
    """nccl-tests rounds AllGather/ReduceScatter pieces to 16 B, so 6-GPU sizes sit just below powers of two."""
    return 2 ** round(math.log2(size))


def collect():
    """One row per (collective, arm, CTA setting, size, GPU count)."""
    sources = [
        ('allreduce', ALLREDUCE / 'gpu_count_scaling/results/2026-11-16/scaling_summary.csv', {'auto'}, 'AllReduce plan 02, 2026-11-16'),
        ('allreduce', ALLREDUCE / 'selection_calibration/results/2026-11-16/calibration_summary.csv', {'64'}, 'AllReduce plan 01 calibration, 2026-11-16'),
        ('allreduce', ALLREDUCE / 'payload_size/results/2026-11-16/sweep_summary.csv', {'auto'}, 'AllReduce plan 03, 2026-11-16'),
        (None, AGRS / 'scaling_summary.csv', {'auto', '64'}, 'AllGather/ReduceScatter plan 01 scaling, 2026-11-18'),
        (None, AGRS / 'payload_summary.csv', {'auto', '64'}, 'AllGather/ReduceScatter plan 01 payload, 2026-11-18'),
    ]
    rows = {}
    for fixed_collective, path, settings, source_name in sources:
        for row in read(path):
            if (row['arm'], row['requested_ctas']) not in SETTINGS or row['requested_ctas'] not in settings:
                continue
            size = nearest_power_of_two(int(row['size_bytes']))
            if size not in (SIZE_2_GIB, SIZE_32_MIB):
                continue
            collective = fixed_collective or row['collective']
            gpu_count = int(row['gpu_count'])
            key = (collective, row['arm'], row['requested_ctas'], size, gpu_count)
            if key in rows:
                # The scaling (2 GiB) source comes first and has 5 runs; the payload sweep repeats 2 GiB with its own buffers.
                continue
            factor = bus_factor(row['arm'], gpu_count, collective)
            algbw = float(row['algbw_median_gbps'])
            rows[key] = {'collective': collective, 'arm': row['arm'], 'requested_ctas': row['requested_ctas'], 'ctas_used': row['ctas_used'],
                         'threads_per_cta': row['threads_per_cta'], 'size_bytes': size, 'gpu_count': gpu_count, 'devices': row['devices'],
                         'runs': row['runs'], 'algbw_median_gbps': algbw, 'bytes_busiest_direction_per_payload_byte': round(factor, 4),
                         'busbw_median_gbps': round(algbw * factor, 1), 'source': source_name}
    return [rows[key] for key in sorted(rows, key=lambda key: (COLLECTIVES.index(key[0]), key[1], key[2], key[3], key[4]))]


def write_csv(rows, output):
    with open(output, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure(rows, output):
    """Rows of panels: 2 GiB, 32 MiB. Columns: Ring and symmetric multicast. One line per collective."""
    figure_handle, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for row_index, size in enumerate((SIZE_2_GIB, SIZE_32_MIB)):
        for column_index, arm in enumerate(('ring', 'sym_multicast')):
            axis = axes[row_index][column_index]
            for collective in COLLECTIVES:
                for setting, style, marker in (('auto', '-', 'o'), ('64', ':', '^')):
                    if (arm, setting) not in SETTINGS:
                        continue
                    series = sorted((row['gpu_count'], row['busbw_median_gbps']) for row in rows if row['collective'] == collective
                                    and row['arm'] == arm and row['requested_ctas'] == setting and row['size_bytes'] == size)
                    if series:
                        label = f"{COLLECTIVE_NAMES[collective]}, {'64 CTAs fixed' if setting == '64' else 'NCCL CTA choice'}"
                        axis.plot([item[0] for item in series], [item[1] for item in series], color=COLLECTIVE_COLOURS[collective],
                                  linestyle=style, marker=marker, markersize=4, linewidth=1.3, label=label)
            axis.axhline(478, color='grey', linewidth=0.8, linestyle=':')
            axis.set_xticks([2, 4, 6, 8])
            axis.set_ylim(0, 500)
            axis.grid(alpha=0.3)
            arm_name = 'Ring' if arm == 'ring' else 'Symmetric multicast'
            axis.set_title(f"{arm_name}, {'2 GiB' if size == SIZE_2_GIB else '32 MiB'} per GPU", fontsize=10)
            if row_index == 1:
                axis.set_xlabel('GPU count')
            if column_index == 0:
                axis.set_ylabel('busBW = link rate in busiest direction, GB/s')
            axis.legend(fontsize=7, loc='lower right')
    axes[0][0].text(2.05, 470, 'NVLink rating, 478 GB/s per direction', fontsize=7, color='grey', va='top')
    figure_handle.tight_layout()
    caption = ('busBW factors (bytes in the busiest direction per payload byte, confirmed by link counters): AllReduce Ring 2(N-1)/N, multicast 1+1/N; '
               'AllGather and ReduceScatter Ring (N-1)/N, multicast 1. Median of 5 runs, except AllReduce Ring at 64 CTAs (3 runs, calibration). '
               'Multicast CTAs: 4 x 512 at 4 GPUs, 2 x 512 at 6 and 8; no multicast at 2 GPUs. AllReduce runs 2026-11-16, others 2026-11-18. '
               'GPUs: 2 = {4,5}, 4 = {4-7}, 6 = {1-6}, 8 = {0-7}.')
    import textwrap
    figure_handle.text(0.5, -0.01, '\n'.join(textwrap.wrap(caption, 170)), ha='center', va='top', fontsize=8)
    figure_handle.savefig(output, dpi=130, bbox_inches='tight')


def composite_rate(rows, arm, setting, size, gpu_count):
    """ReduceScatter then AllGather of the same size: 1 / (1/algBW_RS + 1/algBW_AG), or None if either is missing."""
    rates = {}
    for row in rows:
        if row['arm'] == arm and row['requested_ctas'] == setting and row['size_bytes'] == size and row['gpu_count'] == gpu_count:
            rates[row['collective']] = row['algbw_median_gbps']
    if 'reducescatter' not in rates or 'allgather' not in rates:
        return None
    return 1 / (1 / rates['reducescatter'] + 1 / rates['allgather'])


def figure_algbw(rows, output):
    """Same layout as the busBW figure; algBW, plus the ReduceScatter + AllGather composite (black dashed)."""
    import textwrap
    figure_handle, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for row_index, size in enumerate((SIZE_2_GIB, SIZE_32_MIB)):
        for column_index, arm in enumerate(('ring', 'sym_multicast')):
            axis = axes[row_index][column_index]
            for setting, style, marker in (('auto', '-', 'o'), ('64', ':', '^')):
                if (arm, setting) not in SETTINGS:
                    continue
                setting_text = '64 CTAs fixed' if setting == '64' else 'NCCL CTA choice'
                for collective in COLLECTIVES:
                    series = sorted((row['gpu_count'], row['algbw_median_gbps']) for row in rows if row['collective'] == collective
                                    and row['arm'] == arm and row['requested_ctas'] == setting and row['size_bytes'] == size)
                    if series:
                        axis.plot([item[0] for item in series], [item[1] for item in series], color=COLLECTIVE_COLOURS[collective],
                                  linestyle=style, marker=marker, markersize=4, linewidth=1.3, label=f'{COLLECTIVE_NAMES[collective]}, {setting_text}')
                composite = [(gpu_count, composite_rate(rows, arm, setting, size, gpu_count)) for gpu_count in (2, 4, 6, 8)]
                composite = [item for item in composite if item[1] is not None]
                if composite:
                    axis.plot([item[0] for item in composite], [item[1] for item in composite], color='black', linestyle='--' if setting == 'auto' else (0, (1, 3)),
                              marker='x', markersize=5, linewidth=1.1, label=f'ReduceScatter + AllGather, {setting_text}')
            axis.set_xticks([2, 4, 6, 8])
            axis.set_ylim(0, 900)
            axis.grid(alpha=0.3)
            arm_name = 'Ring' if arm == 'ring' else 'Symmetric multicast'
            axis.set_title(f"{arm_name}, {'2 GiB' if size == SIZE_2_GIB else '32 MiB'} per GPU", fontsize=10)
            if row_index == 1:
                axis.set_xlabel('GPU count')
            if column_index == 0:
                axis.set_ylabel('algBW = size / time of one call, GB/s')
            axis.legend(fontsize=6.5, loc='upper right')
    figure_handle.tight_layout()
    caption = ('Size = nccl-tests size: AllReduce buffer of one GPU, AllGather output, ReduceScatter input. Black: a ReduceScatter followed by an '
               'AllGather of the same size, 1/(1/algBW_RS + 1/algBW_AG), the same work as one AllReduce. Median of 5 runs, except AllReduce Ring at '
               '64 CTAs (3 runs, calibration). Multicast CTAs: 4 x 512 at 4 GPUs, 2 x 512 at 6 and 8; no multicast at 2 GPUs. AllReduce runs '
               '2026-11-16, others 2026-11-18. GPUs: 2 = {4,5}, 4 = {4-7}, 6 = {1-6}, 8 = {0-7}.')
    figure_handle.text(0.5, -0.01, '\n'.join(textwrap.wrap(caption, 170)), ha='center', va='top', fontsize=8)
    figure_handle.savefig(output, dpi=130, bbox_inches='tight')


def best_reference(gpu_count):
    """Highest busBW of any AllGather/ReduceScatter arm and CTA setting at 2 GiB on these GPUs, and which one it is."""
    best = None
    for row in read(AGRS / 'scaling_summary.csv'):
        if int(row['gpu_count']) != gpu_count:
            continue
        value = float(row['busbw_median_gbps'])
        if best is None or value > best[0]:
            best = (value, f"{row['collective']} {row['arm']} {row['requested_ctas']}")
    return best


def figure_expectation(rows, output):
    """Expectation for AllReduce on a full-duplex link: the best per-direction rate of any AllGather/ReduceScatter setting on the
    same GPUs. Measured AllReduce busBW at 2 GiB for Ring (NCCL's CTA choice, 64 CTAs) and multicast."""
    import textwrap
    figure_handle, axis = plt.subplots(figsize=(7.5, 4.6))
    gpu_counts = [2, 4, 6, 8]
    axis.plot(gpu_counts, [best_reference(n)[0] for n in gpu_counts], color='black', linestyle='--', marker='D', markersize=4, linewidth=1.2,
              label='Expectation: best AllGather/ReduceScatter busBW on the same GPUs')
    for arm, setting, colour, style, label in (('ring', '64', '#1f77b4', ':', 'Ring AllReduce, 64 x 512'),
                                               ('ring', 'auto', '#1f77b4', '-', "Ring AllReduce, NCCL's 24 x 512"),
                                               ('sym_multicast', 'auto', '#2ca02c', '-', "Multicast AllReduce, NCCL's CTA choice")):
        series = sorted((row['gpu_count'], row['busbw_median_gbps']) for row in rows if row['collective'] == 'allreduce' and row['arm'] == arm
                        and row['requested_ctas'] == setting and row['size_bytes'] == SIZE_2_GIB)
        axis.plot([item[0] for item in series], [item[1] for item in series], color=colour, linestyle=style, marker='o', markersize=4, linewidth=1.3, label=label)
    axis.axhline(478, color='grey', linewidth=0.8, linestyle=':')
    axis.text(2.05, 470, 'NVLink rating, 478 GB/s per direction', fontsize=7, color='grey', va='top')
    axis.set_xticks(gpu_counts)
    axis.set_ylim(0, 500)
    axis.set_xlabel('GPU count')
    axis.set_ylabel('busBW, GB/s (busiest direction)')
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7.5, loc='lower right')
    axis.set_title('AllReduce busBW against the full-duplex expectation, 2 GiB per GPU', fontsize=10)
    figure_handle.tight_layout()
    caption = ('Expectation: highest busBW of any AllGather or ReduceScatter arm and CTA setting at 2 GiB on the same GPUs (2: classic NVLS AllGather '
               '64 CTAs; 4, 8: multicast AllGather; 6: classic NVLS AllGather 64 CTAs). Medians of 5 runs; Ring AllReduce at 64 CTAs: 3 runs. '
               'GPUs: 2 = {4,5}, 4 = {4-7}, 6 = {1-6}, 8 = {0-7}.')
    figure_handle.text(0.5, -0.01, '\n'.join(textwrap.wrap(caption, 120)), ha='center', va='top', fontsize=7.5)
    figure_handle.savefig(output, dpi=130, bbox_inches='tight')


def main():
    rows = collect()
    write_csv(rows, sys.argv[1])
    figure(rows, sys.argv[2])
    figure_algbw(rows, sys.argv[3])
    if len(sys.argv) > 4:
        figure_expectation(rows, sys.argv[4])
        for gpu_count in (2, 4, 6, 8):
            print('reference', gpu_count, best_reference(gpu_count))
    print(f'{len(rows)} rows -> {sys.argv[1]}; figures -> {sys.argv[2]}, {sys.argv[3]}')


if __name__ == '__main__':
    main()
