#!/usr/bin/env python3
"""Payload report, mechanism section: classic NVLS pipeline steps and excess time per call, from the saved sweep summary only.

1. Excess time per call at payload S: T(S) - S / algBW(8 GiB), where T(S) = S / algBW(S). It is the time a call takes beyond what
   the arm's own large-payload rate would need for S bytes.
2. Classic NVLS steps per channel: each of its 16 channels moves its share S/16 in steps of nHeads x chunk = N x 128 KiB
   (all_reduce.h: loopCount = nHeads * chunkSize; NCCL_NVLS_CHUNKSIZE default 128 KiB, logged "chunkSize 131072"), so
   k = S / (16 x N x 128 KiB). A pipeline of 3 teams (scatter, reduce, gather) with k steps takes about k + 2 step times,
   so the expected share of the large-payload rate is k / (k + 2). This model uses only the code's step size; the measured
   ratio algBW(S) / algBW(8 GiB) is compared with it.

Usage (matplotlib through uv), from experiments/h100-scale/allreduce/scalability/payload_size/:
  uv run --no-project --with matplotlib python ../../common/scripts/payload_mechanisms.py SWEEP_SUMMARY_CSV OUTPUT_CSV OUTPUT_PNG
Output CSV column names are cited by the report: do not rename.
"""
import csv
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MIB = 1 << 20
LARGE = 8192 * MIB
NVLS_CHANNELS = 16
NVLS_CHUNK_BYTES = 128 * 1024
ARM_NAMES = {'ring': 'Ring', 'nvls': 'Classic NVLS', 'sym_multicast': 'Symmetric multicast', 'sym_plain': 'Symmetric plain load/store'}
GPU_MARKERS = {2: 'o', 4: 's', 6: '^', 8: 'D'}


def main():
    source, output_csv, output_png = sys.argv[1], sys.argv[2], sys.argv[3]
    rates = {(row['arm'], int(row['gpu_count']), int(row['size_bytes'])): float(row['algbw_median_gbps']) for row in csv.DictReader(open(source))}
    rows = []
    for (arm, gpu_count, size), rate in sorted(rates.items()):
        if (arm, gpu_count, LARGE) not in rates or size >= LARGE:
            continue
        large_rate = rates[(arm, gpu_count, LARGE)]
        time_us = size / (rate * 1e3)
        ideal_us = size / (large_rate * 1e3)
        row = {'arm': arm, 'gpu_count': gpu_count, 'size_bytes': size, 'algbw_gbps': rate, 'algbw_8gib_gbps': large_rate,
               'ratio_to_own_8gib': round(rate / large_rate, 3), 'time_us': round(time_us, 1), 'excess_time_us': round(time_us - ideal_us, 1),
               'nvls_steps_per_channel': '', 'nvls_model_ratio': ''}
        if arm == 'nvls':
            steps = size / (NVLS_CHANNELS * gpu_count * NVLS_CHUNK_BYTES)
            row['nvls_steps_per_channel'] = round(steps, 2)
            row['nvls_model_ratio'] = round(steps / (steps + 2), 3)
        rows.append(row)
    with open(output_csv, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))
    nvls = [row for row in rows if row['arm'] == 'nvls']
    for gpu_count, marker in GPU_MARKERS.items():
        points = [row for row in nvls if row['gpu_count'] == gpu_count]
        left.scatter([row['nvls_steps_per_channel'] for row in points], [row['ratio_to_own_8gib'] for row in points], marker=marker,
                     s=28, color='#ff7f0e', edgecolor='black', linewidth=0.4, label=f'Measured, {gpu_count} GPUs')
    steps_axis = [0.5 * 1.15 ** i for i in range(60)]
    left.plot(steps_axis, [k / (k + 2) for k in steps_axis], color='black', linewidth=1.1, label='Model k / (k + 2), from the code step size')
    left.set_xscale('log', base=2)
    left.set_xlim(1, 2048)
    left.set_ylim(0, 1.05)
    left.set_xlabel('Pipeline steps per channel, k = S / (16 x N x 128 KiB)')
    left.set_ylabel('algBW(S) / algBW(8 GiB), same arm')
    left.set_title('Classic NVLS: share of its large-payload rate', fontsize=10)
    left.grid(alpha=0.3)
    left.legend(fontsize=7.5, loc='lower right')

    for arm, colour in (('ring', '#1f77b4'), ('nvls', '#ff7f0e'), ('sym_multicast', '#2ca02c'), ('sym_plain', '#9467bd')):
        points = sorted((row['gpu_count'], row['excess_time_us']) for row in rows if row['arm'] == arm and row['size_bytes'] == 32 * MIB)
        right.plot([item[0] for item in points], [item[1] for item in points], color=colour, marker='o', markersize=4, linewidth=1.3, label=ARM_NAMES[arm])
    right.set_xticks([2, 4, 6, 8])
    right.set_ylim(0, 100)
    right.set_xlabel('GPU count')
    right.set_ylabel('Excess time per call at 32 MiB, microseconds')
    right.set_title('Time beyond the arm\'s own 8 GiB rate, 32 MiB per GPU', fontsize=10)
    right.grid(alpha=0.3)
    right.legend(fontsize=7.5, loc='upper left')
    figure.tight_layout()
    caption = ('Left: each point is one GPU count and payload (32 MiB to 4 GiB); CTAs chosen by NCCL (16 x 640). Right: excess time = '
               'S / algBW(S) - S / algBW(8 GiB). Median of 5 runs, 2026-11-16 sweep. GPUs: 2 = {4,5}, 4 = {4-7}, 6 = {1-6}, 8 = {0-7}.')
    figure.text(0.5, -0.01, '\n'.join(textwrap.wrap(caption, 170)), ha='center', va='top', fontsize=8)
    figure.savefig(output_png, dpi=130, bbox_inches='tight')
    print(f'{len(rows)} rows -> {output_csv}; figure -> {output_png}')


if __name__ == '__main__':
    main()
