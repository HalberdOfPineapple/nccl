#!/usr/bin/env python3
"""Plan 06: summary CSV from run_custom_mc.py JSONL files, one row per (GPU count, kernel, CTA count, size).

Only valid runs count. The NCCL column chooses its own CTA count; its rows carry ctas = 'nccl' and the CTAs x
threads/CTA that NCCL logged. Rounds are runs; the timed calls inside a run are not independent repetitions.
Output CSV column names are read by make_figures_custom_mc.py: do not rename.

Usage: summarize_custom_mc.py OUTPUT_CSV INPUT_JSONL [INPUT_JSONL ...]
       summarize_custom_mc.py --flat SUMMARY_CSV GPU_COUNT   (prints the smallest flat CTA count per kernel at 2 GiB)
"""
import csv
import json
import math
from pathlib import Path
import statistics
import sys

THREADS_PER_CTA = {'oneshot': 1024, 'twoshot': 1024, 'poison': 1024}
FLAT_GAIN = 0.01
DISCARD_BELOW_BYTES = 24 * 2**20


def bus_factor(kernel, gpu_count):
    """Bytes each GPU sends on its links per payload byte (hypothesis until E6.3): one-shot N, the others 1 + 1/N."""
    if kernel == 'oneshot':
        return float(gpu_count)
    return 1 + 1 / gpu_count


def collect(paths):
    groups = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not record['valid']:
                continue
            for row in record['rows']:
                # 16 MiB is only the discarded first size of the E6.2 runs (two-shot sizes are rounded down to a multiple of N lines, hence the margin).
                if row['bytes'] < DISCARD_BELOW_BYTES:
                    continue
                # Two-shot sizes are rounded down to a multiple of N lines (at 6 GPUs, 2 GiB - 32 B); key rows by the requested power of two.
                requested_bytes = 2 ** round(math.log2(row['bytes']))
                key = (record['gpu_count'], record['devices'], row['kernel'], row['ctas'], requested_bytes, record['cell_id'].split('_c')[0].split('_')[0])
                group = groups.setdefault(key, {'algbw': [], 'shapes': set(), 'iters': set(), 'measured': set(), 'nccl_kernels': set()})
                group['measured'].add(row['bytes'])
                group['algbw'].append(row['algbw_gbps'])
                group['iters'].add(row['iters'])
                if row['kernel'] == 'nccl':
                    group['shapes'].update(record['nccl_ctas_x_threads'])
                    group['nccl_kernels'].update(record['nccl_kernels'])
    return groups


def summary_rows(groups):
    rows = []
    for (gpu_count, devices, kernel, ctas, size, step), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][5], item[0][2], str(item[0][3]).zfill(4), item[0][4])):
        values = group['algbw']
        median = statistics.median(values)
        shape = ','.join(sorted(group['shapes'])) if kernel == 'nccl' else f'{ctas}x{THREADS_PER_CTA[kernel]}'
        rows.append({'step': step, 'gpu_count': gpu_count, 'devices': devices, 'kernel': kernel, 'ctas': ctas,
                     'ctas_x_threads': shape, 'nccl_kernel': ','.join(sorted(group['nccl_kernels'])), 'size_bytes': size, 'measured_bytes': ','.join(str(n) for n in sorted(group['measured'])), 'timed_calls': ','.join(str(n) for n in sorted(group['iters'])),
                     'runs': len(values), 'algbw_median_gbps': round(median, 2), 'algbw_min_gbps': round(min(values), 2),
                     'algbw_max_gbps': round(max(values), 2), 'bus_factor': round(bus_factor(kernel, gpu_count), 4),
                     'busbw_median_gbps': round(median * bus_factor(kernel, gpu_count), 2)})
    return rows


def flat_counts(summary_csv, gpu_count, size=2 * 2**30):
    """Smallest CTA count after which two successive doublings each gain less than 1% (plan 01 rule), per kernel."""
    rows = [row for row in csv.DictReader(open(summary_csv)) if row['step'] == 'cta' and int(row['gpu_count']) == gpu_count
            and int(row['size_bytes']) == size and row['kernel'] != 'nccl']
    result = {}
    for kernel in ('oneshot', 'twoshot', 'poison'):
        curve = sorted((int(row['ctas']), float(row['algbw_median_gbps'])) for row in rows if row['kernel'] == kernel)
        flat = None
        for index in range(len(curve) - 2):
            gain_one = curve[index + 1][1] / curve[index][1] - 1
            gain_two = curve[index + 2][1] / curve[index + 1][1] - 1
            if gain_one < FLAT_GAIN and gain_two < FLAT_GAIN:
                flat = curve[index][0]
                break
        best = max(curve, key=lambda point: point[1]) if curve else None
        result[kernel] = {'flat_from': flat, 'best': best, 'curve': curve}
    return result


def main():
    if sys.argv[1] == '--flat-values':
        # One-shot, two-shot, poison CTA counts for E6.2; a curve still rising at the last count uses that count.
        counts = flat_counts(sys.argv[2], int(sys.argv[3]))
        print(' '.join(str(info['flat_from'] or info['curve'][-1][0]) for info in counts.values()))
        return
    if sys.argv[1] == '--flat':
        for kernel, info in flat_counts(sys.argv[2], int(sys.argv[3])).items():
            print(kernel, 'flat from', info['flat_from'], 'best', info['best'], 'curve', info['curve'])
        return
    output = Path(sys.argv[1])
    rows = summary_rows(collect(sys.argv[2:]))
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} rows -> {output}')


if __name__ == '__main__':
    main()
