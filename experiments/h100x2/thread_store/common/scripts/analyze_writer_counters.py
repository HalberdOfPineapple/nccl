#!/usr/bin/env python3
"""Requested writer bytes against DRAM and L2 hardware counters (run_writer_counters.py).

Per launch: counted bytes ÷ requested bytes. Counters and the harness record come from different
replay passes of the same program, so a small difference can be run-to-run variation; the tables give
the range over launches and runs. L2 sectors are 32 bytes.
"""
import csv
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_style import set_zero_based_y
from run_store_congest import PEAK_GBPS, RESULTS, ROUNDS
from run_writer_counters import GROUP, settings

RAW = RESULTS / GROUP / 'raw'
SECTOR_BYTES = 32
COLORS = {'gpc': '#b95f0d', 'fabric': '#7b6fb0', 'other': '#8e98a7'}


def counters_by_launch(path):
    launches = {}
    for row in csv.reader(path.open()):
        if len(row) > 14 and row[4].startswith('storeCongest'):
            launches.setdefault(int(row[0]), {})[row[12]] = float(row[14].replace(',', '')) if row[14] != 'n/a' else None
    return [launches[launch] for launch in sorted(launches)]


def launches_of_run(path):
    """Retained launches of one run: requested bytes and rate from the harness, counters from ncu."""
    data = json.loads(path.read_text())
    retained = counters_by_launch(path.with_suffix('.ncu.csv'))[-len(data['ranks'][0]['iterations']):]
    rows = []
    for records, counted in zip(data['ranks'][0]['iterations'], retained):
        writers = [record for record in records if record['role'] == 'writer']
        requested = sum(record['work_units'] for record in writers) * data['writer_block_bytes']
        active_ns = max(record['end_ns'] for record in writers) - min(record['start_ns'] for record in writers)
        duration = counted['gpu__time_duration.sum']
        sectors = counted['lts__t_sectors.sum']
        gpc = counted['lts__t_sectors_srcnode_gpc.sum']
        fabric = counted['lts__t_sectors_srcunit_ltcfabric.sum']
        rows.append(dict(requested_gb=requested / 1e9, requested_gbps=requested / active_ns, dram_write_gb=counted['dram__bytes_write.sum'] / 1e9,
                         dram_read_gb=counted['dram__bytes_read.sum'] / 1e9, dram_gbps=counted['dram__bytes_write.sum'] / duration,
                         dram_percent_of_peak=counted['dram__throughput.avg.pct_of_peak_sustained_elapsed'], kernel_ms=duration / 1e6,
                         dram_over_requested=counted['dram__bytes_write.sum'] / requested, l2_gb=sectors * SECTOR_BYTES / 1e9, l2_gpc_gb=gpc * SECTOR_BYTES / 1e9,
                         l2_fabric_gb=fabric * SECTOR_BYTES / 1e9, l2_write_gb=counted['lts__t_sectors_op_write.sum'] * SECTOR_BYTES / 1e9,
                         l2_read_gb=counted['lts__t_sectors_op_read.sum'] * SECTOR_BYTES / 1e9, l2_tex_gb=counted['lts__t_sectors_srcunit_tex.sum'] * SECTOR_BYTES / 1e9,
                         l2_l1_gb=counted['lts__t_sectors_srcunit_l1.sum'] * SECTOR_BYTES / 1e9,
                         l2_hit_fraction=counted['lts__t_sectors_lookup_hit.sum'] / (counted['lts__t_sectors_lookup_hit.sum'] + counted['lts__t_sectors_lookup_miss.sum'])))
    return rows


def main():
    table = []
    for label in settings():
        launches = []
        runs = 0
        for repetition in range(1, ROUNDS + 1):
            path = RAW / f'{label}_R{repetition}.json'
            if path.exists():
                launches += launches_of_run(path)
                runs += 1
        if not launches:
            continue
        row = dict(setting=label, runs=runs, launches=len(launches))
        for key in launches[0]:
            row[key] = statistics.median(launch[key] for launch in launches)
        row['dram_over_requested_min'] = min(launch['dram_over_requested'] for launch in launches)
        row['dram_over_requested_max'] = max(launch['dram_over_requested'] for launch in launches)
        row['l2_gpc_over_requested'] = row['l2_gpc_gb'] / row['requested_gb']
        row['l2_fabric_over_requested'] = row['l2_fabric_gb'] / row['requested_gb']
        row['l2_over_requested'] = row['l2_gb'] / row['requested_gb']
        table.append(row)
    with (RESULTS / 'counter_check_summary.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    figure, (rate_axis, l2_axis) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw=dict(width_ratios=[2, 3]))
    rate_axis.plot([0, 3500], [0, 3500], color='#888888', linestyle='--', linewidth=.6, label='counted = requested')
    rate_axis.scatter([row['requested_gbps'] for row in table], [row['dram_gbps'] for row in table], s=16, color=COLORS['gpc'], zorder=3, label='one setting')
    rate_axis.axhline(PEAK_GBPS, color='#888888', linewidth=.6, linestyle=':', label=f'HBM peak {PEAK_GBPS:.0f} GB/s')
    rate_axis.set_xlim(0, 3500)
    rate_axis.set_ylim(0, 3500)
    rate_axis.set_xlabel('requested by the writer (GB/s, harness)')
    rate_axis.set_ylabel('counted DRAM writes ÷ kernel time (GB/s)')
    rate_axis.legend(fontsize=7, frameon=False)
    rate_axis.set_title('HBM writes: counter against harness', fontsize=9)
    positions = list(range(len(table)))
    gpc = [row['l2_gpc_gb'] / row['kernel_ms'] * 1000 for row in table]
    fabric = [row['l2_fabric_gb'] / row['kernel_ms'] * 1000 for row in table]
    other = [(row['l2_gb'] - row['l2_gpc_gb'] - row['l2_fabric_gb']) / row['kernel_ms'] * 1000 for row in table]
    l2_axis.bar(positions, gpc, color=COLORS['gpc'], label='from GPCs')
    l2_axis.bar(positions, fabric, bottom=gpc, color=COLORS['fabric'], label='from the L2 fabric (srcunit_ltcfabric)')
    l2_axis.bar(positions, other, bottom=[a + b for a, b in zip(gpc, fabric)], color=COLORS['other'], label='other')
    l2_axis.set_xticks(positions)
    l2_axis.set_xticklabels([row['setting'].replace('_unlimited', ' no limit').replace('_t1024', '').replace('_', ' ') for row in table], rotation=55, ha='right', fontsize=7)
    set_zero_based_y(l2_axis, 6000)
    l2_axis.set_ylabel('L2 sectors × 32 bytes ÷ kernel time (GB/s)')
    l2_axis.legend(fontsize=7, frameon=False)
    l2_axis.grid(axis='y', linewidth=.4, alpha=.5)
    l2_axis.set_title('L2 traffic by source (GPU-wide)', fontsize=9)
    figure.suptitle('Writer alone on GPU 6, median over launches and 3 runs; ncu application replay, clocks and caches left alone', fontsize=9)
    figure.tight_layout()
    figure.savefig(RESULTS / 'figures' / 'counter_check.png', dpi=150)
    plt.close(figure)
    for row in table:
        print(f"{row['setting']:36s} req {row['requested_gbps']:7.1f} GB/s {row['requested_gb']:7.3f} GB  dram {row['dram_write_gb']:7.3f} GB ratio {row['dram_over_requested']:.5f} [{row['dram_over_requested_min']:.5f},{row['dram_over_requested_max']:.5f}] "
              f"dram {row['dram_gbps']:6.0f} GB/s {row['dram_percent_of_peak']:5.1f}%pk  rd {row['dram_read_gb']:.3f}  L2 ×{row['l2_over_requested']:.3f} gpc ×{row['l2_gpc_over_requested']:.3f} fab ×{row['l2_fabric_over_requested']:.3f} wr {row['l2_write_gb']:.1f} rd {row['l2_read_gb']:.2f} hit {row['l2_hit_fraction']:.5f} tex {row['l2_tex_gb']:.1f} l1 {row['l2_l1_gb']:.1f}")


if __name__ == '__main__':
    main()
