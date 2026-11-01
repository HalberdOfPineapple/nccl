#!/usr/bin/env python3
"""Plan 08: tables and figures for the counter sanity checks (E8.0) and bytes stored once (E8.1).

Usage: analyze_peer_store_path.py RESULTS_DATE_DIRECTORY
Writes peer_store_summary.json and figures/ there. Each run gives one value per phase: the median of its
kept transfers. A table or figure then shows the median of those run values, and the runs themselves.
Counters are GPU-wide on the profiled GPU, so small background traffic is included.
"""
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from peer_store_counters import SECTOR_BYTES, label_of_run, median_by_phase, run_rows
from plot_style import set_zero_based_y, zero_based_upper

PEER_COLOR = '#2878b5'
LOCAL_COLOR = '#e1812c'
NO_STORES_COLOR = '#7b8594'
OTHER_SOURCE_COLOR = '#2878b5'
GPC_SOURCE_COLOR = '#e1812c'
FABRIC_SOURCE_COLOR = '#f2c08f'
MEBIBYTE = 2**20
GIBIBYTE = 2**30
STORED_MIB = [256, 512, 1024, 2048]
NVLINK_RECEIVE_USER = 'nvlrx__bytes_packet_request_data_user.sum'
NVLINK_SEND_USER = 'nvltx__bytes_packet_request_data_user.sum'
L2_WRITE_SECTORS = 'lts__t_sectors_op_write.sum'
HBM_WRITE = 'dram__bytes_write.sum'
HBM_READ = 'dram__bytes_read.sum'
GPC_SECTORS = 'lts__t_sectors_srcnode_gpc.sum'
FABRIC_WRITE_SECTORS = 'lts__t_sectors_srcunit_ltcfabric_op_write.sum'
PEER_APERTURE_WRITE_SECTORS = 'lts__t_sectors_aperture_peer_op_write.sum'
WORKLOAD_CAPTION = 'Stores: 16 CTAs × 512 threads/CTA, 4-byte stores, GPU 6 → GPU 7'


def load_runs(raw_directory):
    """{label: [one dict per run]}; each dict has the record and phase medians of that run."""
    runs = {}
    for json_path in sorted(raw_directory.glob('*_R?.json')):
        record, rows = run_rows(json_path)
        runs.setdefault(label_of_run(json_path.stem), []).append({'record': record, 'phases': median_by_phase(rows), 'name': json_path.stem})
    return runs


def phase_value(run, phase, metric):
    return run['phases'].get(phase, {}).get(metric, 0.0)


def sector_bytes(run, phase, metric):
    return phase_value(run, phase, metric) * SECTOR_BYTES


def median_of_runs(runs, value_of_run):
    return statistics.median(value_of_run(run) for run in runs)


def transfer_rates(run):
    """Bytes stored per second, decimal GB/s, for the whole transfer and for the store loop alone."""
    record = run['record']
    rank = record['ranks'][0 if record['store_source'] == 1 else 1]
    whole = []
    loop = []
    for transfer in rank['transfers']:
        stores = [cta for cta in transfer['ctas'] if cta['role'] == 'store']
        start_ns = min(cta['start_ns'] for cta in stores)
        whole.append(record['stored_bytes'] / (max(cta['end_ns'] for cta in stores) - start_ns))
        loop.append(record['stored_bytes'] / (max(cta['issue_end_ns'] for cta in stores) - start_ns))
    return statistics.median(whole), statistics.median(loop)


def wrong_values(runs):
    total = 0
    for run in runs:
        for rank in run['record']['ranks']:
            for transfer in rank['transfers']:
                total += transfer['check_wrong']
    return total


def line_fit(points):
    """Slope, intercept and the largest distance of a point from the line."""
    fit = statistics.linear_regression([x for x, _ in points], [y for _, y in points])
    largest = max(abs(y - (fit.slope * x + fit.intercept)) for x, y in points)
    return fit.slope, fit.intercept, largest


def summarize_enter(destination, source):
    """E8.1 tables. Held bytes are HBM writes during phaseEvict above the no-stores level."""
    no_stores = destination['none_window1ms'] + destination['none_window6ms']
    background_evict = median_of_runs(no_stores, lambda run: phase_value(run, 'phaseEvict', HBM_WRITE))
    summary = {'no_stores_evict_hbm_write_mib': background_evict / MEBIBYTE, 'wrong_values': wrong_values(sum(destination.values(), []) + sum(source.values(), [])), 'cases': {}}
    for prefix in ('peer', 'local'):
        points = []
        for stored_mib in STORED_MIB:
            runs = destination[f'{prefix}_f{stored_mib}']
            points.extend((run['record']['stored_bytes'], sector_bytes(run, 'phaseMain', L2_WRITE_SECTORS)) for run in runs)
            main_mib = median_of_runs(runs, lambda run: phase_value(run, 'phaseMain', HBM_WRITE)) / MEBIBYTE
            quiet_mib = median_of_runs(runs, lambda run: phase_value(run, 'phaseQuietAfter', HBM_WRITE)) / MEBIBYTE
            evict_mib = median_of_runs(runs, lambda run: phase_value(run, 'phaseEvict', HBM_WRITE)) / MEBIBYTE
            summary['cases'][f'{prefix}_f{stored_mib}'] = {
                'stored_mib': stored_mib,
                'runs': len(runs),
                'throughput_gb_s': [round(transfer_rates(run)[0], 1) for run in runs],
                'store_loop_gb_s': [round(transfer_rates(run)[1], 1) for run in runs],
                'nvlink_receive_user_mib': median_of_runs(runs, lambda run: phase_value(run, 'phaseMain', NVLINK_RECEIVE_USER)) / MEBIBYTE,
                'l2_write_mib': median_of_runs(runs, lambda run: sector_bytes(run, 'phaseMain', L2_WRITE_SECTORS)) / MEBIBYTE,
                'l2_gpc_all_operations_mib': median_of_runs(runs, lambda run: sector_bytes(run, 'phaseMain', GPC_SECTORS)) / MEBIBYTE,
                'l2_write_fabric_mib': median_of_runs(runs, lambda run: sector_bytes(run, 'phaseMain', FABRIC_WRITE_SECTORS)) / MEBIBYTE,
                'hbm_read_main_mib': median_of_runs(runs, lambda run: phase_value(run, 'phaseMain', HBM_READ)) / MEBIBYTE,
                'hbm_write_main_mib': main_mib,
                'hbm_write_quiet_after_mib': quiet_mib,
                'hbm_write_evict_mib': evict_mib,
                'held_mib': evict_mib - background_evict / MEBIBYTE,
                'unaccounted_mib': stored_mib - main_mib - quiet_mib - (evict_mib - background_evict / MEBIBYTE),
            }
        slope, intercept, largest = line_fit(points)
        summary[f'{prefix}_l2_write_fit'] = {'slope': slope, 'intercept_mib': intercept / MEBIBYTE, 'largest_distance_mib': largest / MEBIBYTE, 'points': len(points)}
    source_runs = source['peer_f1024']
    summary['source_peer_f1024'] = {
        'throughput_gb_s': [round(transfer_rates(run)[0], 1) for run in source_runs],
        'store_loop_gb_s': [round(transfer_rates(run)[1], 1) for run in source_runs],
        'nvlink_send_user_mib': median_of_runs(source_runs, lambda run: phase_value(run, 'phaseMain', NVLINK_SEND_USER)) / MEBIBYTE,
        'l2_write_mib': median_of_runs(source_runs, lambda run: sector_bytes(run, 'phaseMain', L2_WRITE_SECTORS)) / MEBIBYTE,
        'l2_peer_aperture_write_mib': median_of_runs(source_runs, lambda run: sector_bytes(run, 'phaseMain', PEER_APERTURE_WRITE_SECTORS)) / MEBIBYTE,
        'hbm_write_mib': median_of_runs(source_runs, lambda run: phase_value(run, 'phaseMain', HBM_WRITE)) / MEBIBYTE,
    }
    return summary


def summarize_validate(validate):
    """E8.0 tables, one run per case. Every phase and its HBM and link bytes, in MiB."""
    summary = {}
    for label, runs in validate.items():
        run = runs[0]
        summary[label] = {'stored_mib': run['record']['stored_bytes'] / MEBIBYTE, 'wrong_values': wrong_values(runs), 'phases': {}}
        for phase in run['phases']:
            summary[label]['phases'][phase] = {
                'nvlink_receive_user_mib': phase_value(run, phase, NVLINK_RECEIVE_USER) / MEBIBYTE,
                'l2_write_mib': sector_bytes(run, phase, L2_WRITE_SECTORS) / MEBIBYTE,
                'hbm_write_mib': phase_value(run, phase, HBM_WRITE) / MEBIBYTE,
                'hbm_read_mib': phase_value(run, phase, HBM_READ) / MEBIBYTE,
            }
    return summary


def plot_series(axis, destination, prefix, color, name, value_of_run):
    medians = []
    for stored_mib in STORED_MIB:
        runs = destination[f'{prefix}_f{stored_mib}']
        values = [value_of_run(run) / GIBIBYTE for run in runs]
        axis.scatter([stored_mib / 1024] * len(values), values, s=14, color=color, alpha=0.6, linewidths=0, zorder=3)
        medians.append(statistics.median(values))
    axis.plot([mib / 1024 for mib in STORED_MIB], medians, color=color, linewidth=1.6, marker='o', markersize=4, label=name, zorder=2)


def figure_bytes_follow_stored(destination, output):
    panels = [('NVLink user bytes received on GPU 7', lambda run: phase_value(run, 'phaseMain', NVLINK_RECEIVE_USER)),
              ('L2 write sectors × 32 bytes on GPU 7', lambda run: sector_bytes(run, 'phaseMain', L2_WRITE_SECTORS)),
              ('HBM bytes written on GPU 7, during the stores', lambda run: phase_value(run, 'phaseMain', HBM_WRITE))]
    upper = zero_based_upper([sector_bytes(run, 'phaseMain', L2_WRITE_SECTORS) / GIBIBYTE for runs in destination.values() for run in runs])
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    no_stores = destination['none_window1ms'] + destination['none_window6ms']
    for axis, (title, value_of_run) in zip(axes, panels):
        axis.plot([0, 2], [0, 2], color=NO_STORES_COLOR, linewidth=0.9, linestyle='--', label='Equal to bytes stored')
        plot_series(axis, destination, 'peer', PEER_COLOR, 'Peer stores from GPU 6', value_of_run)
        plot_series(axis, destination, 'local', LOCAL_COLOR, 'Local stores on GPU 7', value_of_run)
        axis.scatter([0] * len(no_stores), [value_of_run(run) / GIBIBYTE for run in no_stores], s=20, color=NO_STORES_COLOR, marker='s', label='No stores', zorder=3)
        axis.set_title(title, fontsize=10.5)
        axis.set_xlabel('Bytes stored once (GiB)')
        axis.set_xlim(-0.08, 2.15)
        axis.set_xticks([0, 0.25, 0.5, 1, 2])
        axis.set_xticklabels(['0', '0.25', '0.5', '1', '2'])
        axis.grid(axis='y', linewidth=0.4, alpha=0.5)
        set_zero_based_y(axis, upper)
    axes[0].set_ylabel('GiB during the stores')
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    figure.suptitle(f'{WORKLOAD_CAPTION}. Dots are runs, lines join medians (3 runs per point)', fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def figure_where_writes_come_from(destination, source, output):
    """Left: who sends the L2 writes on the destination. Right: which GPU sees the peer stores."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    positions = [0, 1]
    gpc_values = []
    fabric_values = []
    other_values = []
    for label in ('peer_f1024', 'local_f1024'):
        runs = destination[label]
        total = median_of_runs(runs, lambda run: sector_bytes(run, 'phaseMain', L2_WRITE_SECTORS)) / GIBIBYTE
        fabric = median_of_runs(runs, lambda run: sector_bytes(run, 'phaseMain', FABRIC_WRITE_SECTORS)) / GIBIBYTE
        gpc = median_of_runs(runs, lambda run: sector_bytes(run, 'phaseMain', GPC_SECTORS)) / GIBIBYTE
        gpc_values.append(gpc)
        fabric_values.append(fabric)
        other_values.append(total - gpc - fabric)
    left.bar(positions, other_values, width=0.5, color=OTHER_SOURCE_COLOR, label='Neither GPC nor fabric')
    left.bar(positions, gpc_values, width=0.5, bottom=other_values, color=GPC_SOURCE_COLOR, label='From a GPC')
    left.bar(positions, fabric_values, width=0.5, bottom=[other + gpc for other, gpc in zip(other_values, gpc_values)], color=FABRIC_SOURCE_COLOR, label='From the L2 fabric')
    left.set_xticks(positions)
    left.set_xticklabels(['Peer stores from GPU 6', 'Local stores on GPU 7'])
    left.set_title('L2 write sectors × 32 bytes on GPU 7, by who sent them', fontsize=10.5)
    left.set_ylabel('GiB during the stores')
    left.legend(frameon=False, fontsize=9, loc='upper left')
    metrics = ['NVLink user bytes', 'L2 writes', 'HBM writes']
    source_runs = source['peer_f1024']
    destination_runs = destination['peer_f1024']
    source_values = [median_of_runs(source_runs, lambda run: phase_value(run, 'phaseMain', NVLINK_SEND_USER)) / GIBIBYTE,
                     median_of_runs(source_runs, lambda run: sector_bytes(run, 'phaseMain', L2_WRITE_SECTORS)) / GIBIBYTE,
                     median_of_runs(source_runs, lambda run: phase_value(run, 'phaseMain', HBM_WRITE)) / GIBIBYTE]
    destination_values = [median_of_runs(destination_runs, lambda run: phase_value(run, 'phaseMain', NVLINK_RECEIVE_USER)) / GIBIBYTE,
                          median_of_runs(destination_runs, lambda run: sector_bytes(run, 'phaseMain', L2_WRITE_SECTORS)) / GIBIBYTE,
                          median_of_runs(destination_runs, lambda run: phase_value(run, 'phaseMain', HBM_WRITE)) / GIBIBYTE]
    for index, (source_value, destination_value) in enumerate(zip(source_values, destination_values)):
        right.bar(index - 0.19, source_value, width=0.34, color=NO_STORES_COLOR, label='Read on source GPU 6' if index == 0 else None)
        right.bar(index + 0.19, destination_value, width=0.34, color=PEER_COLOR, label='Read on destination GPU 7' if index == 0 else None)
        right.text(index - 0.19, source_value + 0.02, f'{source_value:.3f}' if source_value > 0.5 else f'{source_value:.4f}', ha='center', fontsize=8.5)
        right.text(index + 0.19, destination_value + 0.02, f'{destination_value:.3f}', ha='center', fontsize=8.5)
    right.set_xticks(range(len(metrics)))
    right.set_xticklabels(metrics)
    right.set_title('Peer stores, 1 GiB: what each GPU counted', fontsize=10.5)
    right.legend(frameon=False, fontsize=9, loc='upper right', bbox_to_anchor=(1.0, 0.86))
    for axis in (left, right):
        axis.grid(axis='y', linewidth=0.4, alpha=0.5)
        set_zero_based_y(axis, 2)
    figure.suptitle(f'{WORKLOAD_CAPTION}. Medians of 3 runs. NVLink: bytes sent on GPU 6, bytes received on GPU 7', fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def figure_held_until_eviction(destination, output):
    """HBM writes during the eviction reads, the data L2 was still holding after a 5 ms quiet phase."""
    figure, axis = plt.subplots(figsize=(8.5, 3.9))
    width = 0.36
    upper = zero_based_upper([phase_value(run, 'phaseEvict', HBM_WRITE) / MEBIBYTE for runs in destination.values() for run in runs])
    for index, stored_mib in enumerate(STORED_MIB):
        for offset, prefix, color, name in ((-width / 2, 'peer', PEER_COLOR, 'Peer stores from GPU 6'), (width / 2, 'local', LOCAL_COLOR, 'Local stores on GPU 7')):
            runs = destination[f'{prefix}_f{stored_mib}']
            value = median_of_runs(runs, lambda run: phase_value(run, 'phaseEvict', HBM_WRITE)) / MEBIBYTE
            axis.bar(index + offset, value, width=width * 0.92, color=color, label=name if index == 0 else None)
            axis.text(index + offset, value + 0.4, f'{value:.1f}', ha='center', fontsize=8.5)
    no_stores = destination['none_window1ms'] + destination['none_window6ms']
    no_stores_value = median_of_runs(no_stores, lambda run: phase_value(run, 'phaseEvict', HBM_WRITE)) / MEBIBYTE
    axis.bar(len(STORED_MIB), no_stores_value, width=width, color=NO_STORES_COLOR, label='No stores')
    axis.text(len(STORED_MIB), no_stores_value + 0.4, f'{no_stores_value:.1f}', ha='center', fontsize=8.5)
    axis.set_xticks(range(len(STORED_MIB) + 1))
    axis.set_xticklabels([f'{mib} MiB stored' for mib in STORED_MIB] + ['Nothing stored'])
    axis.set_ylabel('HBM bytes written on GPU 7\nduring eviction reads (MiB)')
    axis.set_title(f'{WORKLOAD_CAPTION}. Medians of 3 runs (6 runs for no stores)', fontsize=9.5)
    axis.grid(axis='y', linewidth=0.4, alpha=0.5)
    axis.legend(frameon=False, fontsize=9, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    set_zero_based_y(axis, upper * 1.2)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def figure_launch_boundary(validate, output):
    """8192 MiB stored over a 4 MiB footprint: how much of it reaches HBM, by phase."""
    cases = [('peer_f4_evictinmain0', 'Peer stores,\nevict in next launch'), ('peer_f4_evictinmain1', 'Peer stores,\nevict in same launch'),
             ('local_f4_evictinmain0', 'Local stores,\nevict in next launch'), ('local_f4_evictinmain1', 'Local stores,\nevict in same launch')]
    figure, axis = plt.subplots(figsize=(8.5, 3.9))
    for index, (label, _) in enumerate(cases):
        run = validate[label][0]
        main_mib = phase_value(run, 'phaseMain', HBM_WRITE) / MEBIBYTE
        evict_mib = phase_value(run, 'phaseEvict', HBM_WRITE) / MEBIBYTE
        color = PEER_COLOR if label.startswith('peer') else LOCAL_COLOR
        axis.bar(index, main_mib, width=0.5, color=color)
        axis.bar(index, evict_mib, width=0.5, bottom=main_mib, color=color, alpha=0.45)
        axis.text(index, main_mib + evict_mib + 0.2, f'{main_mib + evict_mib:.1f}', ha='center', fontsize=9)
    axis.set_xticks(range(len(cases)))
    axis.set_xticklabels([name for _, name in cases], fontsize=9)
    axis.set_ylabel('HBM bytes written on GPU 7 (MiB)')
    axis.set_title(f'8192 MiB stored over a 4 MiB footprint, one run per case.\n{WORKLOAD_CAPTION}', fontsize=9.5)
    axis.bar([0], [0], color=NO_STORES_COLOR, label='In the launch that has the stores')
    axis.bar([0], [0], color=NO_STORES_COLOR, alpha=0.45, label='In the next launch (eviction reads)')
    axis.legend(frameon=False, fontsize=9, loc='upper right')
    axis.grid(axis='y', linewidth=0.4, alpha=0.5)
    set_zero_based_y(axis, 8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    results = Path(sys.argv[1])
    destination = load_runs(results / 'enter/counters-destination/raw')
    source = load_runs(results / 'enter/counters-source/raw')
    validate = load_runs(results / 'validate/counters-destination/raw')
    figures = results / 'figures'
    figures.mkdir(exist_ok=True)
    summary = {'enter': summarize_enter(destination, source), 'validate': summarize_validate(validate)}
    (results / 'peer_store_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    figure_launch_boundary(validate, figures / 'validate_launch_boundary.png')
    figure_bytes_follow_stored(destination, figures / 'enter_bytes_follow_stored.png')
    figure_where_writes_come_from(destination, source, figures / 'enter_who_counts_the_stores.png')
    figure_held_until_eviction(destination, figures / 'enter_held_until_eviction.png')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
