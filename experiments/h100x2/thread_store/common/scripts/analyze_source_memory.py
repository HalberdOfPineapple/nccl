#!/usr/bin/env python3
"""Summarize the source-read campaign: calibrations first, then communication outcomes, then reads.

Every table keeps complete role geometries. Invocation medians are the unit; five rotated
rounds per setting. Sections are skipped until their phase has run.
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collect import summarize, percentile, write_csv
from plot_style import zero_based_upper
from run_source_memory import (RESULTS, SELECTION, READ_CTAS, READ_THREADS, COMMUNICATION_CTAS,
                               COMMUNICATION_THREADS, READ_UNROLLS, ROUNDS, group_read_levels)

HEALTHY = '#2878b5'
PERTURBED = '#e1812c'
FIGURES = RESULTS / 'figures'


def load(label):
    path = RESULTS / 'raw' / f'{label}.json'
    return summarize(path) if path.exists() else None


def invocation_medians(labels, field):
    """Median per invocation for every label that exists; None if any is missing."""
    rows = [load(label) for label in labels]
    if any(row is None for row in rows):
        return None
    return [statistics.median(row[field] for row in invocation) for invocation in rows]


def spread(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values),
                iqr=percentile(values, .75) - percentile(values, .25))


def compare(prefix, field):
    """Adjacent H/P pairs: grand medians, paired percent change, and IQR-normalized difference."""
    healthy = invocation_medians([f'{prefix}_H{repetition}' for repetition in range(1, ROUNDS + 1)], field)
    pressure = invocation_medians([f'{prefix}_P{repetition}' for repetition in range(1, ROUNDS + 1)], field)
    if healthy is None or pressure is None:
        return None
    healthy_iqr = percentile(healthy, .75) - percentile(healthy, .25)
    paired = [(perturbed / reference - 1) * 100 for reference, perturbed in zip(healthy, pressure)]
    return dict(healthy=statistics.median(healthy), healthy_min=min(healthy), healthy_max=max(healthy), healthy_iqr=healthy_iqr,
                pressure=statistics.median(pressure), pressure_min=min(pressure), pressure_max=max(pressure),
                paired_percent=statistics.median(paired), paired_min=min(paired), paired_max=max(paired),
                effect_iqr=(statistics.median(pressure) - statistics.median(healthy)) / healthy_iqr if healthy_iqr else None,
                lower_pairs=sum(perturbed < reference for reference, perturbed in zip(healthy, pressure)))


def validation_summary():
    rows = []
    for arm in ['copy_previous', 'copy_block512', 'copy_block1024']:
        values = invocation_medians([f'validate_{arm}_R{repetition}' for repetition in range(1, 4)], 'throughput_gbps')
        rows.append(dict(arm=arm, metric='communication_gbps', **spread(values)))
    for arm in ['destination_reads_previous', 'destination_reads_block512']:
        values = invocation_medians([f'validate_{arm}_R{repetition}' for repetition in range(1, 4)], 'co_rate')
        rows.append(dict(arm=arm, metric='logical_read_gbps', **spread(values)))
    return rows


def read_calibration():
    rows = []
    for memory_threads in READ_THREADS:
        for memory_ctas in READ_CTAS:
            values = invocation_medians([f'reads_m{memory_ctas}_t{memory_threads}_R{repetition}' for repetition in range(1, ROUNDS + 1)], 'co_rate')
            if values is None:
                return None
            rows.append(dict(read_ctas=memory_ctas, read_threads=memory_threads, **spread(values)))
    return rows


def plot_read_calibration(rows):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    upper = zero_based_upper([row['maximum'] for row in rows])
    # Stop short of viridis' pale yellow end, which has too little contrast on white.
    colors = plt.cm.viridis([0.85 * index / (len(READ_THREADS) - 1) for index in range(len(READ_THREADS))])
    for color, memory_threads in zip(colors, READ_THREADS):
        series = [row for row in rows if row['read_threads'] == memory_threads]
        axes[0].errorbar([row['read_ctas'] for row in series], [row['median'] for row in series],
                         yerr=[[row['median'] - row['minimum'] for row in series], [row['maximum'] - row['median'] for row in series]],
                         marker='o', markersize=3, linewidth=1, capsize=2, color=color, label=f'reads: M CTAs × {memory_threads} threads/CTA')
    axes[0].set(xlabel='Source read CTAs M', ylabel='Logical source read rate (GB/s)', title='Read CTAs at fixed threads/CTA', ylim=(0, upper))
    for memory_ctas in [8, 32, 64, 100, 131]:
        series = [row for row in rows if row['read_ctas'] == memory_ctas]
        axes[1].errorbar([row['read_threads'] for row in series], [row['median'] for row in series],
                         yerr=[[row['median'] - row['minimum'] for row in series], [row['maximum'] - row['median'] for row in series]],
                         marker='o', markersize=3, linewidth=1, capsize=2, label=f'reads: {memory_ctas} CTAs × Tm threads/CTA')
    axes[1].set(xlabel='Read threads per CTA Tm', ylabel='Logical source read rate (GB/s)', title='Threads/CTA at fixed read CTAs', ylim=(0, upper), xticks=READ_THREADS)
    for axis in axes:
        axis.legend(fontsize=7)
        axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle('Standalone source reads on GPU 6: copy disabled; source timer control 1 CTA; physical block 1024 threads; 4 loads/lane batch\n'
                    'Points: median of five invocation medians; bars: min–max; logical bytes over union active interval, not physical HBM bytes')
    figure.savefig(FIGURES / 'read_calibration.png', dpi=150)
    plt.close(figure)


def unroll_calibration(selection):
    rows = []
    for memory_ctas, memory_threads in selection.get('unroll_points', []):
        for depth in READ_UNROLLS:
            values = invocation_medians([f'unroll_m{memory_ctas}_t{memory_threads}_u{depth}_R{repetition}' for repetition in range(1, ROUNDS + 1)], 'co_rate')
            if values is None:
                return None
            rows.append(dict(read_ctas=memory_ctas, read_threads=memory_threads, read_unroll=depth, **spread(values)))
    return rows or None


def plot_unroll(rows):
    figure, axis = plt.subplots(figsize=(7, 4.5), layout='constrained')
    points = sorted({(row['read_ctas'], row['read_threads']) for row in rows})
    for memory_ctas, memory_threads in points:
        series = [row for row in rows if (row['read_ctas'], row['read_threads']) == (memory_ctas, memory_threads)]
        axis.errorbar([row['read_unroll'] for row in series], [row['median'] for row in series],
                      yerr=[[row['median'] - row['minimum'] for row in series], [row['maximum'] - row['median'] for row in series]],
                      marker='o', markersize=3, linewidth=1, capsize=2, label=f'reads: {memory_ctas} CTAs × {memory_threads} threads/CTA')
    axis.set(xscale='log', xticks=READ_UNROLLS, xticklabels=[str(depth) for depth in READ_UNROLLS], ylim=(0, zero_based_upper([row['maximum'] for row in rows])),
             xlabel='Requested independent 16-byte loads per read lane before accumulation', ylabel='Logical source read rate (GB/s)', title='Read request batch depth')
    axis.legend(fontsize=7)
    axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle('Standalone source reads on GPU 6; copy disabled; control 1 CTA; physical block 1024\nFive invocations per point; compiler grouping of loads is reported separately', fontsize=9)
    figure.savefig(FIGURES / 'read_unroll.png', dpi=150)
    plt.close(figure)


def communication_calibration():
    rows = []
    for ctas in COMMUNICATION_CTAS:
        for threads in COMMUNICATION_THREADS:
            values = invocation_medians([f'communication_c{ctas}_t{threads}_R{repetition}' for repetition in range(1, ROUNDS + 1)], 'throughput_gbps')
            if values is None:
                return None
            rows.append(dict(ctas=ctas, threads=threads, **spread(values)))
    for row in rows:
        previous = [other for other in rows if other['ctas'] == row['ctas'] and other['threads'] < row['threads']]
        row['change_from_previous_percent'] = (row['median'] / previous[-1]['median'] - 1) * 100 if previous else None
    return rows


def plot_communication(rows, selection):
    figure, axes = plt.subplots(1, len(COMMUNICATION_CTAS), figsize=(14, 4), layout='constrained', sharey=True)
    upper = zero_based_upper([row['maximum'] for row in rows])
    for axis, ctas in zip(axes, COMMUNICATION_CTAS):
        series = [row for row in rows if row['ctas'] == ctas]
        axis.errorbar([row['threads'] for row in series], [row['median'] for row in series],
                      yerr=[[row['median'] - row['minimum'] for row in series], [row['maximum'] - row['median'] for row in series]],
                      marker='o', markersize=3, linewidth=1, capsize=2, color=HEALTHY, label=f'communication: {ctas} CTAs × Tc threads/CTA')
        axis.set(title=f'Communication budget {ctas} CTAs', xlabel='Communication threads per CTA Tc', xticks=COMMUNICATION_THREADS, ylim=(0, upper))
        axis.legend(fontsize=7)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Completed communication (GB/s)')
    figure.suptitle(f"Copy only, GPU 6 → GPU 7: 2048 MiB; batch 8; CTA-local stride; source reads 0 × {selection['read_threads']}; destination control 1 CTA; physical block 1024\n"
                    'Points: median of five invocation medians; bars: min–max')
    figure.savefig(FIGURES / 'communication_calibration.png', dpi=150)
    plt.close(figure)


def control_calibration(selection):
    rows = []
    for group in selection.get('groups', []):
        for memory in group_read_levels(group['ctas']):
            values = invocation_medians([f"controls_c{group['ctas']}_t{group['threads']}_m{memory}_R{repetition}" for repetition in range(1, ROUNDS + 1)], 'co_rate')
            if values is None:
                return None
            rows.append(dict(ctas=group['ctas'], threads=group['threads'], read_ctas=memory, read_threads=selection['read_threads'], **spread(values)))
    return rows or None


def contention_summary(selection):
    communication = []
    phases = []
    concurrent_reads = []
    for group in selection.get('groups', []):
        ctas, threads = group['ctas'], group['threads']
        for memory in group_read_levels(ctas):
            prefix = f'contention_c{ctas}_t{threads}_m{memory}'
            result = compare(prefix, 'throughput_gbps')
            if result is None:
                return None
            communication.append(dict(ctas=ctas, threads=threads, read_ctas=memory, read_threads=selection['read_threads'], **result))
            reads = invocation_medians([f'{prefix}_P{repetition}' for repetition in range(1, ROUNDS + 1)], 'co_rate')
            standalone = invocation_medians([f'controls_c{ctas}_t{threads}_m{memory}_R{repetition}' for repetition in range(1, ROUNDS + 1)], 'co_rate')
            concurrent_reads.append(dict(ctas=ctas, threads=threads, read_ctas=memory, read_threads=selection['read_threads'],
                                         concurrent_gbps=statistics.median(reads), standalone_gbps=statistics.median(standalone) if standalone else None))
            for field in ['completed_us', 'issue_us', 'ordering_us', 'ack_wait_us', 'issue_spread_us']:
                phases.append(dict(ctas=ctas, threads=threads, read_ctas=memory, metric=field, **compare(prefix, field)))
    return dict(communication=communication, phases=phases, concurrent_reads=concurrent_reads) if communication else None


def plot_contention(summary, selection):
    groups = selection['groups']
    figure, axes = plt.subplots(2, len(groups), figsize=(4 * len(groups), 7.5), layout='constrained', squeeze=False)
    throughput_upper = zero_based_upper([row['healthy_max'] for row in summary['communication']] + [row['pressure_max'] for row in summary['communication']])
    read_upper = zero_based_upper([row['concurrent_gbps'] for row in summary['concurrent_reads']] + [row['standalone_gbps'] or 0 for row in summary['concurrent_reads']])
    for column, group in enumerate(groups):
        rows = [row for row in summary['communication'] if row['ctas'] == group['ctas']]
        positions = range(len(rows))
        labels = [str(row['read_ctas']) for row in rows]
        axis = axes[0][column]
        for offset, arm, color, label in [(-.2, 'healthy', HEALTHY, 'Copy only (reads 0)'), (.2, 'pressure', PERTURBED, 'Copy + source reads')]:
            axis.bar([position + offset for position in positions], [row[arm] for row in rows], width=.38, color=color, label=label,
                     yerr=[[row[arm] - row[f'{arm}_min'] for row in rows], [row[f'{arm}_max'] - row[arm] for row in rows]], capsize=2)
        axis.set(title=f"Communication {group['ctas']} × {group['threads']}", xticks=list(positions), xticklabels=labels, ylim=(0, throughput_upper),
                 xlabel=f"Source read CTAs (× {selection['read_threads']} threads/CTA)", ylabel='Completed communication (GB/s)' if column == 0 else None)
        axis.legend(fontsize=7)
        axis.spines[['top', 'right']].set_visible(False)
        reads = [row for row in summary['concurrent_reads'] if row['ctas'] == group['ctas']]
        axis = axes[1][column]
        for offset, field, color, label in [(-.2, 'standalone_gbps', HEALTHY, 'Reads alone (same control CTAs)'), (.2, 'concurrent_gbps', PERTURBED, 'Reads with copy')]:
            axis.bar([position + offset for position in positions], [row[field] or 0 for row in reads], width=.38, color=color, label=label)
        axis.set(xticks=list(positions), xticklabels=labels, ylim=(0, read_upper), xlabel=f"Source read CTAs (× {selection['read_threads']} threads/CTA)",
                 ylabel='Logical source read rate (GB/s)' if column == 0 else None, title='Achieved read rate (secondary)')
        axis.legend(fontsize=7)
        axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle(f"GPU 6 → GPU 7: 2048 MiB; batch 8; CTA-local stride; destination control 1 CTA; physical block 1024; read batch {selection['read_unroll']}\n"
                    'Top: five adjacent copy-only / copy-with-reads pairs per setting (median, min–max). Bottom: standalone reads come from a separate cohort, not a pairing')
    figure.savefig(FIGURES / 'contention.png', dpi=150)
    plt.close(figure)


def plot_phases(summary, selection):
    fields = [('issue_us', 'Source issue loop'), ('ordering_us', 'Source ordering interval'), ('ack_wait_us', 'Source acknowledgement wait')]
    groups = selection['groups']
    figure, axes = plt.subplots(len(groups), 3, figsize=(13, 3 * len(groups)), layout='constrained', squeeze=False)
    for row_index, group in enumerate(groups):
        for column, (field, title) in enumerate(fields):
            rows = [row for row in summary['phases'] if row['ctas'] == group['ctas'] and row['metric'] == field]
            axis = axes[row_index][column]
            upper = zero_based_upper([row[arm] for row in rows for arm in ['healthy', 'pressure']])
            for arm, color, label in [('healthy', HEALTHY, 'Copy only'), ('pressure', PERTURBED, 'Copy + source reads')]:
                axis.plot([str(row['read_ctas']) for row in rows], [row[arm] for row in rows], 'o-', markersize=3, linewidth=1, color=color, label=label)
            axis.set(ylim=(0, upper), title=f"{title}; communication {group['ctas']} × {group['threads']}", xlabel='Source read CTAs', ylabel='Median CTA duration (µs)')
            axis.title.set_fontsize(8)
            axis.spines[['top', 'right']].set_visible(False)
    axes[0][0].legend(fontsize=7)
    figure.suptitle(f"Phase medians; reads × {selection['read_threads']} threads/CTA; phase medians do not add up to completed duration", fontsize=9)
    figure.savefig(FIGURES / 'phases.png', dpi=150)
    plt.close(figure)


def mutual_summary(selection):
    rows = []
    for group in selection.get('groups', []):
        for memory in group.get('mutual_levels', []):
            prefix = f"mutual_c{group['ctas']}_t{group['threads']}_m{memory}"
            result = compare(prefix, 'co_rate')
            if result is None:
                return None
            active = {arm: statistics.median(row['co_active_us'] for repetition in range(1, ROUNDS + 1) for row in load(f'{prefix}_{arm}{repetition}')) for arm in 'HP'}
            rows.append(dict(ctas=group['ctas'], threads=group['threads'], read_ctas=memory, read_threads=selection['read_threads'],
                             solo_active_us=active['H'], concurrent_active_us=active['P'], **result))
    return rows or None


def probe_summary(selection):
    rows = []
    for group in selection.get('groups', []):
        memory = max(group_read_levels(group['ctas']))
        changes = []
        for repetition in range(1, ROUNDS + 1):
            minimal = load(f"probe_c{group['ctas']}_t{group['threads']}_m{memory}_d0_R{repetition}")
            detailed = load(f"probe_c{group['ctas']}_t{group['threads']}_m{memory}_d1_R{repetition}")
            if minimal is None or detailed is None:
                return None
            changes.append((statistics.median(row['completed_us'] for row in detailed) / statistics.median(row['completed_us'] for row in minimal) - 1) * 100)
        rows.append(dict(ctas=group['ctas'], threads=group['threads'], read_ctas=memory, median_percent=statistics.median(changes),
                         minimum_percent=min(changes), maximum_percent=max(changes)))
    return rows or None


def check_records():
    """Correctness, placement, and non-overlapping invocations across every retained record."""
    intervals = []
    windows = 0
    transfers = 0
    for path in sorted((RESULTS / 'raw').glob('*.json')):
        if path.name.endswith(('.before.json', '.after.json', '.telemetry.json')):
            continue
        data = json.loads(path.read_text())
        assert data['verified']
        for rank in data['ranks']:
            for records in rank['iterations']:
                assert len({record['sm_id'] for record in records}) == len(records)
        rows = summarize(path)
        windows += len(rows)
        transfers += len(rows) if data['communication'] else 0
        if data['co_ctas']:
            assert min(row['minimum_overlap_fraction'] for row in rows) == 1 and min(row['co_work_units'] for row in rows) > 0
        telemetry = json.loads(path.with_suffix('.telemetry.json').read_text())
        intervals.append((telemetry['host_start_ns'], telemetry['host_end_ns']))
    intervals.sort()
    assert all(previous[1] <= following[0] for previous, following in zip(intervals, intervals[1:])), 'Overlapping invocations'
    return dict(invocations=len(intervals), retained_windows=windows, retained_transfers=transfers)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    selection = json.loads(SELECTION.read_text()) if SELECTION.exists() else {}
    summary = dict(records=check_records(), validation=validation_summary())
    sections = [('read_calibration', read_calibration, ()), ('unroll', unroll_calibration, (selection,)),
                ('communication_calibration', communication_calibration, ()), ('controls', control_calibration, (selection,)),
                ('contention', contention_summary, (selection,)), ('mutual', mutual_summary, (selection,)), ('probes', probe_summary, (selection,))]
    for name, function, arguments in sections:
        result = function(*arguments)
        if result is not None:
            summary[name] = result
    if 'read_calibration' in summary:
        plot_read_calibration(summary['read_calibration'])
        write_csv(RESULTS / 'read_calibration.csv', summary['read_calibration'])
    if 'unroll' in summary:
        plot_unroll(summary['unroll'])
        write_csv(RESULTS / 'read_unroll.csv', summary['unroll'])
    if 'communication_calibration' in summary:
        plot_communication(summary['communication_calibration'], selection)
        write_csv(RESULTS / 'communication_calibration.csv', summary['communication_calibration'])
    if 'controls' in summary:
        write_csv(RESULTS / 'read_controls.csv', summary['controls'])
    if 'contention' in summary:
        plot_contention(summary['contention'], selection)
        plot_phases(summary['contention'], selection)
        write_csv(RESULTS / 'communication.csv', summary['contention']['communication'])
        write_csv(RESULTS / 'phases.csv', summary['contention']['phases'])
        write_csv(RESULTS / 'concurrent_reads.csv', summary['contention']['concurrent_reads'])
    if 'mutual' in summary:
        write_csv(RESULTS / 'mutual_reads.csv', summary['mutual'])
    (RESULTS / 'analysis_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key in ['records', 'validation']}, indent=2))


if __name__ == '__main__':
    main()
