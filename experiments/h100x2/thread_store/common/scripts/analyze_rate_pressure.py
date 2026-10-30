#!/usr/bin/env python3
"""Tables and figures for rate-controlled memory pressure (source and destination experiments).

Unit of repetition: one run. In the main comparison each run alternates launches with idle
pressure CTAs (reference) and launches with reads at the target, 6 + 6 retained transfers, so
both arms share the run's SM placement and buffers; the run's pair is the median of each arm.
Other comparisons (idle check, duration-matched reads, probes) pair separate adjacent runs.
Communication results come first; read-rate results come second and use their own controls.
Sections whose phase has not run are skipped.
"""
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from collect import summarize, percentile, write_csv
from run_rate_pressure import RESULTS, ROUNDS, TARGET_PERCENTS, PEAK_GBPS, SOURCE_CALIBRATION, DESTINATION_CALIBRATION

HEALTHY = '#2878b5'
PERTURBED = '#e1812c'
SIDES = ['source', 'destination']
SIDE_WORDS = {'source': 'reads on the source GPU', 'destination': 'reads on the destination GPU'}
SIDE_TITLES = {'source': 'Reads on the source GPU', 'destination': 'Reads on the destination GPU'}
FIGURES = RESULTS / 'figures'


def target_levels(side):
    """20-80% use 24 pressure CTAs; the 95% level (48 pressure CTAs) is included when it has run."""
    extra = [95] if (RESULTS / side / 'raw' / 'pairs_p95_R1.json').exists() else []
    return TARGET_PERCENTS + extra


def load(side, label):
    path = RESULTS / side / 'raw' / f'{label}.json'
    return summarize(path) if path.exists() else None


def run_value(side, label, field):
    rows = load(side, label)
    return None if rows is None else statistics.median(row[field] for row in rows)


def runs(side, labels, field):
    values = [run_value(side, label, field) for label in labels]
    return None if any(value is None for value in values) else values


def spread(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values))


def compare(side, healthy_labels, pressure_labels, field):
    """Grand medians, median of paired % changes, and (median_P - median_H) / IQR_H."""
    healthy = runs(side, healthy_labels, field)
    pressure = runs(side, pressure_labels, field)
    if healthy is None or pressure is None:
        return None
    healthy_iqr = percentile(healthy, .75) - percentile(healthy, .25)
    paired = [(perturbed / reference - 1) * 100 for reference, perturbed in zip(healthy, pressure)]
    return dict(healthy=statistics.median(healthy), healthy_min=min(healthy), healthy_max=max(healthy), healthy_iqr=healthy_iqr,
                pressure=statistics.median(pressure), pressure_min=min(pressure), pressure_max=max(pressure),
                paired_percent=statistics.median(paired), paired_min=min(paired), paired_max=max(paired),
                effect_iqr=(statistics.median(pressure) - statistics.median(healthy)) / healthy_iqr if healthy_iqr else None,
                lower_pairs=sum(perturbed < reference for reference, perturbed in zip(healthy, pressure)))


def transfer_tail(rows, field):
    return dict(p05=percentile([row[field] for row in rows], .05), worst=min(row[field] for row in rows))


def compare_within(side, labels, field):
    """Pairs inside alternating runs: per run, median of reading launches versus idle launches."""
    loaded = [load(side, label) for label in labels]
    if any(rows is None for rows in loaded):
        return None
    healthy = [statistics.median(row[field] for row in rows if row['pressure_idle']) for rows in loaded]
    pressure = [statistics.median(row[field] for row in rows if not row['pressure_idle']) for rows in loaded]
    healthy_iqr = percentile(healthy, .75) - percentile(healthy, .25)
    paired = [(perturbed / reference - 1) * 100 for reference, perturbed in zip(healthy, pressure)]
    return dict(healthy=statistics.median(healthy), healthy_min=min(healthy), healthy_max=max(healthy), healthy_iqr=healthy_iqr,
                pressure=statistics.median(pressure), pressure_min=min(pressure), pressure_max=max(pressure),
                paired_percent=statistics.median(paired), paired_min=min(paired), paired_max=max(paired), paired_values=paired,
                effect_iqr=(statistics.median(pressure) - statistics.median(healthy)) / healthy_iqr if healthy_iqr else None,
                lower_pairs=sum(perturbed < reference for reference, perturbed in zip(healthy, pressure)))


def copy_sm_set(side, label):
    """Identifier of the SMs that hosted copy CTAs in this run (fixed for all its launches)."""
    data = json.loads((RESULTS / side / 'raw' / f'{label}.json').read_text())
    sets = {tuple(sorted(record['sm_id'] for record in launch if record['role'] == 0)) for launch in data['ranks'][0]['iterations']}
    assert len(sets) == 1, f'Placement changed inside {label}'
    return hash(next(iter(sets))) & 0xffffff


def selection(side):
    path = RESULTS / side / 'selection.json'
    return json.loads(path.read_text()) if path.exists() else None


def calibration(side):
    grid = SOURCE_CALIBRATION if side == 'source' else DESTINATION_CALIBRATION
    rows = []
    for ctas, threads in grid:
        values = runs(side, [f'calibrate_c{ctas}_t{threads}_R{repetition}' for repetition in range(1, ROUNDS + 1)], 'throughput_gbps')
        if values is None:
            return None
        rows.append(dict(ctas=ctas, threads=threads, **spread(values)))
    return rows


def validation(side):
    result = {}
    for prefix in ['validate', 'validate_final']:
        previous = [statistics.median(row['throughput_gbps'] for row in summarize(RESULTS / side / 'raw' / f'{prefix}_previous_R{repetition}.json'))
                    for repetition in range(1, 4) if (RESULTS / side / 'raw' / f'{prefix}_previous_R{repetition}.json').exists()]
        new = runs(side, [f'{prefix}_new_R{repetition}' for repetition in range(1, 4)], 'throughput_gbps')
        if len(previous) == 3 and new is not None:
            result[prefix] = dict(previous=spread(previous), new=spread(new))
    return result or None


def idle_check(side):
    return compare(side, [f'idle_check_none_R{repetition}' for repetition in range(1, ROUNDS + 1)],
                   [f'idle_check_idle_R{repetition}' for repetition in range(1, ROUNDS + 1)], 'throughput_gbps')


def pressure_tables(side):
    communication = []
    reads = []
    phases = []
    placements = []
    for percent in target_levels(side):
        labels = [f'pairs_p{percent}_R{repetition}' for repetition in range(1, ROUNDS + 1)]
        result = compare_within(side, labels, 'throughput_gbps')
        if result is None:
            return None
        rows = [row for label in labels for row in load(side, label)]
        tails = {'healthy': transfer_tail([row for row in rows if row['pressure_idle']], 'throughput_gbps'),
                 'pressure': transfer_tail([row for row in rows if not row['pressure_idle']], 'throughput_gbps')}
        for label in labels:
            run_rows = load(side, label)
            placements.append(dict(run=label, copy_sm_set=copy_sm_set(side, label),
                                   reference_gbps=statistics.median(row['throughput_gbps'] for row in run_rows if row['pressure_idle'])))
        communication.append(dict(target_percent=percent, **result, healthy_p05=tails['healthy']['p05'], pressure_p05=tails['pressure']['p05'],
                                  healthy_worst=tails['healthy']['worst'], pressure_worst=tails['pressure']['worst']))
        alone = runs(side, [f'alone_p{percent}_R{repetition}' for repetition in range(1, ROUNDS + 1)], 'co_rate')
        concurrent = [statistics.median(row['co_rate'] for row in load(side, label) if not row['pressure_idle']) for label in labels]
        coverage = min(row['minimum_overlap_fraction'] for row in rows if not row['pressure_idle'])
        reads.append(dict(target_percent=percent, target_gbps=PEAK_GBPS * percent / 100,
                          alone_gbps=statistics.median(alone) if alone else None, alone_min=min(alone) if alone else None, alone_max=max(alone) if alone else None,
                          with_copy_gbps=statistics.median(concurrent), with_copy_min=min(concurrent), with_copy_max=max(concurrent),
                          minimum_coverage=coverage))
        for field in ['completed_us', 'issue_us', 'ordering_us', 'ack_wait_us', 'issue_spread_us', 'receiver_wait_us']:
            phases.append(dict(target_percent=percent, metric=field, **compare_within(side, labels, field)))
    references = {name: run_value(side, f'pairs_{name}_reference', 'throughput_gbps') for name in ['opening', 'closing']}
    # How much the reference level varies between runs, and how much of it follows SM placement.
    levels = {}
    for placement in placements:
        levels.setdefault(placement['copy_sm_set'], []).append(placement['reference_gbps'])
    placement_summary = dict(runs=len(placements), distinct_copy_sm_sets=len(levels),
                             reference_min=min(p['reference_gbps'] for p in placements), reference_max=max(p['reference_gbps'] for p in placements),
                             largest_spread_within_one_sm_set=max(max(values) - min(values) for values in levels.values()))
    return dict(communication=communication, reads=reads, phases=phases, references=references, placement=placement_summary)


def mutual(side):
    chosen = selection(side) or {}
    rows = []
    for percent in chosen.get('mutual_percents', []) + ([95] if (RESULTS / side / 'raw' / 'mutual_p95_H1.json').exists() else []):
        healthy = [f'mutual_p{percent}_H{repetition}' for repetition in range(1, ROUNDS + 1)]
        pressure = [f'mutual_p{percent}_P{repetition}' for repetition in range(1, ROUNDS + 1)]
        result = compare(side, healthy, pressure, 'co_rate')
        if result is None:
            return None
        active = {arm: statistics.median(row['co_active_us'] for label in labels for row in load(side, label)) for arm, labels in [('alone', healthy), ('with_copy', pressure)]}
        rows.append(dict(target_percent=percent, alone_active_us=active['alone'], with_copy_active_us=active['with_copy'], **result))
    return rows or None


def probes(side):
    changes = []
    for repetition in range(1, ROUNDS + 1):
        minimal = run_value(side, f'probes_p80_d0_R{repetition}', 'completed_us')
        detailed = run_value(side, f'probes_p80_d1_R{repetition}', 'completed_us')
        if minimal is None or detailed is None:
            return None
        changes.append((detailed / minimal - 1) * 100)
    return dict(median_percent=statistics.median(changes), minimum_percent=min(changes), maximum_percent=max(changes))


def check_records(side):
    """Correctness, placement, coverage, and no overlapping runs on this experiment's pair."""
    intervals = []
    transfers = 0
    windows = 0
    pairs = set()
    for path in sorted((RESULTS / side / 'raw').glob('*.json')):
        if path.name.endswith(('.before.json', '.after.json', '.telemetry.json')):
            continue
        data = json.loads(path.read_text())
        assert data['verified']
        pairs.add((data['source_gpu'], data['destination_gpu']))
        for rank in data['ranks']:
            for records in rank['iterations']:
                assert len({record['sm_id'] for record in records}) == len(records)
        rows = summarize(path)
        windows += len(rows)
        transfers += len(rows) if data['communication'] else 0
        reading = [row for row in rows if not row['pressure_idle']]
        if data['co_ctas'] and data['kind'] >= 4 and reading:
            assert min(row['minimum_overlap_fraction'] for row in reading) == 1 and min(row['co_work_units'] for row in reading) > 0
        telemetry = json.loads(path.with_suffix('.telemetry.json').read_text())
        intervals.append((telemetry['host_start_ns'], telemetry['host_end_ns']))
    intervals.sort()
    assert all(previous[1] <= following[0] for previous, following in zip(intervals, intervals[1:])), 'Overlapping runs on one pair'
    return dict(runs=len(intervals), transfers=transfers, read_only_windows=windows - transfers, gpu_pairs=sorted(pairs))


def style(axis, upper):
    axis.set_ylim(0, upper)
    axis.grid(axis='y', linewidth=.4, alpha=.5)
    axis.spines[['top', 'right']].set_visible(False)


def plot_communication(summaries):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained', sharey=True)
    for axis, side in zip(axes, SIDES):
        tables = summaries[side].get('pressure')
        if not tables:
            axis.set_visible(False)
            continue
        rows = tables['communication']
        chosen = selection(side)
        positions = range(len(rows))
        for offset, arm, color, label in [(-.2, 'healthy', HEALTHY, 'Idle launches (reference)'), (.2, 'pressure', PERTURBED, 'Reading launches')]:
            axis.bar([position + offset for position in positions], [row[arm] for row in rows], width=.38, color=color, label=label,
                     yerr=[[row[arm] - row[f'{arm}_min'] for row in rows], [row[f'{arm}_max'] - row[arm] for row in rows]], capsize=2)
        axis.set_xticks(list(positions), [f"{row['target_percent']}%" for row in rows])
        axis.set(xlabel='Target read rate (% of HBM peak 3352 GB/s)', title=f"{SIDE_TITLES[side]}\ncopy {chosen['ctas']} CTAs × {chosen['threads']} threads/CTA")
        style(axis, 450)
        axis.legend(fontsize=7, loc='lower right')
    axes[0].set_ylabel('Completed copy throughput (GB/s)')
    figure.savefig(FIGURES / 'communication.png', dpi=150)
    plt.close(figure)


def plot_paired_change(summaries):
    # Change inside each run (reading launches vs idle launches on the same SMs); one dot per run.
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained', sharey=True)
    for axis, side in zip(axes, SIDES):
        tables = summaries[side].get('pressure')
        if not tables:
            axis.set_visible(False)
            continue
        for row in tables['communication']:
            values = row['paired_values']
            axis.plot([row['target_percent']] * len(values), values, 'o', color=PERTURBED, markersize=4, alpha=.7)
            axis.plot([row['target_percent'] - 4, row['target_percent'] + 4], [row['paired_percent']] * 2, color='black', linewidth=1.5)
        axis.axhline(0, color='grey', linewidth=.8)
        axis.set_xlim(10, 100)
        axis.set_xticks(target_levels(side), [f"{percent}%" if percent < 95 else "95%\n(48 CTAs)" for percent in target_levels(side)])
        chosen = selection(side)
        axis.set(xlabel='Target read rate (% of HBM peak 3352 GB/s)', title=f"{SIDE_TITLES[side]}; copy {chosen['ctas']} CTAs × {chosen['threads']} threads/CTA")
        axis.grid(axis='y', linewidth=.4, alpha=.5)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Copy throughput change in the same run (%)')
    # Lower limit from the data, rounded down, so no run falls outside the plot.
    lowest = min(value for side in SIDES if summaries[side].get('pressure') for row in summaries[side]['pressure']['communication'] for value in row['paired_values'])
    axes[0].set_ylim(math.floor(lowest) - 1, 1)
    figure.savefig(FIGURES / 'paired_change.png', dpi=150)
    plt.close(figure)


def plot_reads(summaries):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained', sharey=True)
    for axis, side in zip(axes, SIDES):
        tables = summaries[side].get('pressure')
        if not tables:
            axis.set_visible(False)
            continue
        rows = tables['reads']
        axis.plot([0, 100], [0, 100], color='grey', linestyle='--', linewidth=.8, label='achieved = target')
        # Lines join only the 24-pressure-CTA levels; the 95% level (48 CTAs) is drawn on its own.
        main = [row for row in rows if row['target_percent'] < 95]
        extra = [row for row in rows if row['target_percent'] == 95]
        axis.plot([row['target_percent'] for row in main], [100 * row['alone_gbps'] / PEAK_GBPS for row in main], 'o-', color=HEALTHY, linewidth=1, markersize=4, label='Reads alone (copy off), 24 pressure CTAs')
        axis.plot([row['target_percent'] for row in main], [100 * row['with_copy_gbps'] / PEAK_GBPS for row in main], 's-', color=PERTURBED, linewidth=1, markersize=4, label='Reads while the copy runs, 24 pressure CTAs')
        for row in extra:
            axis.plot(row['target_percent'], 100 * row['alone_gbps'] / PEAK_GBPS, 'D', color=HEALTHY, markersize=6, markerfacecolor='none', label='Reads alone, 48 pressure CTAs')
            axis.plot(row['target_percent'], 100 * row['with_copy_gbps'] / PEAK_GBPS, 'D', color=PERTURBED, markersize=6, markerfacecolor='none', label='Reads while the copy runs, 48 pressure CTAs')
        axis.set_xlim(0, 100)
        axis.xaxis.set_major_locator(MultipleLocator(20))
        axis.set(xlabel='Target read rate (% of HBM peak)', title=f'{SIDE_TITLES[side]}: achieved read rate')
        style(axis, 100)
        axis.legend(fontsize=7, loc='upper left')
    axes[0].set_ylabel('Achieved read rate (% of HBM peak 3352 GB/s)')
    figure.savefig(FIGURES / 'read_rate.png', dpi=150)
    plt.close(figure)


def plot_phases(summaries):
    fields = [('issue_us', 'Issue loop'), ('ordering_us', 'Ordering'), ('ack_wait_us', 'Acknowledgement wait')]
    figure, axes = plt.subplots(2, 3, figsize=(13, 6.5), layout='constrained', squeeze=False)
    for row_index, side in enumerate(SIDES):
        tables = summaries[side].get('pressure')
        for column, (field, title) in enumerate(fields):
            axis = axes[row_index][column]
            if not tables:
                axis.set_visible(False)
                continue
            rows = [row for row in tables['phases'] if row['metric'] == field]
            upper = max(row[arm] for row in rows for arm in ['healthy', 'pressure']) * 1.25
            for arm, color, label in [('healthy', HEALTHY, 'Copy with idle pressure CTAs'), ('pressure', PERTURBED, 'Copy with reads')]:
                main = [row for row in rows if row['target_percent'] < 95]
                axis.plot([row['target_percent'] for row in main], [row[arm] for row in main], 'o-', color=color, linewidth=1, markersize=4, label=label)
                # The 95% level uses 48 pressure CTAs: separate, unconnected marker.
                for row in rows:
                    if row['target_percent'] == 95:
                        axis.plot(row['target_percent'], row[arm], 'D', color=color, markersize=6, markerfacecolor='none')
            axis.set_xticks([row['target_percent'] for row in rows], [f"{row['target_percent']}%" if row['target_percent'] < 95 else "95%\n(48)" for row in rows])
            axis.set(title=f'{title}: {SIDE_WORDS[side]}', xlabel='Target read rate', ylabel='Median CTA time (µs)')
            axis.title.set_fontsize(9)
            style(axis, upper)
    axes[0][0].legend(fontsize=7)
    figure.savefig(FIGURES / 'phases.png', dpi=150)
    plt.close(figure)


def plot_calibration(summaries):
    rows = summaries['source'].get('calibration')
    if not rows:
        return
    figure, axis = plt.subplots(figsize=(7, 4.4), layout='constrained')
    for color, ctas in zip(['#3a923a', '#9372b2', '#2a9d9a'], sorted({row['ctas'] for row in rows})):
        series = [row for row in rows if row['ctas'] == ctas]
        axis.errorbar([row['threads'] for row in series], [row['median'] for row in series],
                      yerr=[[row['median'] - row['minimum'] for row in series], [row['maximum'] - row['median'] for row in series]],
                      marker='o', markersize=4, linewidth=1, capsize=2, color=color, label=f'copy: {ctas} CTAs × threads/CTA')
    axis.set(xticks=[512, 768, 1024], xlabel='Copy threads per CTA', ylabel='Completed copy throughput (GB/s)',
             title='Copy alone next to 24 idle pressure CTAs (source budget: copy CTAs ≤ 108)')
    axis.title.set_fontsize(9)
    style(axis, 450)
    axis.legend(fontsize=7, loc='lower right')
    figure.savefig(FIGURES / 'source_calibration.png', dpi=150)
    plt.close(figure)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for side in SIDES:
        if not (RESULTS / side / 'raw').exists():
            summaries[side] = {}
            continue
        summary = dict(records=check_records(side))
        for name, function in [('validation', validation), ('calibration', calibration), ('idle_check', idle_check),
                               ('pressure', pressure_tables), ('mutual', mutual), ('probes', probes)]:
            if name == 'validation' and side != 'destination':
                continue
            if name != 'calibration' and name != 'validation' and selection(side) is None:
                continue
            result = function(side)
            if result is not None:
                summary[name] = result
        if 'calibration' in summary:
            write_csv(RESULTS / side / 'calibration.csv', summary['calibration'])
        if 'pressure' in summary:
            write_csv(RESULTS / side / 'communication.csv', summary['pressure']['communication'])
            write_csv(RESULTS / side / 'read_rate.csv', summary['pressure']['reads'])
            write_csv(RESULTS / side / 'phases.csv', summary['pressure']['phases'])
        (RESULTS / side / 'analysis_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        summaries[side] = summary
    plot_calibration(summaries)
    if any('pressure' in summary for summary in summaries.values()):
        plot_communication(summaries)
        plot_paired_change(summaries)
        plot_reads(summaries)
        plot_phases(summaries)
    print(json.dumps({side: summary.get('records') for side, summary in summaries.items()}, indent=1))


if __name__ == '__main__':
    main()
