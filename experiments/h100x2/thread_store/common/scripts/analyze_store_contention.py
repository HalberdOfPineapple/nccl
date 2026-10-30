#!/usr/bin/env python3
"""Plan 07 tables and figures for the copy next to a co-runner: thread-store writer in three
placements (E7.3, E7.4, E7.4b), TMA bulk reads (E7.5) and TMA bulk stores (E7.6, with its
calibration alone).

Unit of repetition: one run. Each mixed run alternates 6 idle-writer and 6 writing transfers; the
run's copy change is median(writing) / median(idle) - 1 over its own transfers, so each run carries
its own reference. Tables give the median of the per-run changes and their min-max over 3 runs.
Reference variability: interquartile range of the idle transfers of all mixed runs of a placement.
Writer: rate during the copy (writing transfers) against the writer alone on the same SMs, in
windows matched to its measured active time.
"""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_style import set_zero_based_y, zero_based_upper
from run_store_congest import (CONTENTION_PLACEMENTS, MAXIMUM_ROUNDS, PEAK_GBPS, RESULTS, TMA_WRITER_CHUNKS, TMA_WRITER_CTAS,
                               contention_levels, rate_levels, selection, summarize)

GROUP = RESULTS / 'contention' / 'raw'
FIGURES = RESULTS / 'figures'
HEALTHY = '#1f77b4'      # idle writers (project colour for healthy)
PERTURBED = '#ff7f0e'    # writing (project colour for perturbed)
PLACEMENT_TITLES = {
    'separate': 'separate GPCs (E7.3)\ncopy 16 SMs in G6; writer 64 SMs in the other 7 GPCs',
    'shared': 'shared GPCs (E7.4)\ncopy 2 SMs + writer 6-9 SMs in every GPC',
    'crowded': 'crowded GPCs (E7.4b)\nwriter fills G6, G7, G1, G2 next to 2 copy SMs each',
}
LEVEL_NAMES = {'p20': '20%', 'p40': '40%', 'p60': '60%', 'p80': '80%', 'unlimited': 'no limit'}


def quartiles(values):
    ordered = sorted(values)
    cut = statistics.quantiles(ordered, n=4, method='inclusive')
    return cut[0], cut[2]


def runs(prefix, group='contention'):
    found = []
    for repetition in range(1, MAXIMUM_ROUNDS + 1):
        path = RESULTS / group / 'raw' / f'{prefix}_R{repetition}.json'
        if path.exists():
            found.append(summarize(path))
    return found


def median_of(rows, key, idle):
    values = [row[key] for row in rows if row['writer_idle'] == idle and key in row]
    return statistics.median(values) if values else None


def mixed_run_summary(data, rows):
    result = dict(gpus=f"{data['source_gpu']},{data['destination_gpu']}")
    for key in ('throughput_gbps', 'issue_us', 'ordering_us', 'ack_wait_us', 'completed_us'):
        idle = median_of(rows, key, 1)
        writing = median_of(rows, key, 0)
        if idle is not None:
            result[f'{key}_idle'] = idle
            result[f'{key}_writing'] = writing
    result['copy_change'] = result['throughput_gbps_writing'] / result['throughput_gbps_idle'] - 1
    result['writer_gbps'] = median_of(rows, 'writer_gbps', 0)
    result['idle_transfers'] = [row['throughput_gbps'] for row in rows if row['writer_idle']]
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def style(axis):
    axis.grid(axis='y', linewidth=.4, alpha=.5)
    axis.spines[['top', 'right']].set_visible(False)


def contention_table():
    rows = []
    per_run = {}
    for placement in CONTENTION_PLACEMENTS:
        idle_pool = []
        placement_rows = []
        for level, target in contention_levels():
            mixed = [mixed_run_summary(*run) for run in runs(f'mixed_{placement}_{level}')]
            alone = [statistics.median(row['writer_gbps'] for row in run_rows) for _, run_rows in runs(f'writer_{placement}_{level}')]
            if not mixed:
                continue
            per_run[(placement, level)] = mixed
            for run in mixed:
                idle_pool += run['idle_transfers']
            changes = [run['copy_change'] for run in mixed]
            row = dict(placement=placement, level=level, target_gbps=target, runs=len(mixed),
                       copy_idle_gbps=statistics.median(run['throughput_gbps_idle'] for run in mixed),
                       copy_writing_gbps=statistics.median(run['throughput_gbps_writing'] for run in mixed),
                       copy_change_percent=100 * statistics.median(changes),
                       copy_change_min_percent=100 * min(changes), copy_change_max_percent=100 * max(changes))
            for key in ('issue_us', 'ordering_us', 'ack_wait_us'):
                if f'{key}_idle' in mixed[0]:
                    row[f'{key}_idle'] = statistics.median(run[f'{key}_idle'] for run in mixed)
                    row[f'{key}_writing'] = statistics.median(run[f'{key}_writing'] for run in mixed)
            row['writer_with_copy_gbps'] = statistics.median(run['writer_gbps'] for run in mixed)
            row['writer_alone_gbps'] = statistics.median(alone) if alone else ''
            row['writer_change_percent'] = 100 * (row['writer_with_copy_gbps'] / row['writer_alone_gbps'] - 1) if alone else ''
            placement_rows.append(row)
        if not placement_rows:
            continue
        lower, upper = quartiles(idle_pool)
        for row in placement_rows:
            row['idle_iqr_gbps'] = upper - lower
            change_gbps = row['copy_writing_gbps'] - row['copy_idle_gbps']
            row['effect_in_iqr'] = change_gbps / (upper - lower) if upper > lower else ''
        rows += placement_rows
    return rows, per_run


def references():
    rows = []
    for placement in CONTENTION_PLACEMENTS:
        for when in ('open', 'close'):
            path = GROUP / f'reference_{placement}_{when}.json'
            if path.exists():
                _, run_rows = summarize(path)
                rows.append(dict(placement=placement, reference=when, copy_gbps=statistics.median(row['throughput_gbps'] for row in run_rows)))
    return rows


def probe_cost():
    rows = []
    for detailed in (1, 0):
        for run_data, run_rows in runs(f'probe_separate_unlimited_d{detailed}'):
            summary = mixed_run_summary(run_data, run_rows)
            rows.append(dict(detailed=detailed, copy_idle_gbps=summary['throughput_gbps_idle'], copy_writing_gbps=summary['throughput_gbps_writing'],
                             copy_change_percent=100 * summary['copy_change'], writer_gbps=summary['writer_gbps']))
    return rows


def crowded_cta_table():
    """Per copy CTA in the crowded placement: issue loop with idle and writing writers, by whether
    the CTA's GPC holds writer CTAs."""
    rows = []
    for level, _ in contention_levels():
        groups = {}
        for data, _ in runs(f'mixed_crowded_{level}'):
            writer_gpcs = {data['gpc_of_sm'][sm] for sm, role in enumerate(data['role_of_sm']) if role == 2}
            for idle, records in zip(data['ranks'][0]['writer_idle_by_iteration'], data['ranks'][0]['iterations']):
                for record in records:
                    if record['role'] != 'copy':
                        continue
                    gpc = data['gpc_of_sm'][record['sm_id']]
                    key = ('with writers' if gpc in writer_gpcs else 'without writers', idle)
                    groups.setdefault(key, []).append((record['issue_end_ns'] - record['start_ns']) / 1000)
        for place in ('with writers', 'without writers'):
            if (place, 1) in groups:
                idle_value = statistics.median(groups[(place, 1)])
                writing_value = statistics.median(groups[(place, 0)])
                rows.append(dict(level=level, copy_ctas_gpc=place, ctas=len(groups[(place, 1)]) // max(1, len(runs(f'mixed_crowded_{level}'))) // 6,
                                 issue_us_idle=idle_value, issue_us_writing=writing_value, change_percent=100 * (writing_value / idle_value - 1)))
    return rows


def copy_figure(rows, per_run):
    placements = [placement for placement in CONTENTION_PLACEMENTS if any(row['placement'] == placement for row in rows)]
    upper = zero_based_upper([run['throughput_gbps_idle'] for runs_list in per_run.values() for run in runs_list] +
                             [run['throughput_gbps_writing'] for runs_list in per_run.values() for run in runs_list])
    figure, axes = plt.subplots(1, len(placements), figsize=(4.4 * len(placements), 3.9), sharey=True, squeeze=False)
    for axis, placement in zip(axes[0], placements):
        levels = [level for level, _ in contention_levels() if (placement, level) in per_run]
        for index, level in enumerate(levels):
            mixed = per_run[(placement, level)]
            axis.scatter([index - .12] * len(mixed), [run['throughput_gbps_idle'] for run in mixed], s=14, color=HEALTHY, label='writers idle' if index == 0 else None, zorder=3)
            axis.scatter([index + .12] * len(mixed), [run['throughput_gbps_writing'] for run in mixed], s=14, color=PERTURBED, marker='s', label='writers writing' if index == 0 else None, zorder=3)
        writer_rates = {row['level']: row['writer_with_copy_gbps'] for row in rows if row['placement'] == placement}
        axis.set_xticks(range(len(levels)))
        axis.set_xticklabels([f"{LEVEL_NAMES[level]}\n{writer_rates[level]:.0f} GB/s" for level in levels], fontsize=7)
        axis.set_xlabel('writer target (top) and achieved write rate (bottom)', fontsize=8)
        axis.set_title(PLACEMENT_TITLES[placement], fontsize=8)
        set_zero_based_y(axis, upper)
        style(axis)
    axes[0][0].set_ylabel('copy throughput (GB/s)')
    axes[0][0].legend(fontsize=7, frameon=False, loc='lower left')
    figure.suptitle('Copy 16 CTAs × 512 threads/CTA with writer 64 CTAs × 1024 threads/CTA on the same GPU; '
                    'each point: median of 6 transfers of one run (3 runs per level)', fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES / 'contention_copy.png', dpi=150)
    plt.close(figure)


def writer_figure(rows):
    placements = [placement for placement in CONTENTION_PLACEMENTS if any(row['placement'] == placement and row['writer_alone_gbps'] != '' for row in rows)]
    if not placements:
        return
    figure, axes = plt.subplots(1, len(placements), figsize=(4.4 * len(placements), 3.6), sharey=True, squeeze=False)
    for axis, placement in zip(axes[0], placements):
        series = [row for row in rows if row['placement'] == placement]
        positions = range(len(series))
        axis.bar([index - .2 for index in positions], [row['writer_alone_gbps'] for row in series], .4, color=HEALTHY, label='writer alone (copy idle)')
        axis.bar([index + .2 for index in positions], [row['writer_with_copy_gbps'] for row in series], .4, color=PERTURBED, label='writer with copy')
        axis.set_xticks(list(positions))
        axis.set_xticklabels([LEVEL_NAMES[row['level']] for row in series], fontsize=8)
        axis.set_xlabel('writer target (% of HBM peak)', fontsize=8)
        axis.set_title(PLACEMENT_TITLES[placement], fontsize=8)
        set_zero_based_y(axis, 3500)
        style(axis)
    axes[0][0].set_ylabel('achieved write rate (GB/s)')
    axes[0][0].legend(fontsize=7, frameon=False, loc='upper left')
    figure.suptitle(f'Writer 64 CTAs × 1024 threads/CTA alone and next to the copy (HBM peak {PEAK_GBPS:.0f} GB/s); median of 3 runs', fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES / 'contention_writer.png', dpi=150)
    plt.close(figure)


def crowded_figure(cta_rows):
    if not cta_rows:
        return
    levels = [level for level, _ in contention_levels() if any(row['level'] == level for row in cta_rows)]
    figure, axis = plt.subplots(figsize=(8.6, 3.6))
    width = .2
    for offset, (place, idle, color, hatch) in enumerate((('with writers', 1, HEALTHY, ''), ('with writers', 0, PERTURBED, ''),
                                                           ('without writers', 1, HEALTHY, '//'), ('without writers', 0, PERTURBED, '//'))):
        values = [next(row for row in cta_rows if row['level'] == level and row['copy_ctas_gpc'] == place)['issue_us_idle' if idle else 'issue_us_writing'] for level in levels]
        axis.bar([index + (offset - 1.5) * width for index in range(len(levels))], values, width, color=color, hatch=hatch, edgecolor='white',
                 label=f"copy CTAs in GPCs {place}, writers {'idle' if idle else 'writing'}")
    axis.set_xticks(range(len(levels)))
    axis.set_xticklabels([LEVEL_NAMES[level] for level in levels])
    axis.set_xlabel('writer target (% of HBM peak)')
    axis.set_ylabel('copy CTA issue loop (µs, median)')
    set_zero_based_y(axis, 8000)
    style(axis)
    axis.legend(fontsize=6.5, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
    counts = {row['copy_ctas_gpc']: row['ctas'] for row in cta_rows}
    axis.set_title(f"Crowded GPCs (E7.4b): issue loop of copy CTAs in GPCs with writers ({counts['with writers']} CTAs)\n"
                   f"and in GPCs without writers ({counts['without writers']} CTAs); median over CTAs, transfers and 3 runs", fontsize=7.5)
    figure.tight_layout()
    figure.savefig(FIGURES / 'contention_crowded_cta.png', dpi=150)
    plt.close(figure)


def contrast(group, kinds):
    """E7.5/E7.6: per co-runner kind and level, the same per-run copy pairs as contention_table().
    kinds: (kind, label of its mixed runs, label of its alone runs or None), labels with {level}."""
    rows, per_run = [], {}
    for kind, mixed_label, alone_label in kinds:
        idle_pool = []
        kind_rows = []
        for level, _ in rate_levels():
            mixed = [mixed_run_summary(*run) for run in runs(mixed_label.format(level=level), group)]
            if not mixed:
                continue
            per_run[(kind, level)] = mixed
            for run in mixed:
                idle_pool += run['idle_transfers']
            alone = [statistics.median(row['writer_gbps'] for row in run_rows) for _, run_rows in runs(alone_label.format(level=level), group)] if alone_label else []
            changes = [run['copy_change'] for run in mixed]
            row = dict(kind=kind, level=level, runs=len(mixed),
                       target_gbps=json.loads((RESULTS / group / 'raw' / f"{mixed_label.format(level=level)}_R1.json").read_text())['target_gbps'],
                       copy_idle_gbps=statistics.median(run['throughput_gbps_idle'] for run in mixed),
                       copy_writing_gbps=statistics.median(run['throughput_gbps_writing'] for run in mixed),
                       copy_change_percent=100 * statistics.median(changes), copy_change_min_percent=100 * min(changes), copy_change_max_percent=100 * max(changes))
            for key in ('issue_us', 'ordering_us', 'ack_wait_us'):
                row[f'{key}_idle'] = statistics.median(run[f'{key}_idle'] for run in mixed)
                row[f'{key}_writing'] = statistics.median(run[f'{key}_writing'] for run in mixed)
            row['co_runner_with_copy_gbps'] = statistics.median(run['writer_gbps'] for run in mixed)
            row['co_runner_alone_gbps'] = statistics.median(alone) if alone else ''
            row['co_runner_change_percent'] = 100 * (row['co_runner_with_copy_gbps'] / row['co_runner_alone_gbps'] - 1) if alone else ''
            kind_rows.append(row)
        if kind_rows:
            lower, upper = quartiles(idle_pool)
            for row in kind_rows:
                row['idle_iqr_gbps'] = upper - lower
                row['effect_in_iqr'] = (row['copy_writing_gbps'] - row['copy_idle_gbps']) / (upper - lower) if upper > lower else ''
        rows += kind_rows
    return rows, per_run


def contrast_figure(rows, per_run, kinds, titles, path, suptitle, idle_label, busy_label, rate_label):
    upper = zero_based_upper([run[key] for runs_list in per_run.values() for run in runs_list for key in ('throughput_gbps_idle', 'throughput_gbps_writing')])
    figure, axes = plt.subplots(1, len(kinds), figsize=(4.6 * len(kinds), 3.9), sharey=True, squeeze=False)
    for axis, kind in zip(axes[0], kinds):
        levels = [level for level, _ in rate_levels() if (kind, level) in per_run]
        for index, level in enumerate(levels):
            mixed = per_run[(kind, level)]
            axis.scatter([index - .12] * len(mixed), [run['throughput_gbps_idle'] for run in mixed], s=14, color=HEALTHY, label=idle_label if index == 0 else None, zorder=3)
            axis.scatter([index + .12] * len(mixed), [run['throughput_gbps_writing'] for run in mixed], s=14, color=PERTURBED, marker='s', label=busy_label if index == 0 else None, zorder=3)
        rates = {row['level']: row['co_runner_with_copy_gbps'] for row in rows if row['kind'] == kind}
        axis.set_xticks(range(len(levels)))
        axis.set_xticklabels([f"{LEVEL_NAMES[level]}\n{rates[level]:.0f} GB/s" for level in levels], fontsize=7)
        axis.set_xlabel(rate_label, fontsize=8)
        axis.set_title(titles[kind], fontsize=8)
        set_zero_based_y(axis, upper)
        style(axis)
    axes[0][0].set_ylabel('copy throughput (GB/s)')
    axes[0][0].legend(fontsize=7, frameon=False, loc='lower left')
    figure.suptitle(suptitle, fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES / path, dpi=150)
    plt.close(figure)


def tma_writer_calibration():
    """E7.6 calibration: the TMA store writer alone, spread, no limit (CTAs × chunks) and rate levels."""
    sweep, rate = [], []
    for ctas in TMA_WRITER_CTAS:
        for chunks in TMA_WRITER_CHUNKS:
            found = runs(f'tma_writer_w{ctas}_k{chunks}_unlimited_spread', 'tma_writer_alone')
            if found:
                values = [statistics.median(row['writer_gbps'] for row in run_rows) for _, run_rows in found]
                sweep.append(dict(writer_ctas=ctas, chunks_in_flight=chunks, runs=len(values), median=statistics.median(values), minimum=min(values), maximum=max(values),
                                  percent_of_peak=100 * statistics.median(values) / PEAK_GBPS, per_cta_gbps=statistics.median(values) / ctas,
                                  writer_slowest_to_fastest=min(min(row['writer_slowest_to_fastest'] for row in run_rows) for _, run_rows in found)))
    try:
        chosen = selection()['tma_writer']
    except KeyError:
        chosen = None
    if chosen:
        for level, target in rate_levels():
            found = runs(f"tma_writer_w{chosen['ctas']}_k{chosen['chunks']}_{level}_spread", 'tma_writer_alone')
            if found:
                values = [statistics.median(row['writer_gbps'] for row in run_rows) for _, run_rows in found]
                rate.append(dict(writer_ctas=chosen['ctas'], chunks_in_flight=chosen['chunks'], level=level, target_gbps=target, runs=len(values),
                                 median=statistics.median(values), minimum=min(values), maximum=max(values),
                                 achieved_over_target=statistics.median(values) / target if target else ''))
    if not sweep:
        return sweep, rate
    figure, (cta_axis, rate_axis) = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw=dict(width_ratios=[3, 2]))
    for color, chunks in zip(('#3a923a', '#9372b2', '#c03d3e'), TMA_WRITER_CHUNKS):
        series = [row for row in sweep if row['chunks_in_flight'] == chunks]
        cta_axis.plot([row['writer_ctas'] for row in series], [row['percent_of_peak'] for row in series], marker='o', markersize=3, linewidth=1, color=color,
                      label=f'{chunks} × 64 KiB chunk{"s" if chunks > 1 else ""} in flight per CTA')
    cta_axis.set_xlabel('TMA writer CTAs (one per SM)')
    cta_axis.set_ylabel('achieved write rate (% of HBM peak)')
    cta_axis.set_ylim(0, 100)
    cta_axis.legend(fontsize=7, frameon=False)
    style(cta_axis)
    cta_axis.set_title('No rate limit, spread over all GPCs; median of 3 runs', fontsize=8)
    if rate:
        limited = [row for row in rate if row['target_gbps']]
        rate_axis.plot([100 * row['target_gbps'] / PEAK_GBPS for row in limited], [100 * row['median'] / PEAK_GBPS for row in limited], marker='o', markersize=4, linewidth=1, color='#3a923a', label='achieved')
        for row in rate:
            if not row['target_gbps']:
                rate_axis.axhline(100 * row['median'] / PEAK_GBPS, color='#3a923a', linestyle=':', linewidth=.8, label=f"no limit: {100 * row['median'] / PEAK_GBPS:.1f}%")
        rate_axis.plot([0, 100], [0, 100], color='#888888', linestyle='--', linewidth=.6, label='achieved = target')
        rate_axis.set_xlim(0, 100)
        rate_axis.set_ylim(0, 100)
        rate_axis.set_xlabel('target write rate (% of HBM peak)')
        rate_axis.set_ylabel('achieved write rate (% of HBM peak)')
        rate_axis.legend(fontsize=7, frameon=False)
        rate_axis.set_title(f"Selected: {chosen['ctas']} CTAs × {chosen['chunks']} chunk in flight; median of 3 runs", fontsize=8)
        style(rate_axis)
    figure.suptitle(f'TMA bulk-store writer alone (64 KiB chunks from shared memory into a 1 GiB buffer; HBM peak {PEAK_GBPS:.0f} GB/s)', fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES / 'tma_writer.png', dpi=150)
    plt.close(figure)
    return sweep, rate


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows, per_run = contention_table()
    if not rows:
        print('no contention data')
        return
    write_csv(RESULTS / 'contention_summary.csv', rows)
    reference_rows = references()
    if reference_rows:
        write_csv(RESULTS / 'contention_references.csv', reference_rows)
    probe_rows = probe_cost()
    if probe_rows:
        write_csv(RESULTS / 'contention_probe_cost.csv', probe_rows)
    cta_rows = crowded_cta_table()
    if cta_rows:
        write_csv(RESULTS / 'contention_crowded_cta.csv', cta_rows)
    copy_figure(rows, per_run)
    writer_figure(rows)
    crowded_figure(cta_rows)
    direction_kinds = [('writes', 'mixed_writes_{level}', None), ('reads', 'mixed_reads_{level}', 'reader_alone_{level}')]
    direction_rows, direction_runs = contrast('direction', direction_kinds)
    if direction_rows:
        write_csv(RESULTS / 'direction_summary.csv', direction_rows)
        contrast_figure(direction_rows, direction_runs, ['writes', 'reads'],
                        {'writes': 'local writes: thread stores, 64 CTAs × 1024 threads/CTA\n(E7.3 setting, repeated in this session)',
                         'reads': 'local reads: TMA bulk reads, 48 CTAs × 3 × 64 KiB\ntargets = write rates achieved in E7.3'},
                        'direction_copy.png', 'Copy 16 CTAs × 512 threads/CTA in G6; co-runner in the other 7 GPCs; each point: median of 6 transfers of one run',
                        'co-runner idle', 'co-runner active', 'co-runner target (top) and achieved rate (bottom)')
    mechanism_kinds = [('thread', 'mixed_thread_{level}', None), ('tma', 'mixed_tma_{level}', 'tma_alone_{level}')]
    mechanism_rows, mechanism_runs = contrast('mechanism', mechanism_kinds)
    if mechanism_rows:
        write_csv(RESULTS / 'mechanism_summary.csv', mechanism_rows)
        contrast_figure(mechanism_rows, mechanism_runs, ['thread', 'tma'],
                        {'thread': 'thread stores: 64 CTAs × 1024 threads/CTA, 16-byte stores',
                         'tma': 'TMA bulk stores: 64 CTAs × 1 chunk of 64 KiB in flight'},
                        'mechanism_copy.png', 'Copy 16 CTAs × 512 threads/CTA in G6; writer in the other 7 GPCs; each point: median of 6 transfers of one run',
                        'writer idle', 'writer writing', 'writer target (top) and achieved write rate (bottom)')
    sweep_rows, rate_rows = tma_writer_calibration()
    if sweep_rows:
        write_csv(RESULTS / 'tma_writer_sweep_summary.csv', sweep_rows)
    if rate_rows:
        write_csv(RESULTS / 'tma_writer_rate_summary.csv', rate_rows)
    print(f'{len(rows)} placement-level rows, {len(reference_rows)} references, {len(probe_rows)} probe runs, {len(cta_rows)} per-CTA rows')


if __name__ == '__main__':
    main()
