#!/usr/bin/env python3
"""Plan 01 figures and report tables, generated only from the saved session summaries.

Usage: make_figures.py DATED_RESULTS_DIRECTORY
Writes figures/*.png and report_summary.json inside that directory. Reference arms are blue and
perturbed (mixed) arms are orange. Every throughput axis starts at zero.
"""
import json
from pathlib import Path
import re
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REFERENCE_COLOR = '#2f6fb0'
PERTURBED_COLOR = '#e07b22'
PERTURBED_SECOND_COLOR = '#f3b77e'
SECOND_SERIES_COLOR = '#7f7f7f'
CTA_COUNTS = [4, 8, 16, 32, 48, 64]
ROLE_TITLES = {'multicast': 'NVLS AllGather (st_multicast)', 'reduce': 'NVLS ReduceScatter (ld_reduce)', 'allreduce': 'NVLS AllReduce (same CTAs)', 'plain_load': 'Plain load, ring', 'plain_store': 'Plain store, ring'}
BASE_CONFIG = '32 CTAs × 512 threads/CTA'


def zero_based_upper(values):
    return MaxNLocator(nbins=5).tick_values(0, max(values) * 1.15)[-1]


def set_zero_based_y(axis, upper, label):
    axis.set_ylim(0, upper)
    axis.set_ylabel(label)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(axis='y', color='#dddddd', linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right']].set_visible(False)


def load_rows(results, session):
    return {row['run']: row for row in json.loads((results / session / f'{session}_summary.json').read_text())}


def rows_of_arm(rows, arm):
    return [row for name, row in rows.items() if re.fullmatch(rf'{arm}_R\d+', name)]


def interquartile_range(values):
    quartiles = statistics.quantiles(values, n=4, method='inclusive')
    return quartiles[2] - quartiles[0]


def describe_arm(arm_rows):
    """Medians over rounds for one arm, with the spread of each role's busBW over rounds."""
    description = {'rounds': len(arm_rows), 'roles': {}}
    for role in arm_rows[0]['roles']:
        rates = [row['roles'][role]['rate_per_issuing_gpu_gbps'] for row in arm_rows]
        buses = [row['roles'][role]['bus_bw_gbps'] for row in arm_rows]
        description['roles'][role] = {
            'rate_median_gbps': statistics.median(rates),
            'bus_median_gbps': statistics.median(buses),
            'bus_iqr_gbps': interquartile_range(buses) if len(buses) > 1 else None,
            'cta_chunk_ratio_median': statistics.median(row['roles'][role]['cta_chunk_ratio_max_over_min'] for row in arm_rows),
        }
    description['link_rx_gbps_per_gpu'] = [statistics.median(row['link_rx_gbps_estimate'][gpu] for row in arm_rows) for gpu in range(4)]
    description['link_tx_gbps_per_gpu'] = [statistics.median(row['link_tx_gbps_estimate'][gpu] for row in arm_rows) for gpu in range(4)]
    description['busiest_direction_gbps'] = statistics.median(max(statistics.fmean(row['link_rx_gbps_estimate']), statistics.fmean(row['link_tx_gbps_estimate'])) for row in arm_rows)
    if all(values['ranks_issuing'] == 4 for values in arm_rows[0]['roles'].values()):
        description['link_model_bus_median_gbps'] = statistics.median(link_model_bus(row) for row in arm_rows)
    description['mismatches'] = sum(row['mismatches'] for row in arm_rows)
    overlaps = [row['overlap_min'] for row in arm_rows if row['overlap_min'] is not None]
    description['overlap_min'] = min(overlaps) if overlaps else None
    description['sm_clock_mhz'] = sorted({row['sm_clock_mhz'] for row in arm_rows})
    description['clock_reasons'] = sorted({reason for row in arm_rows for reason in row['clock_reasons']})
    description['power_max_w'] = max(row['power_max_w'] for row in arm_rows)
    return description


def round_ratios(rows, numerator, denominators):
    """One ratio per round, from run medians of the same round. A callable picks each arm's value."""
    rounds = sorted({int(name.rsplit('_R', 1)[1]) for name in rows})
    ratios = []
    for round_index in rounds:
        reference = statistics.fmean(pick(rows[f'{arm}_R{round_index}']) for arm, pick in denominators)
        arm, pick = numerator
        ratios.append(pick(rows[f'{arm}_R{round_index}']) / reference)
    return ratios


def bus_of(role):
    return lambda row: row['roles'][role]['bus_bw_gbps']


def rate_of(role):
    return lambda row: row['roles'][role]['rate_per_issuing_gpu_gbps']


def summed_bus(row):
    return sum(values['bus_bw_gbps'] for values in row['roles'].values())


def busiest_direction(row):
    return max(statistics.fmean(row['link_rx_gbps_estimate']), statistics.fmean(row['link_tx_gbps_estimate']))


# Link-model busBW (N5): the data one GPU must move in its busier link direction, computed from the
# role rates. With n = 4 GPUs all issuing, a role at x GB/s of its own part per GPU moves, into / out of
# each GPU: st_multicast 4x / x, ld_reduce x / 4x, AllReduce 5x / 5x, plain load or store x / x.
# For one collective this is algBW x 1 (AllGather, ReduceScatter) or algBW x (1 + 1/n) (AllReduce),
# where algBW = n x. Measured link counters confirm the model on this machine (report, link data).
LINK_MODEL_IN = {'multicast': 4, 'reduce': 1, 'allreduce': 5, 'plain_load': 1, 'plain_store': 1}
LINK_MODEL_OUT = {'multicast': 1, 'reduce': 4, 'allreduce': 5, 'plain_load': 1, 'plain_store': 1}


def link_model_bus(row):
    """Only for arms where all 4 GPUs issue every role."""
    for values in row['roles'].values():
        if values['ranks_issuing'] != 4:
            raise ValueError(f"link-model busBW needs all 4 GPUs issuing: {row['run']}")
    received = sum(LINK_MODEL_IN[role] * values['rate_per_issuing_gpu_gbps'] for role, values in row['roles'].items())
    sent = sum(LINK_MODEL_OUT[role] * values['rate_per_issuing_gpu_gbps'] for role, values in row['roles'].items())
    return max(received, sent)


def ratio_summary(values):
    return {'per_round': values, 'median': statistics.median(values), 'minimum': min(values), 'maximum': max(values)}


def figure_calibration(rows, output):
    """Measured CTA scaling at both thread counts, one panel per role, shared scale."""
    figure, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    upper = zero_based_upper([link_model_bus(row) for row in rows.values()])
    for axis, (role, title) in zip(axes, ROLE_TITLES.items()):
        for threads, color, marker in ((256, SECOND_SERIES_COLOR, 's'), (512, REFERENCE_COLOR, 'o')):
            values = [link_model_bus(rows[f'e1_1_{role}_c{ctas}_t{threads}_R1']) for ctas in CTA_COUNTS]
            axis.plot(CTA_COUNTS, values, color=color, marker=marker, markersize=4, linewidth=1.5, label=f'{threads} threads/CTA')
        axis.axvline(32, color='#999999', linewidth=0.8, linestyle=':')
        axis.set_title(title, fontsize=10)
        axis.set_xlabel('Active CTAs of this role')
        axis.set_xticks(CTA_COUNTS)
        axis.tick_params(axis='x', labelsize=8)
        set_zero_based_y(axis, upper, 'Link-model busBW, GB/s' if axis is axes[0] else '')
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    figure.suptitle('Link-model busBW against active CTAs, 4 GPUs. One run per setting; the dotted line marks the chosen 32 CTAs', fontsize=11)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def draw_bars(axis, labels, heights, colors):
    bars = axis.bar(range(len(labels)), heights, color=colors, width=0.62)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, fontsize=8)
    return bars


def label_bars(axis, positions, heights):
    for position, height in zip(positions, heights):
        axis.text(position, height, f'{height:.0f}', ha='center', va='bottom', fontsize=8)


def legend_for_reference_and_mix(axis):
    handles = [plt.Rectangle((0, 0), 1, 1, color=REFERENCE_COLOR), plt.Rectangle((0, 0), 1, 1, color=PERTURBED_COLOR)]
    axis.legend(handles, ['Reference', 'Mixed'], frameon=False, fontsize=8, loc='upper right')


def figure_multimem(arms, output):
    """E1.2: link-model busBW, role byte rate and busiest measured link direction."""
    labels = ['NVLS\nAllGather', 'NVLS\nReduceScatter', 'AG + RS,\nseparate\nCTAs', 'NVLS\nAllReduce,\nsame CTAs']
    single_multicast = arms['e1_2_single_multicast']
    single_reduce = arms['e1_2_single_reduce']
    separate = arms['e1_2_mix_separate']
    same = arms['e1_2_mix_same']
    colors = [REFERENCE_COLOR, REFERENCE_COLOR, PERTURBED_COLOR, PERTURBED_COLOR]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    link_model = [arm['link_model_bus_median_gbps'] for arm in (single_multicast, single_reduce, separate, same)]
    draw_bars(axes[0], labels, link_model, colors)
    label_bars(axes[0], range(4), link_model)
    set_zero_based_y(axes[0], 500, 'Link-model busBW, GB/s')
    axes[0].set_title('Link-model busBW', fontsize=10)
    legend_for_reference_and_mix(axes[0])
    rate_labels = ['st_multicast\nin AG', 'st_multicast\nAG + RS', 'ld_reduce\nin RS', 'ld_reduce\nAG + RS', 'NVLS\nAllReduce']
    rates = [single_multicast['roles']['multicast']['rate_median_gbps'], separate['roles']['multicast']['rate_median_gbps'], single_reduce['roles']['reduce']['rate_median_gbps'], separate['roles']['reduce']['rate_median_gbps'], same['roles']['allreduce']['rate_median_gbps']]
    draw_bars(axes[1], rate_labels, rates, [REFERENCE_COLOR, PERTURBED_COLOR, REFERENCE_COLOR, PERTURBED_COLOR, PERTURBED_COLOR])
    axes[1].tick_params(axis='x', labelsize=6.5)
    label_bars(axes[1], range(5), rates)
    set_zero_based_y(axes[1], 120, 'Rate per GPU, GB/s')
    axes[1].set_title('Role byte rate per GPU', fontsize=10)
    busiest = [single_multicast['busiest_direction_gbps'], single_reduce['busiest_direction_gbps'], separate['busiest_direction_gbps'], same['busiest_direction_gbps']]
    draw_bars(axes[2], labels, busiest, colors)
    label_bars(axes[2], range(4), busiest)
    set_zero_based_y(axes[2], 500, 'Measured data on links, GB/s per GPU')
    axes[2].axhline(478, color='#555555', linewidth=0.8, linestyle='--')
    axes[2].text(3.4, 482, 'link speed 478', ha='right', va='bottom', fontsize=7)
    axes[2].set_title('Busiest measured link direction', fontsize=10)
    figure.suptitle(f'Multimem arms, each role {BASE_CONFIG}, 4 GPUs, medians of 6 rounds. AG = AllGather, RS = ReduceScatter', fontsize=11)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def figure_plain(arms, output):
    """E1.3: plain ring busBW, with the mix stacked by role."""
    load_alone = arms['e1_3_plain_load']['roles']['plain_load']['bus_median_gbps']
    store_alone = arms['e1_3_plain_store']['roles']['plain_store']['bus_median_gbps']
    mix_load = arms['e1_3_plain_mix']['roles']['plain_load']['bus_median_gbps']
    mix_store = arms['e1_3_plain_mix']['roles']['plain_store']['bus_median_gbps']
    figure, axis = plt.subplots(figsize=(5.4, 3.8))
    draw_bars(axis, ['Plain load\nalone', 'Plain store\nalone', 'Plain load\n+ plain store'], [load_alone, store_alone, mix_load + mix_store], [REFERENCE_COLOR, REFERENCE_COLOR, PERTURBED_COLOR])
    axis.bar(2, mix_load, color=PERTURBED_SECOND_COLOR, width=0.62)
    axis.text(2, mix_load / 2, 'plain load', ha='center', va='center', fontsize=7)
    axis.text(2, mix_load + mix_store / 2, 'plain store', ha='center', va='center', fontsize=7)
    label_bars(axis, range(3), [load_alone, store_alone, mix_load + mix_store])
    set_zero_based_y(axis, 500, 'Link-model busBW, GB/s')
    legend_for_reference_and_mix(axis)
    axis.set_title(f'Plain ring arms, each role {BASE_CONFIG},\n4 GPUs, medians of 6 rounds', fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def figure_split(arms, output):
    """E1.4: each role's rate alone and next to the other role on the other two GPUs."""
    labels = ['Multicast from\nranks 0, 1, alone', 'Multicast from\nranks 0, 1, split mix', 'Reduce by\nranks 2, 3, alone', 'Reduce by\nranks 2, 3, split mix']
    rates = [arms['e1_4_multicast_g01']['roles']['multicast']['rate_median_gbps'], arms['e1_4_split_mix']['roles']['multicast']['rate_median_gbps'], arms['e1_4_reduce_g23']['roles']['reduce']['rate_median_gbps'], arms['e1_4_split_mix']['roles']['reduce']['rate_median_gbps']]
    figure, axis = plt.subplots(figsize=(7, 3.8))
    draw_bars(axis, labels, rates, [REFERENCE_COLOR, PERTURBED_COLOR, REFERENCE_COLOR, PERTURBED_COLOR])
    label_bars(axis, range(4), rates)
    set_zero_based_y(axis, 250, 'Rate per issuing GPU, GB/s')
    legend_for_reference_and_mix(axis)
    axis.set_title(f'Different GPUs issue each half, multicast group of all 4 GPUs,\neach role {BASE_CONFIG}, medians of 6 rounds', fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def figure_nccl(harness, nccl, output):
    """E1.7: link-model busBW of NCCL collectives next to the harness role loops."""
    labels = ['AllGather', 'ReduceScatter', 'AllReduce']
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    positions = range(3)
    width = 0.36
    axis.bar([position - width / 2 for position in positions], nccl, width=width, color=REFERENCE_COLOR, label='NCCL, 4 CTAs × 512 threads/CTA (5 rounds)')
    axis.bar([position + width / 2 for position in positions], harness, width=width, color=SECOND_SERIES_COLOR, label='Harness, 32 CTAs × 512 threads/CTA (6 rounds)')
    label_bars(axis, [position - width / 2 for position in positions], nccl)
    label_bars(axis, [position + width / 2 for position in positions], harness)
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, fontsize=8)
    set_zero_based_y(axis, 500, 'Link-model busBW, GB/s')
    axis.legend(frameon=False, fontsize=8, loc='upper left')
    axis.set_title('Real NCCL collectives and the harness, 4 GPUs, medians', fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def figure_links(arms, output):
    """Measured link data per GPU in each direction, for the arms where all GPUs issue the same work."""
    selected = [('e1_2_single_multicast', 'NVLS\nAllGather'), ('e1_2_single_reduce', 'NVLS\nReduceScatter'), ('e1_2_mix_separate', 'AG + RS,\nseparate CTAs'), ('e1_2_mix_same', 'NVLS AllReduce,\nsame CTAs'), ('e1_3_plain_load', 'Plain load\nalone'), ('e1_3_plain_store', 'Plain store\nalone'), ('e1_3_plain_mix', 'Plain load\n+ plain store')]
    received = [statistics.fmean(arms[name]['link_rx_gbps_per_gpu']) for name, _ in selected]
    sent = [statistics.fmean(arms[name]['link_tx_gbps_per_gpu']) for name, _ in selected]
    figure, axis = plt.subplots(figsize=(9.5, 3.8))
    width = 0.36
    positions = list(range(len(selected)))
    axis.bar([position - width / 2 for position in positions], received, width=width, color='#4d4d4d', label='Data into each GPU')
    axis.bar([position + width / 2 for position in positions], sent, width=width, color='#b3b3b3', label='Data out of each GPU')
    label_bars(axis, [position - width / 2 for position in positions], received)
    label_bars(axis, [position + width / 2 for position in positions], sent)
    axis.axhline(478, color='#555555', linewidth=0.8, linestyle='--')
    axis.text(positions[-1] + 0.5, 482, 'link speed 478', ha='right', va='bottom', fontsize=7)
    axis.set_xticks(positions)
    axis.set_xticklabels([label for _, label in selected], fontsize=8)
    set_zero_based_y(axis, 600, 'GB/s per GPU')
    axis.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
    axis.set_title(f'Whole-run link data, each role {BASE_CONFIG}, 4 GPUs, medians of 6 rounds', fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def nccl_summary(results):
    rows = json.loads((results / 'e1_7' / 'e1_7_summary.json').read_text())
    medians = {}
    for collective in ('all_gather', 'reduce_scatter', 'all_reduce'):
        selected = [row for row in rows if row['collective'] == collective]
        buses = [row['out_of_place_busbw_gbps'] for row in selected]
        medians[collective] = {'bus_median_gbps': statistics.median(buses), 'bus_iqr_gbps': interquartile_range(buses), 'in_place_bus_median_gbps': statistics.median(row['in_place_busbw_gbps'] for row in selected), 'alg_median_gbps': statistics.median(row['out_of_place_algbw_gbps'] for row in selected), 'channels': sorted({value for row in selected for value in row['channel_values']}), 'threads': sorted({value for row in selected for value in row['thread_values']}), 'wrong': sum(row['out_of_place_wrong'] + row['in_place_wrong'] for row in selected), 'other_kernel_lines': sum(row['other_kernel_lines'] for row in selected)}
    for collective, factor in NCCL_LINK_MODEL_FACTOR.items():
        medians[collective]['link_model_bus_median_gbps'] = medians[collective]['alg_median_gbps'] * factor
    ratios = []
    link_ratios = []
    for round_index in range(1, 6):
        bus = {row['collective']: row['out_of_place_busbw_gbps'] for row in rows if row['run'].endswith(f'_R{round_index}')}
        ratios.append(bus['all_reduce'] / ((bus['all_gather'] + bus['reduce_scatter']) / 2))
        link = {row['collective']: row['out_of_place_algbw_gbps'] * NCCL_LINK_MODEL_FACTOR[row['collective']] for row in rows if row['run'].endswith(f'_R{round_index}')}
        link_ratios.append(link['all_reduce'] / ((link['all_gather'] + link['reduce_scatter']) / 2))
    return medians, ratio_summary(ratios), ratio_summary(link_ratios)


# N5 for real collectives: nccl-tests algBW (buffer bytes / time) times the link-model factor at n = 4.
NCCL_LINK_MODEL_FACTOR = {'all_gather': 1.0, 'reduce_scatter': 1.0, 'all_reduce': 1.25}


def main():
    results = Path(sys.argv[1])
    figures = results / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    calibration = load_rows(results, 'e1_1')
    multimem = load_rows(results, 'e1_2')
    plain = load_rows(results, 'e1_3')
    split = load_rows(results, 'e1_4')
    arms = {}
    for rows, names in ((multimem, ['e1_2_single_multicast', 'e1_2_single_reduce', 'e1_2_mix_separate', 'e1_2_mix_same']), (plain, ['e1_3_plain_load', 'e1_3_plain_store', 'e1_3_plain_mix']), (split, ['e1_4_multicast_g01', 'e1_4_reduce_g23', 'e1_4_split_mix'])):
        for name in names:
            arms[name] = describe_arm(rows_of_arm(rows, name))
    references = (('e1_2_single_multicast', bus_of('multicast')), ('e1_2_single_reduce', bus_of('reduce')))
    busiest_references = (('e1_2_single_multicast', busiest_direction), ('e1_2_single_reduce', busiest_direction))
    plain_references = (('e1_3_plain_load', bus_of('plain_load')), ('e1_3_plain_store', bus_of('plain_store')))
    plain_link_references = (('e1_3_plain_load', busiest_direction), ('e1_3_plain_store', busiest_direction))
    ratios = {
        'allreduce_bus_over_reference': ratio_summary(round_ratios(multimem, ('e1_2_mix_same', bus_of('allreduce')), references)),
        'separate_sum_bus_over_allreduce': ratio_summary(round_ratios(multimem, ('e1_2_mix_separate', summed_bus), (('e1_2_mix_same', bus_of('allreduce')),))),
        'multicast_rate_separate_over_alone': ratio_summary(round_ratios(multimem, ('e1_2_mix_separate', rate_of('multicast')), (('e1_2_single_multicast', rate_of('multicast')),))),
        'reduce_rate_separate_over_alone': ratio_summary(round_ratios(multimem, ('e1_2_mix_separate', rate_of('reduce')), (('e1_2_single_reduce', rate_of('reduce')),))),
        'allreduce_busiest_link_over_reference': ratio_summary(round_ratios(multimem, ('e1_2_mix_same', busiest_direction), busiest_references)),
        'separate_busiest_link_over_reference': ratio_summary(round_ratios(multimem, ('e1_2_mix_separate', busiest_direction), busiest_references)),
        'plain_sum_bus_over_reference': ratio_summary(round_ratios(plain, ('e1_3_plain_mix', summed_bus), plain_references)),
        'plain_busiest_link_over_reference': ratio_summary(round_ratios(plain, ('e1_3_plain_mix', busiest_direction), plain_link_references)),
        'split_multicast_rate_over_alone': ratio_summary(round_ratios(split, ('e1_4_split_mix', rate_of('multicast')), (('e1_4_multicast_g01', rate_of('multicast')),))),
        'split_reduce_rate_over_alone': ratio_summary(round_ratios(split, ('e1_4_split_mix', rate_of('reduce')), (('e1_4_reduce_g23', rate_of('reduce')),))),
    }
    link_model_references = (('e1_2_single_multicast', link_model_bus), ('e1_2_single_reduce', link_model_bus))
    ratios['allreduce_link_model_over_reference'] = ratio_summary(round_ratios(multimem, ('e1_2_mix_same', link_model_bus), link_model_references))
    ratios['separate_link_model_over_reference'] = ratio_summary(round_ratios(multimem, ('e1_2_mix_separate', link_model_bus), link_model_references))
    ratios['separate_link_model_over_allreduce'] = ratio_summary(round_ratios(multimem, ('e1_2_mix_separate', link_model_bus), (('e1_2_mix_same', link_model_bus),)))
    ratios['plain_link_model_over_reference'] = ratio_summary(round_ratios(plain, ('e1_3_plain_mix', link_model_bus), (('e1_3_plain_load', link_model_bus), ('e1_3_plain_store', link_model_bus))))
    calibration_table = {name: link_model_bus(row) for name, row in calibration.items()}
    nccl_medians, nccl_ratio, nccl_link_ratio = nccl_summary(results)
    ratios['nccl_allreduce_bus_over_reference'] = nccl_ratio
    ratios['nccl_allreduce_link_model_over_reference'] = nccl_link_ratio
    (results / 'report_summary.json').write_text(json.dumps({'arms': arms, 'ratios': ratios, 'nccl': nccl_medians, 'calibration_link_model_gbps': calibration_table}, indent=1) + '\n')
    figure_calibration(calibration, figures / 'calibration_cta_scaling.png')
    figure_multimem(arms, figures / 'multimem_arms.png')
    figure_plain(arms, figures / 'plain_arms.png')
    figure_split(arms, figures / 'split_mix.png')
    figure_links(arms, figures / 'link_data.png')
    harness_arms = ('e1_2_single_multicast', 'multicast'), ('e1_2_single_reduce', 'reduce'), ('e1_2_mix_same', 'allreduce')
    harness = {'nccl_tests': [arms[arm]['roles'][role]['bus_median_gbps'] for arm, role in harness_arms], 'link_model': [arms[arm]['link_model_bus_median_gbps'] for arm, _ in harness_arms]}
    collectives = ('all_gather', 'reduce_scatter', 'all_reduce')
    nccl = {'nccl_tests': [nccl_medians[name]['bus_median_gbps'] for name in collectives], 'link_model': [nccl_medians[name]['link_model_bus_median_gbps'] for name in collectives]}
    figure_nccl(harness['link_model'], nccl['link_model'], figures / 'nccl_cross_check.png')


if __name__ == '__main__':
    main()
