#!/usr/bin/env python3
"""E7.0 analysis: group SMs that ran CTAs of the same cluster; write maps, a table and a figure.

Two SMs belong to the same GPC if they ever ran CTAs of one cluster. An SM that never ran a CTA
of a cluster with 4 or more CTAs cannot be placed this way: clusters of 2 CTAs may sit inside one
TPC (two SMs), so they do not connect TPCs. Such SMs get GPC -1 ("unmapped").
GPC numbers are given in order of each group's smallest SM ID.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from run_gpc_map import EXPERIMENT, RESULTS

MAPS = EXPERIMENT / 'configs/gpc_map'
FIGURES = RESULTS / 'figures'
# Categorical colours for GPCs; grey is unmapped. Blue/orange keep their healthy/perturbed meaning elsewhere.
GPC_COLORS = ['#3a923a', '#9372b2', '#c03d3e', '#845b53', '#2a9d9a', '#b8860b', '#5f6f8f', '#c2649a']
UNMAPPED_COLOR = '#d0d0d0'


def group_sms(probe):
    parent = list(range(probe['sm_count']))

    def find(sm):
        while parent[sm] != sm:
            parent[sm] = parent[parent[sm]]
            sm = parent[sm]
        return sm

    in_large_cluster = set()
    tpc_pairs = set()
    for size in probe['cluster_sizes']:
        for launch in size['launches']:
            clusters = {}
            for cluster, sm in launch:
                clusters.setdefault(cluster, []).append(sm)
            for sms in clusters.values():
                if size['cluster_size'] >= 4:
                    in_large_cluster.update(sms)
                if size['cluster_size'] == 2:
                    tpc_pairs.add(tuple(sorted(sms)))
                for sm in sms[1:]:
                    parent[find(sm)] = find(sms[0])
    groups = {}
    for sm in range(probe['sm_count']):
        if sm in in_large_cluster:
            groups.setdefault(find(sm), []).append(sm)
    ordered = sorted(groups.values(), key=min)
    gpc_of_sm = [-1] * probe['sm_count']
    for gpc, sms in enumerate(ordered):
        for sm in sms:
            gpc_of_sm[sm] = gpc
    return gpc_of_sm, ordered, sorted(tpc_pairs)


def build(gpu):
    probe = json.loads((RESULTS / 'gpc_map/raw' / f'gpu{gpu}.json').read_text())
    gpc_of_sm, groups, tpc_pairs = group_sms(probe)
    unmapped = [sm for sm, gpc in enumerate(gpc_of_sm) if gpc < 0]
    capacity = {size['cluster_size']: size['max_active_clusters'] for size in probe['cluster_sizes']}
    summary = dict(gpu=gpu, uuid=probe['uuid'], sm_count=probe['sm_count'], repeats=probe['repeats'],
                   max_active_clusters=capacity, gpc_sizes=[len(group) for group in groups],
                   unmapped_sms=unmapped, distinct_pairs_in_2_clusters=len(tpc_pairs),
                   gpcs={str(gpc): group for gpc, group in enumerate(groups)})
    MAPS.mkdir(parents=True, exist_ok=True)
    # The harness reads "uuid" and "gpc_of_sm"; it refuses a map whose UUID is not the source GPU's.
    content = dict(summary, source=f'results/2026-10-29/gpc_map/raw/gpu{gpu}.json', gpc_of_sm=gpc_of_sm)
    (MAPS / f'gpu{gpu}.json').write_text(json.dumps(content, indent=1).replace('\n  ', ' ') + '\n')
    return summary, gpc_of_sm


def draw(maps):
    FIGURES.mkdir(parents=True, exist_ok=True)
    columns = 12
    # Identical maps are drawn once; the title names every GPU that has this map.
    gpus = [summary['gpu'] for summary, _ in maps]
    if all(gpc_of_sm == maps[0][1] for _, gpc_of_sm in maps):
        maps = maps[:1]
    figure, axes = plt.subplots(1, len(maps), figsize=(6.2 * len(maps), 6.4))
    axes = axes if len(maps) > 1 else [axes]
    for axis, (summary, gpc_of_sm) in zip(axes, maps):
        for sm, gpc in enumerate(gpc_of_sm):
            row, column = divmod(sm, columns)
            color = GPC_COLORS[gpc % len(GPC_COLORS)] if gpc >= 0 else UNMAPPED_COLOR
            axis.add_patch(Rectangle((column, -row), .92, .92, color=color, linewidth=0))
            axis.text(column + .46, -row + .58, str(sm), ha='center', va='center', fontsize=7, color='white' if gpc >= 0 else '#444444')
            axis.text(column + .46, -row + .25, f'G{gpc}' if gpc >= 0 else '?', ha='center', va='center', fontsize=6, color='white' if gpc >= 0 else '#444444')
        rows = (len(gpc_of_sm) + columns - 1) // columns
        axis.set_xlim(-.1, columns)
        axis.set_ylim(-rows + .9, 1.02)
        axis.set_aspect('equal')
        axis.axis('off')
        sizes = ', '.join(f'G{gpc}: {size}' for gpc, size in enumerate(summary['gpc_sizes']))
        names = ', '.join(map(str, gpus)) + ' (identical maps)' if len(maps) == 1 and len(gpus) > 1 else str(summary['gpu'])
        axis.set_title(f"SM-to-GPC map from cluster launches, GPU {names}\neach box is one SM (SM ID, GPC)\n{sizes}; unmapped (grey): {len(summary['unmapped_sms'])}", fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES / 'gpc_map.png', dpi=150)
    plt.close(figure)


def main():
    maps = []
    for path in sorted((RESULTS / 'gpc_map/raw').glob('gpu?.json')):
        maps.append(build(int(path.stem[3:])))
    (RESULTS / 'gpc_map_summary.json').write_text(json.dumps([summary for summary, _ in maps], indent=1) + '\n')
    draw(maps)
    for summary, gpc_of_sm in maps:
        print(summary['gpu'], summary['gpc_sizes'], 'unmapped', summary['unmapped_sms'], summary['max_active_clusters'])
    if len(maps) == 2:
        print('same map on both GPUs:', maps[0][1] == maps[1][1])


if __name__ == '__main__':
    main()
