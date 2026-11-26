#!/usr/bin/env python3
"""Figures for GB200 plan 01 from summarize_nccl.py tables. Run where matplotlib exists (the container).

Usage: make_figures.py <tables_dir> --output <figures_dir>

  calibration_busbw_vs_ctas.png   E1.2: busBW (median over rounds, IQR bars) against CTAs per arm;
                                  left panel symmetric kernels, right panel channel-based kernels;
                                  copy-engine AllGather and the nominal link rate as reference lines
  payload_busbw_vs_size.png       E1.3: busBW against payload size per arm at the calibrated CTAs
                                  (only if e1_3 blocks are present)
  payload_ratios_vs_size.png      E1.3: AllReduce busBW / mean of the multicast halves, and algBW / Ring
  link_counters_e1_4.png          E1.4: data and raw-minus-data rate per arm and direction (counters.csv)

Chart rules (experiments/CLAUDE.md): zero-based y-axes with rounded headroom, shared y-scale across
panels, fixed colour per arm, thin marks, legend, labels carry the full configuration.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

NOMINAL_LINK_GBPS = 18 * 53.125
ARM_COLOURS = {
    'AG-MC': '#1f77b4', 'RS-MC': '#ff7f0e', 'AR-MC': '#d62728', 'AR-TMA': '#2ca02c', 'AR-LDST': '#9467bd',
    'AR-NVLS': '#8c564b', 'AR-Ring': '#e377c2', 'AG-Ring': '#7f7f7f', 'RS-Ring': '#bcbd22', 'AG-CE': '#17becf',
}
ARM_LABELS = {
    'AG-MC': 'AllGather, symmetric multimem.st (AllGather_STMC)',
    'RS-MC': 'ReduceScatter, symmetric multimem.ld_reduce (ReduceScatter_LDMC)',
    'AR-MC': 'AllReduce, symmetric both multimem (RSxLDMC_AGxSTMC)',
    'AR-TMA': 'AllReduce, symmetric TMA load/store (RSxTmaLD_AGxTmaST)',
    'AR-LDST': 'AllReduce, symmetric plain load/store (RSxLD_AGxST)',
    'AR-NVLS': 'AllReduce, NVLS (non-symmetric)',
    'AR-Ring': 'AllReduce, Ring Simple', 'AG-Ring': 'AllGather, Ring Simple', 'RS-Ring': 'ReduceScatter, Ring Simple',
    'AG-CE': 'AllGather, copy engine, unicast to 3 peers (no CTAs)',
}
SYMMETRIC_ARMS = ['AG-MC', 'RS-MC', 'AR-MC', 'AR-TMA', 'AR-LDST']
CHANNEL_ARMS = ['AR-NVLS', 'AR-Ring', 'AG-Ring', 'RS-Ring']
SIZE_BYTES = {'K': 1 << 10, 'M': 1 << 20, 'G': 1 << 30}


def readConfigs(path):
    with path.open() as handle:
        return [row for row in csv.DictReader(handle) if row['busbw_median_gbps']]


def roundedCeiling(value):
    """Upper y limit: next multiple of 100 GB/s above the value, with at least 5% headroom."""
    return 100 * math.ceil(value * 1.05 / 100)


def threadsOf(launched):
    """'AllReduce_RSxLDMC_AGxSTMC ctas=8 threads=512' -> '512'; channel arms do not log threads."""
    for part in launched.split():
        if part.startswith('threads='):
            return part.split('=')[1]
    return 'default'


def calibrationFigure(rows, output):
    calibration = [row for row in rows if row['block'] == 'e1_2']
    if not calibration:
        return None
    copyEngine = [float(row['busbw_median_gbps']) for row in calibration if row['arm'] == 'AG-CE']
    yMaximum = roundedCeiling(max([float(row['busbw_median_gbps']) for row in calibration] + [NOMINAL_LINK_GBPS]))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for axis, arms, title in ((axes[0], SYMMETRIC_ARMS, 'Symmetric kernels (CTAs = NCCL_SYM_CTAS)'),
                              (axes[1], CHANNEL_ARMS, 'Channel-based kernels (CTAs = channels)')):
        for arm in arms:
            points = sorted((int(row['ctas_requested']), float(row['busbw_median_gbps']), float(row['busbw_iqr_gbps'] or 0),
                             row['launched']) for row in calibration if row['arm'] == arm and row['ctas_requested'] != 'default')
            if not points:
                continue
            threads = threadsOf(points[-1][3])
            axis.errorbar([point[0] for point in points], [point[1] for point in points],
                          yerr=[point[2] / 2 for point in points], marker='o', markersize=3, linewidth=1,
                          capsize=2, color=ARM_COLOURS[arm], label=f'{ARM_LABELS[arm]}; threads/CTA {threads}')
        if copyEngine:
            axis.axhline(copyEngine[0], color=ARM_COLOURS['AG-CE'], linestyle='--', linewidth=1,
                         label=f"{ARM_LABELS['AG-CE']}: {copyEngine[0]:.0f} GB/s")
        axis.axhline(NOMINAL_LINK_GBPS, color='black', linestyle=':', linewidth=1,
                     label=f'Nominal link rate per direction, 18 x 53.125 = {NOMINAL_LINK_GBPS:.0f} GB/s')
        axis.set_xscale('log', base=2)
        axis.set_xlabel('CTAs per GPU')
        axis.set_title(title, fontsize=10)
        axis.set_ylim(0, yMaximum)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, loc='lower right')
    axes[0].set_ylabel('busBW per direction, GB/s (median of 3 rounds; bar = IQR)')
    figure.suptitle('CTA calibration, 4 GB200 GPUs on one node, 2 GiB per buffer, float', fontsize=11)
    figure.tight_layout()
    path = output / 'calibration_busbw_vs_ctas.png'
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def payloadFigure(rows, output):
    payload = [row for row in rows if row['block'].startswith('e1_3_calibrated')]
    if not payload:
        return None
    yMaximum = roundedCeiling(max([float(row['busbw_median_gbps']) for row in payload] + [NOMINAL_LINK_GBPS]))
    figure, axis = plt.subplots(figsize=(10, 6))
    for arm in ARM_COLOURS:
        points = sorted((int(row['size'][:-1]) * SIZE_BYTES[row['size'][-1]], float(row['busbw_median_gbps']),
                         float(row['busbw_iqr_gbps'] or 0), row['ctas_requested'], row['launched'])
                        for row in payload if row['arm'] == arm)
        if not points:
            continue
        ctas = points[0][3]
        label = f"{ARM_LABELS[arm]}; {ctas} CTAs x {threadsOf(points[0][4])} threads/CTA" if ctas != 'default' else ARM_LABELS[arm]
        axis.errorbar([point[0] for point in points], [point[1] for point in points],
                      yerr=[point[2] / 2 for point in points], marker='o', markersize=3, linewidth=1, capsize=2,
                      color=ARM_COLOURS[arm], label=label)
    axis.axhline(NOMINAL_LINK_GBPS, color='black', linestyle=':', linewidth=1, label='Nominal link rate per direction')
    axis.set_xscale('log', base=2)
    axis.set_xlabel('Payload per buffer, bytes')
    axis.set_ylabel('busBW per direction, GB/s (median of rounds; bar = IQR)')
    axis.set_ylim(0, yMaximum)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, loc='lower right')
    axis.set_title('Throughput against payload, calibrated CTAs, 4 GB200 GPUs on one node, float', fontsize=11)
    figure.tight_layout()
    path = output / 'payload_busbw_vs_size.png'
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def ratioFigure(rows, output):
    """E1.3: mixing ratio (G3) and gain over Ring (G4) against payload, from calibrated medians."""
    payload = [row for row in rows if row['block'].startswith('e1_3_calibrated')]
    if not payload:
        return None
    lookup = {(row['arm'], row['size']): row for row in payload}
    sizes = sorted({row['size'] for row in payload}, key=lambda size: int(size[:-1]) * SIZE_BYTES[size[-1]])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for arm, colour in (('AR-MC', ARM_COLOURS['AR-MC']), ('AR-NVLS', ARM_COLOURS['AR-NVLS'])):
        xs, mixing = [], []
        for size in sizes:
            needed = [(arm, size), ('AG-MC', size), ('RS-MC', size)]
            if all(key in lookup for key in needed):
                halves = (float(lookup[('AG-MC', size)]['busbw_median_gbps']) + float(lookup[('RS-MC', size)]['busbw_median_gbps'])) / 2
                xs.append(int(size[:-1]) * SIZE_BYTES[size[-1]])
                mixing.append(float(lookup[(arm, size)]['busbw_median_gbps']) / halves)
        axes[0].plot(xs, mixing, marker='o', markersize=3, linewidth=1, color=colour, label=ARM_LABELS[arm])
    axes[0].axhline(1.0, color='black', linestyle=':', linewidth=1, label='No loss (1.0)')
    axes[0].set_title('AllReduce busBW ÷ mean busBW of the multicast AllGather and ReduceScatter', fontsize=9)
    for arm in ('AR-MC', 'AR-NVLS', 'AR-TMA', 'AR-LDST'):
        xs, gain = [], []
        for size in sizes:
            if (arm, size) in lookup and ('AR-Ring', size) in lookup:
                xs.append(int(size[:-1]) * SIZE_BYTES[size[-1]])
                gain.append(float(lookup[(arm, size)]['algbw_median_gbps']) / float(lookup[('AR-Ring', size)]['algbw_median_gbps']))
        axes[1].plot(xs, gain, marker='o', markersize=3, linewidth=1, color=ARM_COLOURS[arm], label=ARM_LABELS[arm])
    axes[1].axhline(1.0, color='black', linestyle=':', linewidth=1, label='Same as Ring (1.0)')
    axes[1].axhline(1.2, color='grey', linestyle='--', linewidth=1, label='Byte-count bound for multicast at 4 GPUs (1.2)')
    axes[1].set_title('AllReduce algBW ÷ Ring AllReduce algBW', fontsize=9)
    for axis in axes:
        axis.set_xscale('log', base=2)
        axis.set_xlabel('Payload per buffer, bytes')
        axis.set_ylim(0, 1.4)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, loc='lower right')
    axes[0].set_ylabel('Ratio (medians of 5 rounds)')
    figure.suptitle('Each arm at its calibrated CTA count, 4 GB200 GPUs on one node, float', fontsize=11)
    figure.tight_layout()
    path = output / 'payload_ratios_vs_size.png'
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def countersFigure(tables, output):
    """E1.4: per arm and direction, data rate and the rest of the raw link rate (stacked bars)."""
    path = tables / 'counters.csv'
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['block'] == 'e1_4']
    if not rows:
        return None
    groups = {}
    for row in rows:
        groups.setdefault((row['arm'], row['ctas_requested'], row['direction']), []).append(row)
    order = ['AG-MC', 'RS-MC', 'AR-MC', 'AR-NVLS', 'AR-TMA', 'AR-Ring', 'AG-CE']
    labels, data, other, colours = [], [], [], []
    for arm in order:
        for direction, name in (('rx', 'received'), ('tx', 'sent')):
            members = [members for key, members in groups.items() if key[0] == arm and key[2] == direction]
            if not members:
                continue
            members = members[0]
            dataRate = statistics.median(float(member['data_rate_gbps']) for member in members)
            rawRate = statistics.median(float(member['raw_link_rate_gbps']) for member in members)
            ctas = members[0]['ctas_requested']
            ctasText = 'copy engine' if ctas == 'default' else f'{ctas} CTAs'
            labels.append(f'{arm} ({ctasText}), {name}')
            data.append(dataRate)
            other.append(rawRate - dataRate)
            colours.append(ARM_COLOURS[arm])
    figure, axis = plt.subplots(figsize=(10, 6.5))
    positions = range(len(labels))
    axis.barh(positions, data, color=colours, height=0.6, label='Data (data counter)')
    axis.barh(positions, other, left=data, color='lightgrey', edgecolor='grey', height=0.6,
              label='Raw minus data (headers, requests, responses)')
    axis.axvline(NOMINAL_LINK_GBPS, color='black', linestyle=':', linewidth=1, label='Nominal link rate per direction')
    axis.set_yticks(list(positions))
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0, roundedCeiling(NOMINAL_LINK_GBPS))
    axis.set_xlabel('Rate per GPU and direction, GB/s (median of 5 rounds x 4 GPUs; long minus short runs)')
    axis.grid(alpha=0.3, axis='x')
    axis.legend(fontsize=8, loc='upper right')
    axis.set_title('NVLink driver counters at calibrated CTA counts, 2 GiB, 4 GB200 GPUs on one node', fontsize=11)
    figure.tight_layout()
    path = output / 'link_counters_e1_4.png'
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('tables', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    rows = readConfigs(arguments.tables / 'configs.csv')
    for path in (calibrationFigure(rows, arguments.output), payloadFigure(rows, arguments.output),
                 ratioFigure(rows, arguments.output), countersFigure(arguments.tables, arguments.output)):
        if path:
            print(f'wrote {path}')


if __name__ == '__main__':
    main()
