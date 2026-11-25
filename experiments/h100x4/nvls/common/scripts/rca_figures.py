#!/usr/bin/env python3
"""Figures for RCA plan 01 (rca/multimem_contention), from the derived CSVs of one results date.

Usage: rca_figures.py <results/<date> directory>
Each figure is drawn only when its CSV exists, and is written to <results>/figures/. Colours follow the
observation figures: alone (reference) blue, mixed orange; protocol and other non-data bytes in lighter
or grey tones. Rates and byte ratios start at zero; comparable panels share a scale.
"""
import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REFERENCE_COLOR = '#2f6fb0'
REFERENCE_LIGHT = '#9dbfe3'
PERTURBED_COLOR = '#e07b22'
PERTURBED_LIGHT = '#f3c08f'
NEUTRAL_COLOR = '#7f7f7f'
NEUTRAL_LIGHT = '#c8c8c8'


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def style(axis):
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.grid(axis='y', color='#dddddd', linewidth=0.6)
    axis.set_axisbelow(True)


def save(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    print(f'wrote {path}')


# E1.1: busiest direction of each arm, split into data and the rest of the raw bytes.
E1_1_ARMS = [
    ('e1_1_allgather', 'NVLS AllGather\nst_multicast 32×512', False),
    ('e1_1_reducescatter', 'NVLS ReduceScatter\nld_reduce 32×512', False),
    ('e1_1_allreduce', 'NVLS AllReduce\nboth on the same\n32×512 CTAs', True),
    ('e1_1_separate', 'Separate CTAs\nst_multicast 32×512\n+ ld_reduce 32×512', True),
    ('e1_1_plain_store', 'Plain store\n32×512', False),
    ('e1_1_plain_load', 'Plain load\n32×512', False),
    ('e1_1_plain_mix', 'Plain load 32×512\n+ plain store 32×512', True),
]


def figure_e1_1(results):
    path = results / 'e1_1' / 'summary.csv'
    if not path.exists():
        return
    rows = read_rows(path)
    figure, axis = plt.subplots(figsize=(10.5, 4.6))
    labels = []
    for position, (arm, label, mixed) in enumerate(E1_1_ARMS):
        members = [row for row in rows if row['arm'] == arm]
        busiest = max(members, key=lambda row: float(row['data_rate_gbps']))
        data_rate = float(busiest['data_rate_gbps'])
        raw_rate = float(busiest['raw_rate_gbps'])
        data_color, rest_color = (PERTURBED_COLOR, PERTURBED_LIGHT) if mixed else (REFERENCE_COLOR, REFERENCE_LIGHT)
        axis.bar(position, data_rate, width=0.6, color=data_color, edgecolor='white', linewidth=1)
        axis.bar(position, raw_rate - data_rate, bottom=data_rate, width=0.6, color=rest_color, edgecolor='white', linewidth=1)
        axis.text(position, data_rate / 2, f'{data_rate:.0f}', ha='center', va='center', color='white', fontsize=8)
        axis.text(position, raw_rate + 6, f'{raw_rate:.0f} raw\n×{float(busiest["raw_over_data"]):.2f}', ha='center', va='bottom', fontsize=7.5)
        labels.append(label)
    axis.set_xticks(range(len(labels)), labels, fontsize=7.5)
    axis.set_ylim(0, 500)
    axis.set_ylabel('GB/s in the busiest link direction, per GPU')
    axis.set_title('Data and other raw bytes in the busiest direction, 4 GPUs (4–7), CTAs × threads/CTA per role', fontsize=10)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=REFERENCE_COLOR), plt.Rectangle((0, 0), 1, 1, color=REFERENCE_LIGHT),
        plt.Rectangle((0, 0), 1, 1, color=PERTURBED_COLOR), plt.Rectangle((0, 0), 1, 1, color=PERTURBED_LIGHT),
    ]
    axis.legend(handles, ['Data, one operation', 'Other raw bytes, one operation', 'Data, mixed', 'Other raw bytes, mixed'], fontsize=7.5, ncol=4, loc='upper center', frameon=False)
    style(axis)
    save(figure, results / 'figures' / 'e1_1_raw_vs_data.png')


# E1.2: raw bytes per payload byte against the achieved data rate, busy and low-data directions.
def figure_e1_2(results):
    path = results / 'e1_2' / 'summary.csv'
    if not path.exists():
        return
    rows = read_rows(path)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    roles = [
        ('multicast', 'st_multicast alone (NVLS AllGather): busy = received', 'received', 'sent'),
        ('reduce', 'ld_reduce alone (NVLS ReduceScatter): busy = sent', 'sent', 'received'),
    ]
    for axis, (role, title, busy, idle) in zip(axes, roles):
        members = [row for row in rows if row['role_a'] == role]
        settings = sorted({row['arm'] for row in members}, key=lambda arm: float(next(row['data_rate_gbps'] for row in members if row['arm'] == arm and row['direction'] == busy)))
        for direction, color, name in ((busy, REFERENCE_COLOR, 'Busy direction'), (idle, NEUTRAL_COLOR, 'Low-data direction')):
            x_values = []
            y_values = []
            for arm in settings:
                busy_row = next(row for row in members if row['arm'] == arm and row['direction'] == busy)
                this_row = next(row for row in members if row['arm'] == arm and row['direction'] == direction)
                x_values.append(float(busy_row['data_rate_gbps']))
                y_values.append(float(this_row['raw_bytes_per_payload_byte']))
            axis.plot(x_values, y_values, color=color, linewidth=2, marker='o', markersize=5, label=name)
        for arm in settings:
            busy_row = next(row for row in members if row['arm'] == arm and row['direction'] == busy)
            axis.annotate(f"{busy_row['ctas_a']}×{busy_row['threads_per_cta']}", (float(busy_row['data_rate_gbps']), float(busy_row['raw_bytes_per_payload_byte'])), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=7)
        axis.set_xlim(0, 400)
        axis.set_ylim(0, 2.0)
        axis.set_xlabel('Achieved data rate, busy direction (GB/s per GPU)')
        axis.set_title(title, fontsize=9)
        style(axis)
    axes[0].set_ylabel('Raw link bytes per payload byte')
    axes[0].legend(fontsize=8, frameon=False, loc='lower left')
    figure.suptitle('Raw bytes per payload byte as the issue rate rises (labels: CTAs × threads/CTA), 4 GPUs (4–7)', fontsize=10)
    save(figure, results / 'figures' / 'e1_2_issue_rate.png')


PACKET_PARTS = [
    ('request_user_per_payload_byte', 'Request data', REFERENCE_COLOR),
    ('response_user_per_payload_byte', 'Response data', '#7aa6d6'),
    ('request_protocol_per_payload_byte', 'Request protocol', PERTURBED_COLOR),
    ('response_protocol_per_payload_byte', 'Response protocol', PERTURBED_LIGHT),
    ('raw_outside_packets_per_payload_byte', 'Raw counter bytes outside the packet counters', NEUTRAL_LIGHT),
]


def packet_composition(results, experiment, arms, title, filename):
    """Stacked bytes per payload byte per arm and direction (R1-R4 plus raw bytes the profiler does not count)."""
    path = results / experiment / 'nsys_summary.csv'
    if not path.exists():
        return
    rows = read_rows(path)
    figure, axis = plt.subplots(figsize=(11, 4.8))
    positions = []
    labels = []
    group_centers = []
    position = 0
    for arm, label in arms:
        group_start = position
        for direction in ('sent', 'received'):
            matches = [row for row in rows if row['arm'] == arm and row['direction'] == direction]
            if not matches:
                continue
            row = matches[0]
            bottom = 0.0
            for key, _name, color in PACKET_PARTS:
                value = max(float(row.get(key) or 0), 0.0)
                axis.bar(position, value, bottom=bottom, width=0.7, color=color, edgecolor='white', linewidth=1)
                bottom += value
            axis.text(position, bottom + 0.03, f'{bottom:.2f}', ha='center', va='bottom', fontsize=7)
            positions.append(position)
            labels.append(direction)
            position += 1
        group_centers.append(((group_start + position - 1) / 2, label))
        position += 0.6
    axis.set_xticks(positions, labels, fontsize=7.5)
    # Arm names sit below the pair of direction ticks, so long configurations do not collide.
    for center, label in group_centers:
        axis.text(center, -0.08, label, ha='center', va='top', fontsize=7.5, transform=axis.get_xaxis_transform())
    axis.set_ylim(0, 2.0)
    axis.set_ylabel('Bytes per payload byte')
    axis.set_title(title, fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _key, _name, color in PACKET_PARTS]
    axis.legend(handles, [name for _key, name, _color in PACKET_PARTS], fontsize=7.5, ncol=3, loc='upper left', frameon=False)
    style(axis)
    save(figure, results / 'figures' / filename)


def figure_e1_4(results):
    arms = [('e1_4_multicast', 'st_multicast\n32×512'), ('e1_4_reduce', 'ld_reduce\n32×512'), ('e1_4_plain_store', 'plain store\n32×512'), ('e1_4_plain_load', 'plain load\n32×512')]
    packet_composition(results, 'e1_4', arms, 'Packet bytes per payload byte, each operation alone, all 4 GPUs (4–7) issue, CTAs × threads/CTA', 'e1_4_packets_alone.png')


def figure_e1_6(results):
    arms = [('e1_6_allgather', 'AllGather\nst_multicast 32×512'), ('e1_6_reducescatter', 'ReduceScatter\nld_reduce 32×512'), ('e1_6_allreduce_same_ctas', 'AllReduce, same\n32×512 CTAs'), ('e1_6_separate_ctas', 'Separate CTAs\n32×512 + 32×512')]
    packet_composition(results, 'e1_6', arms, 'Packet bytes per payload byte, operations alone and mixed, 4 GPUs (4–7), CTAs × threads/CTA', 'e1_6_packets_mix.png')


def figure_e1_5(results):
    """Non-data bytes per payload byte and data rate against access width, for both address patterns."""
    path = results / 'e1_5' / 'nsys_summary.csv'
    if not path.exists():
        return
    rows = read_rows(path)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, role, title in ((axes[0], 'multicast', 'st_multicast alone (NVLS AllGather)'), (axes[1], 'reduce', 'ld_reduce alone (NVLS ReduceScatter)')):
        for pattern, color, name in (('contiguous', REFERENCE_COLOR, 'Warp-contiguous'), ('thread_contiguous', PERTURBED_COLOR, 'Thread-contiguous')):
            widths = []
            values = []
            for width in (4, 8, 16):
                members = [row for row in rows if row['role_a'] == role and row['pattern'] == pattern and int(row['access_bytes']) == width]
                if not members:
                    continue
                total = 0.0
                for row in members:
                    total += float(row['request_protocol_per_payload_byte']) + float(row['response_protocol_per_payload_byte']) + max(float(row.get('raw_outside_packets_per_payload_byte') or 0), 0.0)
                widths.append(width)
                values.append(total)
            axis.plot(widths, values, color=color, linewidth=2, marker='o', markersize=6, label=name)
        axis.set_xticks([4, 8, 16])
        axis.set_xlabel('Bytes per thread and access')
        axis.set_title(title + ', 32 CTAs × 512 threads/CTA', fontsize=9)
        style(axis)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel('Non-data bytes per payload byte,\nboth directions summed')
    axes[0].legend(fontsize=8, frameon=False)
    figure.suptitle('Protocol plus other raw bytes against access width, 4 GPUs (4–7)', fontsize=10)
    save(figure, results / 'figures' / 'e1_5_width_protocol.png')
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for axis, role, busy, title in ((axes[0], 'multicast', 'received', 'st_multicast alone'), (axes[1], 'reduce', 'sent', 'ld_reduce alone')):
        for pattern, color, name in (('contiguous', REFERENCE_COLOR, 'Warp-contiguous'), ('thread_contiguous', PERTURBED_COLOR, 'Thread-contiguous')):
            points = sorted((int(row['access_bytes']), float(row['user_rate_gbps'])) for row in rows if row['role_a'] == role and row['pattern'] == pattern and row['direction'] == busy)
            if points:
                axis.plot([point[0] for point in points], [point[1] for point in points], color=color, linewidth=2, marker='o', markersize=6, label=name)
        axis.set_xticks([4, 8, 16])
        axis.set_xlabel('Bytes per thread and access')
        axis.set_title(title + f', busy ({busy}) direction, 32×512', fontsize=9)
        style(axis)
    axes[0].set_ylim(0, 450)
    axes[0].set_ylabel('Data rate, GB/s per GPU')
    axes[0].legend(fontsize=8, frameon=False)
    save(figure, results / 'figures' / 'e1_5_width_rate.png')


def figure_e1_7(results):
    """Each operation's role byte rate against the other operation's CTA count (512 threads/CTA)."""
    path = results / 'e1_7' / 'role_rates.csv'
    if not path.exists():
        return
    rows = read_rows(path)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    sweeps = [
        (axes[0], 'reduce_varied', 'st_multicast fixed at 32 CTAs, ld_reduce CTAs varied', 'ld_reduce CTAs (512 threads/CTA)'),
        (axes[1], 'multicast_varied', 'ld_reduce fixed at 32 CTAs, st_multicast CTAs varied', 'st_multicast CTAs (512 threads/CTA)'),
    ]
    for axis, sweep, title, xlabel in sweeps:
        members = [row for row in rows if row['sweep'] == sweep]
        for role, color, name in (('multicast', REFERENCE_COLOR, 'st_multicast role'), ('reduce', PERTURBED_COLOR, 'ld_reduce role')):
            points = sorted((int(row['varied_ctas']), float(row['rate_gbps'])) for row in members if row['role'] == role)
            if points:
                axis.plot([point[0] for point in points], [point[1] for point in points], color=color, linewidth=2, marker='o', markersize=6, label=name)
        axis.set_xticks([0, 4, 8, 16, 32, 64])
        axis.set_xlabel(xlabel)
        axis.set_title(title, fontsize=9)
        style(axis)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Role byte rate per GPU, GB/s\n(own part; × 4 = data in the busy direction alone)')
    axes[0].legend(fontsize=8, frameon=False)
    figure.suptitle('How the two operations share the link, 4 GPUs (4–7)', fontsize=10)
    save(figure, results / 'figures' / 'e1_7_share.png')


def figure_e1_8(results):
    """Median and 90th-percentile probe latency per background."""
    path = results / 'e1_8' / 'probe_latency.csv'
    if not path.exists():
        return
    rows = read_rows(path)
    backgrounds = [('none', 'None'), ('allgather', 'st_multicast\n32×512'), ('reducescatter', 'ld_reduce\n32×512'), ('allreduce', 'AllReduce\n32×512'), ('plain_store', 'plain store\n32×512')]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, probe, title in ((axes[0], 'reduce', 'Probe: one ld_reduce at a time'), (axes[1], 'multicast', 'Probe: one st_multicast + system fence at a time')):
        for index, (background, label) in enumerate(backgrounds):
            members = [row for row in rows if row['probe'] == probe and row['background'] == background]
            if not members:
                continue
            median = statistics.median(float(row['probe_median_ns']) for row in members)
            p90 = statistics.median(float(row['probe_p90_ns']) for row in members)
            color = REFERENCE_COLOR if background == 'none' else PERTURBED_COLOR
            axis.bar(index, median, width=0.6, color=color, edgecolor='white')
            axis.plot([index - 0.2, index + 0.2], [p90, p90], color='black', linewidth=1.2)
            axis.text(index, p90 + 700, f'{median:.0f}', ha='center', va='bottom', fontsize=7.5)
        axis.set_xticks(range(len(backgrounds)), [label for _background, label in backgrounds], fontsize=7.5)
        axis.set_title(title, fontsize=9)
        style(axis)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel('Latency per operation, ns\n(bar: median; line: 90th percentile)')
    figure.suptitle('Single-operation latency under background traffic on all 4 GPUs (4–7), CTAs × threads/CTA', fontsize=10)
    save(figure, results / 'figures' / 'e1_8_latency.png')


def figure_e1_9(results):
    """E1.9 (8 GPUs, NCCL kernels): packet bytes per payload byte per kernel and direction.

    Raw-counter bytes outside the packet counters are not shown: the driver counter covers the whole
    nccl-tests process (input preparation and checks), not only the collective kernels.
    """
    path = results / 'e1_9' / 'nccl_summary.csv'
    if not path.exists():
        return
    rows = read_rows(path)
    cells = [
        ('e1_9_allgather_sym_multicast', 'Multicast AllGather\nAllGather_STMC'),
        ('e1_9_reducescatter_sym_multicast', 'Multicast ReduceScatter\nReduceScatter_LDMC'),
        ('e1_9_allreduce_sym_multicast', 'Multicast AllReduce\nAllReduce_RSxLDMC_AGxSTMC'),
        ('e1_9_allreduce_ring', 'Ring AllReduce\n(Simple protocol)'),
    ]
    figure, axis = plt.subplots(figsize=(11, 4.8))
    positions = []
    labels = []
    group_centers = []
    position = 0
    for cell, label in cells:
        group_start = position
        for direction in ('sent', 'received'):
            matches = [row for row in rows if row['cell'] == cell and row['direction'] == direction]
            if not matches:
                continue
            row = matches[0]
            bottom = 0.0
            for key, _name, color in PACKET_PARTS[:4]:
                value = float(row[key])
                axis.bar(position, value, bottom=bottom, width=0.7, color=color, edgecolor='white', linewidth=1)
                bottom += value
            axis.text(position, bottom + 0.03, f'{bottom:.2f}', ha='center', va='bottom', fontsize=7)
            positions.append(position)
            labels.append(direction)
            position += 1
        ctas = matches[0]['ctas'] if matches else ''
        if '..' in ctas:  # Ring records the channel range, for example '0..23' = 24 CTAs
            low, high = ctas.split('..')
            ctas = str(int(high) - int(low) + 1)
        group_centers.append(((group_start + position - 1) / 2, f'{label}\nNCCL: {ctas} CTAs × 512'))
        position += 0.6
    axis.set_xticks(positions, labels, fontsize=7.5)
    for center, label in group_centers:
        axis.text(center, -0.08, label, ha='center', va='top', fontsize=7.5, transform=axis.get_xaxis_transform())
    axis.set_ylim(0, 2.4)
    axis.set_ylabel('Packet bytes per payload byte')
    axis.set_title('Packet bytes per payload byte, NCCL kernels on 8 GPUs (0–7), 2 GiB', fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _key, _name, color in PACKET_PARTS[:4]]
    axis.legend(handles, [name for _key, name, _color in PACKET_PARTS[:4]], fontsize=7.5, ncol=4, loc='upper left', frameon=False)
    style(axis)
    save(figure, results / 'figures' / 'e1_9_packets_nccl.png')


def main():
    results = Path(sys.argv[1])
    figure_e1_1(results)
    figure_e1_2(results)
    figure_e1_4(results)
    figure_e1_5(results)
    figure_e1_6(results)
    figure_e1_7(results)
    figure_e1_8(results)
    figure_e1_9(results)


if __name__ == '__main__':
    main()
