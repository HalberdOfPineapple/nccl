#!/usr/bin/env python3
"""Raw against data NVLink bytes per payload byte, per arm and direction, from link_bytes.py outputs of the same runs.

Inputs: two CSVs written by `link_bytes.py ... --counter data` and `--counter raw` for the same JSONL. Output: one row per
(collective, arm, GPU count, direction) with medians over GPUs and pairs, the raw/data ratio, and the data and raw link
rates (algBW x bytes per payload byte). Raw bytes include packet headers and request packets; data bytes are payload only.
Usage: raw_vs_data.py DATA_CSV RAW_CSV OUTPUT_CSV [OUTPUT_PNG]   (the figure needs matplotlib: run through uv)
"""
import csv
import statistics
import sys

LINK_RATING_GBPS = 478.0


def grouped(path):
    groups = {}
    for row in csv.DictReader(open(path)):
        # AllReduce rows of link_bytes.py carry no collective (empty when the CSV also holds other collectives).
        key = (row.get('collective') or 'allreduce', row['arm'], int(row['gpu_count']), row['direction'])
        groups.setdefault(key, []).append(row)
    return groups


def main():
    data, raw = grouped(sys.argv[1]), grouped(sys.argv[2])
    rows = []
    for key in sorted(data):
        collective, arm, gpu_count, direction = key
        data_value = statistics.median(float(row['link_bytes_per_payload_byte_median']) for row in data[key])
        raw_value = statistics.median(float(row['link_bytes_per_payload_byte_median']) for row in raw[key])
        algbw = statistics.median(float(row['algbw_long_run_gbps']) for row in data[key])
        rows.append({'collective': collective, 'arm': arm, 'gpu_count': gpu_count, 'direction': direction, 'pairs': len(data[key]),
                     'algbw_long_run_gbps': round(algbw, 1), 'data_bytes_per_payload_byte': round(data_value, 4),
                     'raw_bytes_per_payload_byte': round(raw_value, 4), 'raw_over_data': round(raw_value / data_value, 3) if data_value else '',
                     'data_link_rate_gbps': round(algbw * data_value, 1), 'raw_link_rate_gbps': round(algbw * raw_value, 1),
                     'raw_share_of_rating': round(algbw * raw_value / LINK_RATING_GBPS, 3)})
    with open(sys.argv[3], 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    if len(sys.argv) > 4:
        figure(rows, sys.argv[4])


def figure(rows, output):
    """Per pattern and direction: data and raw link rate, with the 478 GB/s rating and the raw/data ratio above each pair."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = {('allreduce', 'ring'): 'Ring\nAllReduce', ('allgather', 'ring'): 'Ring\nAllGather', ('allreduce', 'sym_multicast'): 'Multicast\nAllReduce',
             ('allgather', 'sym_multicast'): 'Multicast\nAllGather', ('reducescatter', 'sym_multicast'): 'Multicast\nReduceScatter'}
    order = [('allreduce', 'ring'), ('allgather', 'ring'), ('allreduce', 'sym_multicast'), ('allgather', 'sym_multicast'), ('reducescatter', 'sym_multicast')]
    figure_handle, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for axis, direction, title in zip(axes, ('Tx', 'Rx'), ('Sent (Tx)', 'Received (Rx)')):
        for position, key in enumerate(order):
            row = [item for item in rows if (item['collective'], item['arm']) == key and item['direction'] == direction][0]
            axis.bar(position - 0.2, row['data_link_rate_gbps'], width=0.4, color='#4c72b0', label='Data payload' if position == 0 else None)
            axis.bar(position + 0.2, row['raw_link_rate_gbps'], width=0.4, color='#dd8452', label='Raw (payload + headers + requests)' if position == 0 else None)
            axis.text(position + 0.2, row['raw_link_rate_gbps'] + 6, f"x{row['raw_over_data']:.2f}", ha='center', fontsize=7.5)
        axis.axhline(478, color='grey', linewidth=0.8, linestyle=':')
        axis.set_xticks(range(len(order)))
        axis.set_xticklabels([names[key] for key in order], fontsize=8)
        axis.set_ylim(0, 580)
        axis.grid(alpha=0.3, axis='y')
        axis.set_title(f'{title}, 8 GPUs {{0-7}}, 2 GiB', fontsize=10)
    axes[0].set_ylabel('Link rate per GPU, GB/s')
    axes[0].legend(fontsize=7.5, loc='upper right', ncol=2)
    axes[0].text(-0.4, 470, 'NVLink rating 478 GB/s', fontsize=7, color='grey', va='top')
    figure_handle.tight_layout()
    caption = ('Link rate = algBW x link bytes per payload byte (long-minus-short pairs, 3 rounds, median over GPUs). Label over the raw bar: raw / data. '
               "CTAs chosen by NCCL: Ring 24 x 512, multicast 2 x 512.")
    figure_handle.text(0.5, -0.02, caption, ha='center', va='top', fontsize=8)
    figure_handle.savefig(output, dpi=130, bbox_inches='tight')


if __name__ == '__main__':
    main()
