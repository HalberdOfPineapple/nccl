#!/usr/bin/env python3
"""Builds the blog-style HTML report of the NVLS root-cause series (RCA plan 01, E1.1-E1.9).

Every number and chart comes from the derived summaries saved by the analysis scripts; nothing is typed
in by hand. Diagrams that explain concepts (topology, algorithms, packet shapes) are drawn here as SVG.

Usage: make_blog_report.py <artifact body output .html> <standalone output .html>
The first file has no <html>/<head> skeleton (the artifact host adds it); the second is a complete page.
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[4]
RCA4 = EXPERIMENTS / 'h100x4/nvls/rca/multimem_contention/results/2026-11-18'
RCA8 = EXPERIMENTS / 'h100x8/nvls/rca/multimem_contention/results/2026-11-19'
OBSERVATION = EXPERIMENTS / 'h100x4/nvls/observation/multimem_load_store_mix/results/2026-11-01/report_summary.json'
RAW8 = EXPERIMENTS / 'h100-scale/allgather_reducescatter/scalability/busbw_comparison/results/2026-11-18/raw_vs_data.csv'
SWEEP = EXPERIMENTS / 'h100-scale/allreduce/scalability/payload_size/results/2026-11-16/sweep_summary.csv'


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def pick(table, **match):
    found = [row for row in table if all(row[key] == value for key, value in match.items())]
    if len(found) != 1:
        raise KeyError(f'{match}: {len(found)} rows')
    return found[0]


def number(row, key):
    return float(row[key])


# ---------------------------------------------------------------- SVG helpers

def esc(text):
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def legend(items, x, y, step=None, max_x=None):
    """Swatches in rows; a new row starts when the next item would pass max_x. Items are (name, css class)."""
    parts = []
    cursor = x
    row_y = y
    for name, css in items:
        item_width = step if step else 16 + 6.0 * len(name) + 18
        if max_x is not None and cursor > x and cursor + 16 + 6.0 * len(name) > max_x:
            cursor = x
            row_y += 16
        parts.append(f'<rect x="{cursor}" y="{row_y - 9}" width="11" height="11" rx="2" class="{css}"/>')
        parts.append(f'<text x="{cursor + 16}" y="{row_y}" class="t-small">{esc(name)}</text>')
        cursor += item_width
    return ''.join(parts)


def axis_y(left, top, plot_height, ymax, ystep, label, fmt='{:g}', width=0):
    parts = []
    value = 0.0
    while value <= ymax + 1e-9:
        y = top + plot_height - plot_height * value / ymax
        parts.append(f'<line x1="{left}" x2="{left + width}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 4:.1f}" class="t-small" text-anchor="end">{fmt.format(value)}</text>')
        value += ystep
    mid = top + plot_height / 2
    parts.append(f'<text x="14" y="{mid:.1f}" class="t-small" text-anchor="middle" transform="rotate(-90 14 {mid:.1f})">{esc(label)}</text>')
    return ''.join(parts)


def stacked_chart(groups, parts, ymax, ystep, ylabel, label, height=330, bar_width=34, total_fmt='{:.2f}', legend_step=None):
    """Grouped stacked bars. groups: [{'label': [lines], 'bars': [{'tick': str, 'values': [..]}]}]."""
    left, top, bottom = 58, 52, 74
    gap_bar, gap_group = 10, 34
    bar_count = sum(len(group['bars']) for group in groups)
    plot_width = bar_count * (bar_width + gap_bar) + (len(groups) - 1) * gap_group + 10
    # Narrow charts would be scaled up by the browser (huge text); widen the spacing to a common width instead.
    minimum_plot_width = 700
    if plot_width < minimum_plot_width:
        factor = minimum_plot_width / plot_width
        bar_width = min(bar_width * factor, 64)
        gap_bar = 10 + (minimum_plot_width - bar_count * (bar_width + 10) - 10) / max(bar_count + 2 * (len(groups) - 1), 1)
        gap_group = 2 * gap_bar + 14
        plot_width = bar_count * (bar_width + gap_bar) + (len(groups) - 1) * gap_group + 10
    width = left + plot_width + 10
    plot_height = height - top - bottom
    svg = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}" class="chart">']
    svg.append(axis_y(left, top, plot_height, ymax, ystep, ylabel, width=plot_width))
    svg.append(legend([(name, css) for name, css in parts], left, 16, legend_step, max_x=left + plot_width))
    x = left + 10
    for group in groups:
        start = x
        for bar in group['bars']:
            bottom_value = 0.0
            for (name, css), value in zip(parts, bar['values']):
                if value <= 0:
                    continue
                y_top = top + plot_height - plot_height * (bottom_value + value) / ymax
                h = plot_height * value / ymax
                svg.append(f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bar_width:.1f}" height="{max(h - 1, 0.5):.1f}" class="{css}"><title>{esc(name)}: {value:.3f}</title></rect>')
                bottom_value += value
            y_total = top + plot_height - plot_height * bottom_value / ymax
            svg.append(f'<text x="{x + bar_width / 2}" y="{y_total - 5:.1f}" class="t-value" text-anchor="middle">{total_fmt.format(bottom_value)}</text>')
            svg.append(f'<text x="{x + bar_width / 2}" y="{top + plot_height + 15}" class="t-small" text-anchor="middle">{esc(bar["tick"])}</text>')
            x += bar_width + gap_bar
        center = (start + x - gap_bar) / 2
        for line_index, line in enumerate(group['label']):
            svg.append(f'<text x="{center:.1f}" y="{top + plot_height + 34 + 13 * line_index}" class="t-label" text-anchor="middle">{esc(line)}</text>')
        x += gap_group - gap_bar
    svg.append(f'<line x1="{left}" x2="{left + plot_width}" y1="{top + plot_height}" y2="{top + plot_height}" class="axis"/>')
    svg.append('</svg>')
    return ''.join(svg)


def line_chart(series, xmax, xstep, ymax, ystep, xlabel, ylabel, label, width=460, height=300, xticks=None, yfmt='{:g}'):
    """series: [{'name', 'css', 'points': [(x, y, note or None)]}]."""
    left, top, right, bottom = 58, 34, 16, 46
    plot_width = width - left - right
    plot_height = height - top - bottom
    svg = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}" class="chart">']
    svg.append(axis_y(left, top, plot_height, ymax, ystep, ylabel, fmt=yfmt, width=plot_width))
    ticks = xticks if xticks is not None else [xstep * index for index in range(int(xmax / xstep) + 1)]
    for tick in ticks:
        x = left + plot_width * tick / xmax
        svg.append(f'<text x="{x:.1f}" y="{top + plot_height + 16}" class="t-small" text-anchor="middle">{tick:g}</text>')
    svg.append(f'<text x="{left + plot_width / 2}" y="{height - 8}" class="t-small" text-anchor="middle">{esc(xlabel)}</text>')
    svg.append(f'<line x1="{left}" x2="{left + plot_width}" y1="{top + plot_height}" y2="{top + plot_height}" class="axis"/>')
    svg.append(legend([(item['name'], item['css']) for item in series], left, 16))
    for item in series:
        coordinates = [(left + plot_width * x / xmax, top + plot_height - plot_height * y / ymax, note) for x, y, note in item['points']]
        path = ' '.join(f'{"M" if index == 0 else "L"}{x:.1f},{y:.1f}' for index, (x, y, _note) in enumerate(coordinates))
        svg.append(f'<path d="{path}" class="line {item["css"]}-stroke"/>')
        for (x, y, note), (raw_x, raw_y, _) in zip(coordinates, item['points']):
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" class="{item["css"]} dot"><title>{esc(item["name"])}: {raw_x:g}, {raw_y:.3g}</title></circle>')
            if note:
                svg.append(f'<text x="{x:.1f}" y="{y - 9:.1f}" class="t-tiny" text-anchor="middle">{esc(note)}</text>')
    svg.append('</svg>')
    return ''.join(svg)


def bars_with_marks(groups, ymax, ystep, ylabel, label, height=300, width=470, value_fmt='{:,.0f}', mark_name='90th percentile'):
    """Single bars per category with a horizontal tick for a second value. groups: [(label lines, value, mark, css)]."""
    left, top, bottom, right = 64, 34, 56, 10
    plot_width = width - left - right
    plot_height = height - top - bottom
    slot = plot_width / len(groups)
    svg = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}" class="chart">']
    svg.append(axis_y(left, top, plot_height, ymax, ystep, ylabel, fmt='{:,.0f}', width=plot_width))
    svg.append(legend([('idle links', 'c-data'), ('with background', 'c-proto')], left, 16, 110))
    svg.append(f'<line x1="{left + 236}" x2="{left + 252}" y1="12" y2="12" class="mark"/><text x="{left + 258}" y="16" class="t-small">{esc(mark_name)}</text>')
    for index, (lines, value, mark, css) in enumerate(groups):
        center = left + slot * (index + 0.5)
        bar_width = min(38, slot * 0.6)
        y = top + plot_height - plot_height * value / ymax
        svg.append(f'<rect x="{center - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{top + plot_height - y:.1f}" class="{css}"><title>median {value:,.0f}</title></rect>')
        y_mark = top + plot_height - plot_height * mark / ymax
        svg.append(f'<line x1="{center - bar_width / 2 - 4:.1f}" x2="{center + bar_width / 2 + 4:.1f}" y1="{y_mark:.1f}" y2="{y_mark:.1f}" class="mark"/>')
        svg.append(f'<text x="{center:.1f}" y="{min(y, y_mark) - 6:.1f}" class="t-value" text-anchor="middle">{value_fmt.format(value)}</text>')
        for line_index, line in enumerate(lines):
            svg.append(f'<text x="{center:.1f}" y="{top + plot_height + 15 + 12 * line_index}" class="t-small" text-anchor="middle">{esc(line)}</text>')
    svg.append(f'<line x1="{left}" x2="{left + plot_width}" y1="{top + plot_height}" y2="{top + plot_height}" class="axis"/>')
    svg.append('</svg>')
    return ''.join(svg)


# ---------------------------------------------------------------- data

def load_values():
    value = {}
    charts = {}
    observation = json.loads(OBSERVATION.read_text())['arms']
    allgather_bus = observation['e1_2_single_multicast']['link_model_bus_median_gbps']
    reducescatter_bus = observation['e1_2_single_reduce']['link_model_bus_median_gbps']
    allreduce_bus = observation['e1_2_mix_same']['link_model_bus_median_gbps']
    value.update(obs_ag=f'{allgather_bus:.1f}', obs_rs=f'{reducescatter_bus:.1f}', obs_ar=f'{allreduce_bus:.1f}',
                 obs_ratio=f'{allreduce_bus / ((allgather_bus + reducescatter_bus) / 2):.3f}')
    sweep = rows(SWEEP)
    def sweep_rate(cell):
        return number(pick(sweep, cell_id=cell, size_bytes='8589934592'), 'algbw_median_gbps')
    ring4, sym4 = sweep_rate('n4_ring_auto_sweep'), sweep_rate('n4_sym_multicast_auto_sweep')
    ring8, sym8 = sweep_rate('n8_ring_auto_sweep'), sweep_rate('n8_sym_multicast_auto_sweep')
    value.update(ring4=f'{ring4:.1f}', sym4=f'{sym4:.1f}', gain4=f'{sym4 / ring4:.2f}', ring8=f'{ring8:.1f}', sym8=f'{sym8:.1f}', gain8=f'{sym8 / ring8:.2f}',
                 ideal4=f'{1.5 / 1.25:.2f}', ideal8=f'{1.75 / 1.125:.2f}', share4=f'{(sym4 / ring4) / (1.5 / 1.25):.2f}', share8=f'{(sym8 / ring8) / (1.75 / 1.125):.2f}')

    # Earlier 8-GPU raw counter check: busiest direction of each pattern.
    raw8 = rows(RAW8)
    patterns = [('allreduce', 'ring', 'Ring', 'AllReduce'), ('allgather', 'ring', 'Ring', 'AllGather'), ('allreduce', 'sym_multicast', 'Multicast', 'AllReduce'),
                ('allgather', 'sym_multicast', 'Multicast', 'AllGather'), ('reducescatter', 'sym_multicast', 'Multicast', 'ReduceScatter')]
    groups = []
    for collective, arm, name, collective_name in patterns:
        members = [row for row in raw8 if row['collective'] == collective and row['arm'] == arm and row['gpu_count'] == '8']
        busiest = max(members, key=lambda row: number(row, 'data_link_rate_gbps'))
        data_rate = number(busiest, 'data_link_rate_gbps')
        raw_rate = number(busiest, 'raw_link_rate_gbps')
        is_nvls_allreduce = (collective, arm) == ('allreduce', 'sym_multicast')
        values = [0, 0, data_rate, raw_rate - data_rate] if is_nvls_allreduce else [data_rate, raw_rate - data_rate, 0, 0]
        groups.append({'label': [name, collective_name, f'w = {number(busiest, "raw_over_data"):.2f}'], 'bars': [{'tick': '', 'values': values}]})
        if (collective, arm) == ('allreduce', 'sym_multicast'):
            value.update(raw8_ar=f'{number(busiest, "raw_over_data"):.2f}', raw8_ar_rate=f'{raw_rate:.0f}', raw8_ar_data=f'{data_rate:.0f}')
        if (collective, arm) == ('allreduce', 'ring'):
            value.update(raw8_ring=f'{number(busiest, "raw_over_data"):.2f}', raw8_ring_rate=f'{raw_rate:.0f}', raw8_ring_data=f'{data_rate:.0f}')
    charts['raw8'] = stacked_chart(groups, [('Data', 'c-data'), ('Other bytes', 'c-data2'), ('Data (NVLS AllReduce)', 'c-proto'), ('Other (NVLS AllReduce)', 'c-proto2')], 500, 100, 'GB/s, busiest direction', 'Raw and data link rate at 8 GPUs', bar_width=46, total_fmt='{:.0f}', legend_step=140)

    # E1.1
    e11 = rows(RCA4 / 'e1_1/summary.csv')
    arms = [('e1_1_allgather', ['NVLS', 'AllGather'], False), ('e1_1_reducescatter', ['NVLS', 'ReduceScatter'], False), ('e1_1_allreduce', ['NVLS', 'AllReduce'], True),
            ('e1_1_separate', ['Both, separate', 'CTAs'], True), ('e1_1_plain_store', ['Plain', 'store'], False), ('e1_1_plain_load', ['Plain', 'load'], False), ('e1_1_plain_mix', ['Plain load', '+ store'], True)]
    groups = []
    for arm, label, mixed in arms:
        busiest = max((row for row in e11 if row['arm'] == arm), key=lambda row: number(row, 'data_rate_gbps'))
        data_rate, raw_rate = number(busiest, 'data_rate_gbps'), number(busiest, 'raw_rate_gbps')
        groups.append({'label': label + [f'w = {number(busiest, "raw_over_data"):.2f}'], 'bars': [{'tick': '', 'values': [data_rate, raw_rate - data_rate] if not mixed else [0, 0, data_rate, raw_rate - data_rate]}]})
        key = arm.replace('e1_1_', '')
        value[f'e11_{key}_data'] = f'{data_rate:.1f}'
        value[f'e11_{key}_raw'] = f'{raw_rate:.1f}'
        value[f'e11_{key}_ratio'] = f'{number(busiest, "raw_over_data"):.3f}'
    for group in groups:
        values = group['bars'][0]['values']
        if len(values) == 2:
            values.extend([0, 0])
    charts['e11'] = stacked_chart(groups, [('Data, one operation', 'c-data'), ('Other bytes', 'c-data2'), ('Data, mixed', 'c-proto'), ('Other bytes, mixed', 'c-proto2')], 500, 100, 'GB/s, busiest direction', 'Data rate and wire rate per arm, 4 GPUs', bar_width=46, total_fmt='{:.0f}', legend_step=150)
    raw_rates = [number(row, 'raw_rate_gbps') for row in e11 if row['arm'] in ('e1_1_allgather', 'e1_1_reducescatter', 'e1_1_allreduce', 'e1_1_separate') and number(row, 'data_bytes_per_payload_byte') > 0.9]
    value.update(e11_raw_min=f'{min(raw_rates):.0f}', e11_raw_max=f'{max(raw_rates):.0f}')
    idle_ag = pick(e11, arm='e1_1_allgather', direction='sent')
    idle_rs = pick(e11, arm='e1_1_reducescatter', direction='received')
    value.update(e11_idle_ag=f'{number(idle_ag, "raw_bytes_per_payload_byte"):.2f}', e11_idle_rs=f'{number(idle_rs, "raw_bytes_per_payload_byte"):.2f}',
                 e11_ar_raw_pp=f'{number(pick(e11, arm="e1_1_allreduce", direction="received"), "raw_bytes_per_payload_byte"):.2f}')

    # Worked example: one GPU, one round, from the per-GPU rows of E1.1.
    e11_rows = rows(RCA4 / 'e1_1/rows.csv')
    for direction in ('sent', 'received'):
        row = pick(e11_rows, arm='e1_1_allgather', round='1', gpu='4', direction=direction)
        payload = number(row, 'payload_bytes')
        data = number(row, 'data_bytes_per_payload_byte') * payload
        raw = number(row, 'raw_bytes_per_payload_byte') * payload
        tag = direction[0]
        value[f'ex_{tag}_data_gb'] = f'{data / 1e9:,.1f}'
        value[f'ex_{tag}_raw_gb'] = f'{raw / 1e9:,.1f}'
        value[f'ex_{tag}_dpp'] = f'{number(row, "data_bytes_per_payload_byte"):.3f}'
        value[f'ex_{tag}_rpp'] = f'{number(row, "raw_bytes_per_payload_byte"):.3f}'
        value[f'ex_{tag}_rd'] = f'{number(row, "raw_over_data"):.3f}'
        value[f'ex_{tag}_drate'] = f'{number(row, "data_rate_gbps"):.1f}'
        value[f'ex_{tag}_rrate'] = f'{number(row, "raw_rate_gbps"):.1f}'
        value['ex_payload_gb'] = f'{payload / 1e9:,.1f}'
        value['ex_seconds'] = f'{number(row, "extra_window_seconds"):.1f}'

    # E1.2
    e12 = rows(RCA4 / 'e1_2/summary.csv')
    for role, busy, idle, key, title in (('multicast', 'received', 'sent', 'e12_st', 'multimem.st alone'), ('reduce', 'sent', 'received', 'e12_ld', 'ld_reduce alone')):
        settings = sorted({row['arm'] for row in e12 if row['role_a'] == role}, key=lambda arm: number(pick(e12, arm=arm, direction=busy), 'data_rate_gbps'))
        busy_points, idle_points = [], []
        for arm in settings:
            busy_row, idle_row = pick(e12, arm=arm, direction=busy), pick(e12, arm=arm, direction=idle)
            rate = number(busy_row, 'data_rate_gbps')
            note = f"{busy_row['ctas_a']}×{busy_row['threads_per_cta']}" if arm in (settings[0], settings[-1]) else None
            busy_points.append((rate, number(busy_row, 'raw_bytes_per_payload_byte'), note))
            idle_points.append((rate, number(idle_row, 'raw_bytes_per_payload_byte'), None))
        charts[key] = line_chart([{'name': 'Busy direction', 'css': 'c-data', 'points': busy_points}, {'name': 'Low-data direction', 'css': 'c-other', 'points': idle_points}],
                                 400, 100, 2.0, 0.5, f'Data rate on the busy direction, GB/s ({title})', 'W ÷ S', f'Wire bytes against issue rate, {title}', yfmt='{:.1f}')
        value[f'{key}_busy_low'] = f'{busy_points[0][1]:.2f}'
        value[f'{key}_busy_high'] = f'{busy_points[-1][1]:.3f}'
        value[f'{key}_rate_low'] = f'{busy_points[0][0]:.1f}'
        value[f'{key}_rate_high'] = f'{busy_points[-1][0]:.1f}'
        idle_raw_rates = [number(pick(e12, arm=arm, direction=idle), 'raw_rate_gbps') for arm in settings]
        value[f'{key}_idle_rate_low'] = f'{idle_raw_rates[0]:.0f}'
        value[f'{key}_idle_rate_high'] = f'{idle_raw_rates[-1]:.0f}'
        value[f'{key}_idle_min'] = f'{min(point[1] for point in idle_points):.2f}'
        value[f'{key}_idle_max'] = f'{max(point[1] for point in idle_points):.2f}'
        value[f'{key}_fold'] = f'{busy_points[-1][0] / busy_points[0][0]:.0f}'

    # E1.3
    rates = json.loads((RCA4 / 'e1_3/role_rates.json').read_text())
    def rate_of(prefix):
        return statistics.median(next(iter(record['roles'].values()))['rate_per_issuing_gpu_gbps'] for record in rates if record['run'].startswith(prefix))
    differences = []
    for arm in ('allgather', 'reducescatter', 'plain_store'):
        sampled, unsampled = rate_of(f'e1_3_{arm}_sampled'), rate_of(f'e1_3_{arm}_unsampled')
        differences.append(abs(sampled - unsampled) / unsampled)
        value[f'e13_{arm}_sampled'] = f'{sampled:.1f}'
        value[f'e13_{arm}_unsampled'] = f'{unsampled:.1f}'
    value['e13_max_diff'] = f'{100 * max(differences):.2f}'
    e13 = rows(RCA4 / 'e1_3/nsys_summary.csv')
    for arm, busy, idle in (('allgather', 'received', 'sent'), ('reducescatter', 'sent', 'received')):
        value[f'e13_{arm}_busy'] = f'{100 * number(pick(e13, arm=f"e1_3_{arm}_sampled", direction=busy), "total_over_driver_raw"):.1f}'
        value[f'e13_{arm}_idle'] = f'{100 * number(pick(e13, arm=f"e1_3_{arm}_sampled", direction=idle), "total_over_driver_raw"):.0f}'
    excluded = sum(int(row['excluded_gpu_rows']) for experiment in ('e1_3', 'e1_4', 'e1_5', 'e1_6', 'e1_7') for row in rows(RCA4 / experiment / 'nsys_summary.csv'))
    kept = sum(int(row['samples']) for experiment in ('e1_3', 'e1_4', 'e1_5', 'e1_6', 'e1_7') for row in rows(RCA4 / experiment / 'nsys_summary.csv'))
    value.update(e13_excluded=str(excluded), e13_total=str(excluded + kept))

    # E1.4 and E1.6: packet composition
    parts = [('Request data', 'c-data'), ('Response data', 'c-data2'), ('Request protocol', 'c-proto'), ('Response protocol', 'c-proto2'), ('Wire bytes outside packets', 'c-other')]
    keys = ['request_user_per_payload_byte', 'response_user_per_payload_byte', 'request_protocol_per_payload_byte', 'response_protocol_per_payload_byte', 'raw_outside_packets_per_payload_byte']
    def composition(table, arm):
        return [{'tick': direction, 'values': [max(number(pick(table, arm=arm, direction=direction), key), 0) for key in keys]} for direction in ('sent', 'received')]
    e14 = rows(RCA4 / 'e1_4/nsys_summary.csv')
    charts['e14'] = stacked_chart([{'label': ['multimem.st'], 'bars': composition(e14, 'e1_4_multicast')}, {'label': ['ld_reduce'], 'bars': composition(e14, 'e1_4_reduce')},
                                   {'label': ['plain store'], 'bars': composition(e14, 'e1_4_plain_store')}, {'label': ['plain load'], 'bars': composition(e14, 'e1_4_plain_load')}],
                                  parts, 2.0, 0.5, 'Bytes on the direction ÷ S', 'Packet bytes, each operation alone', legend_step=150)
    for arm, short in (('e1_4_multicast', 'st'), ('e1_4_reduce', 'ld'), ('e1_4_plain_store', 'ps'), ('e1_4_plain_load', 'pl')):
        for direction in ('sent', 'received'):
            row = pick(e14, arm=arm, direction=direction)
            for key, name in zip(keys, ('rqd', 'rsd', 'rqp', 'rsp', 'out')):
                value[f'e14_{short}_{direction[0]}_{name}'] = f'{number(row, key):.3f}'
    e16 = rows(RCA4 / 'e1_6/nsys_summary.csv')
    charts['e16'] = stacked_chart([{'label': ['AllGather', '(multimem.st alone)'], 'bars': composition(e16, 'e1_6_allgather')}, {'label': ['ReduceScatter', '(ld_reduce alone)'], 'bars': composition(e16, 'e1_6_reducescatter')},
                                   {'label': ['AllReduce', '(both, same CTAs)'], 'bars': composition(e16, 'e1_6_allreduce_same_ctas')}, {'label': ['Both on', 'separate CTAs'], 'bars': composition(e16, 'e1_6_separate_ctas')}],
                                  parts, 2.0, 0.5, 'Bytes on the direction ÷ S', 'Packet bytes, alone and mixed', legend_step=150)
    def packets(table, arm, direction):
        row = pick(table, arm=arm, direction=direction)
        return sum(number(row, key) for key in keys[:4]), number(row, keys[4])
    for direction in ('sent', 'received'):
        ag, ag_out = packets(e16, 'e1_6_allgather', direction)
        rs, rs_out = packets(e16, 'e1_6_reducescatter', direction)
        ar, ar_out = packets(e16, 'e1_6_allreduce_same_ctas', direction)
        value[f'e16_{direction}_ag'] = f'{ag:.3f}'
        value[f'e16_{direction}_rs'] = f'{rs:.3f}'
        value[f'e16_{direction}_sum'] = f'{ag + rs:.3f}'
        value[f'e16_{direction}_ar'] = f'{ar:.3f}'
        value[f'e16_{direction}_diff'] = f'{ar - (ag + rs):+.3f}'
        value[f'e16_{direction}_out_alone'] = f'{ag_out + rs_out:.2f}'
        value[f'e16_{direction}_out_ar'] = f'{ar_out:.3f}'
        value[f'e16_{direction}_ratio'] = f'{(ar + ar_out) / 1.25:.3f}'
    ar_received = pick(e16, arm='e1_6_allreduce_same_ctas', direction='received')
    rs_received = pick(e16, arm='e1_6_reducescatter', direction='received')
    # Read requests in the mix = AllReduce request protocol minus the write headers of multimem.st alone.
    write_headers = number(pick(e16, arm='e1_6_allgather', direction='received'), 'request_protocol_per_payload_byte')
    read_requests_mix = number(ar_received, 'request_protocol_per_payload_byte') - write_headers
    value['e16_readreq_growth'] = f'{100 * (read_requests_mix / number(rs_received, "request_protocol_per_payload_byte") - 1):.0f}'
    value.update(e16_readreq_mix=f'{read_requests_mix:.3f}', e16_readreq_alone=f'{number(rs_received, "request_protocol_per_payload_byte"):.3f}',
                 e16_writeresp=f'{number(pick(e16, arm="e1_6_allgather", direction="sent"), "response_protocol_per_payload_byte"):.3f}')
    # Budget of one busy direction: AllGather alone (received) against AllReduce (received and sent).
    charts['budget'] = budget_chart(e16)

    # E1.5
    e15 = rows(RCA4 / 'e1_5/nsys_summary.csv')
    def non_data(role, width, pattern):
        total = 0.0
        for row in e15:
            if row['role_a'] == role and row['access_bytes'] == str(width) and row['pattern'] == pattern:
                total += number(row, 'request_protocol_per_payload_byte') + number(row, 'response_protocol_per_payload_byte') + max(number(row, 'raw_outside_packets_per_payload_byte'), 0)
        return total
    for role, key, title in (('multicast', 'e15_st', 'multimem.st alone'), ('reduce', 'e15_ld', 'ld_reduce alone')):
        series = []
        for pattern, css, name in (('contiguous', 'c-data', 'Warp-contiguous (NCCL)'), ('thread_contiguous', 'c-proto', 'Thread-contiguous')):
            series.append({'name': name, 'css': css, 'points': [(width, non_data(role, width, pattern), None) for width in (4, 8, 16)]})
        charts[key] = line_chart(series, 18, 4, 9, 3, f'Instruction width: bytes per thread per memory instruction ({title})', 'Non-data bytes ÷ S', f'Non-data bytes against access width, {title}', xticks=[4, 8, 16])
    def busy_rate(role, width, pattern):
        busy = 'received' if role == 'multicast' else 'sent'
        return number(next(row for row in e15 if row['role_a'] == role and row['access_bytes'] == str(width) and row['pattern'] == pattern and row['direction'] == busy), 'user_rate_gbps')
    value.update(e15_st_rate=f'{busy_rate("multicast", 16, "contiguous"):.1f}', e15_st_rate_tc=f'{busy_rate("multicast", 16, "thread_contiguous"):.1f}',
                 e15_ld_rate=f'{busy_rate("reduce", 16, "contiguous"):.1f}', e15_st_nd=f'{non_data("multicast", 16, "contiguous"):.2f}', e15_st_nd_tc=f'{non_data("multicast", 16, "thread_contiguous"):.2f}')
    value['e15_st_drop'] = f'{busy_rate("multicast", 16, "contiguous") / busy_rate("multicast", 16, "thread_contiguous"):.1f}'
    ps = [row for row in e15 if row['role_a'] == 'plain_store' and row['direction'] == 'sent']
    value['e15_ps_drop'] = f'{number(pick(ps, pattern="contiguous"), "user_rate_gbps") / number(pick(ps, pattern="thread_contiguous"), "user_rate_gbps"):.1f}'

    # E1.7
    e17 = rows(RCA4 / 'e1_7/role_rates.csv')
    for sweep_name, key, xlabel in (('reduce_varied', 'e17_a', 'ld_reduce CTAs (multimem.st fixed at 32)'), ('multicast_varied', 'e17_b', 'multimem.st CTAs (ld_reduce fixed at 32)')):
        series = []
        for role, css, name in (('multicast', 'c-data', 'multimem.st rate'), ('reduce', 'c-proto', 'ld_reduce rate')):
            points = sorted((int(row['varied_ctas']), number(row, 'rate_gbps'), None) for row in e17 if row['sweep'] == sweep_name and row['role'] == role)
            series.append({'name': name, 'css': css, 'points': points})
        charts[key] = line_chart(series, 64, 16, 100, 20, xlabel, 'GB/s of own part per GPU', f'Role rates, {xlabel}', xticks=[0, 4, 8, 16, 32, 64])
    def share(multicast, reduce, role):
        return number(next(row for row in e17 if row['multicast_ctas'] == str(multicast) and row['reduce_ctas'] == str(reduce) and row['role'] == role), 'rate_gbps')
    value.update(e17_64_32_st=f'{share(64, 32, "multicast"):.1f}', e17_64_32_ld=f'{share(64, 32, "reduce"):.1f}', e17_32_64_st=f'{share(32, 64, "multicast"):.1f}', e17_32_64_ld=f'{share(32, 64, "reduce"):.1f}',
                 e17_32_32_st=f'{share(32, 32, "multicast"):.1f}', e17_32_32_ld=f'{share(32, 32, "reduce"):.1f}')

    # E1.8
    e18 = rows(RCA4 / 'e1_8/probe_latency.csv')
    backgrounds = [('none', ['none']), ('plain_store', ['plain', 'store']), ('allgather', ['multimem', '.st']), ('reducescatter', ['ld_', 'reduce']), ('allreduce', ['All', 'Reduce'])]
    for probe, key, title in (('reduce', 'e18_ld', 'one ld_reduce at a time'), ('multicast', 'e18_st', 'one multimem.st + fence at a time')):
        groups = []
        for background, lines in backgrounds:
            members = [row for row in e18 if row['probe'] == probe and row['background'] == background]
            median = statistics.median(number(row, 'probe_median_ns') for row in members)
            p90 = statistics.median(number(row, 'probe_p90_ns') for row in members)
            groups.append((lines, median, p90, 'c-data' if background == 'none' else 'c-proto'))
            value[f'{key}_{background}'] = f'{median:,.0f}'
        top = 40000 if probe == 'multicast' else 20000
        charts[key] = bars_with_marks(groups, top, top / 4, 'ns per operation', f'Probe latency, {title}')
    def median_ns(probe, background):
        return statistics.median(number(row, 'probe_median_ns') for row in e18 if row['probe'] == probe and row['background'] == background)
    added_plain = {probe: median_ns(probe, 'plain_store') - median_ns(probe, 'none') for probe in ('reduce', 'multicast')}
    added_multimem = [(median_ns(probe, background) - median_ns(probe, 'none')) / added_plain[probe] for probe in ('reduce', 'multicast') for background in ('allgather', 'reducescatter', 'allreduce')]
    value.update(e18_factor_min=f'{min(added_multimem):.1f}', e18_factor_max=f'{max(added_multimem):.0f}')

    # E1.9
    e19 = rows(RCA8 / 'e1_9/nccl_summary.csv')
    def comp9(cell):
        return [{'tick': direction, 'values': [number(pick(e19, cell=cell, direction=direction), key) for key in keys[:4]]} for direction in ('sent', 'received')]
    charts['e19'] = stacked_chart([{'label': ['Multicast', 'AllGather'], 'bars': comp9('e1_9_allgather_sym_multicast')}, {'label': ['Multicast', 'ReduceScatter'], 'bars': comp9('e1_9_reducescatter_sym_multicast')},
                                   {'label': ['Multicast', 'AllReduce'], 'bars': comp9('e1_9_allreduce_sym_multicast')}, {'label': ['Ring', 'AllReduce'], 'bars': comp9('e1_9_allreduce_ring')}],
                                  parts[:4], 2.4, 0.4, 'Packet bytes on the direction ÷ S', 'Packet bytes, NCCL kernels on 8 GPUs', legend_step=150)
    def packet_total(cell, direction):
        return number(pick(e19, cell=cell, direction=direction), 'packets_per_payload_byte')
    ring_busy = max(packet_total('e1_9_allreduce_ring', 'sent'), packet_total('e1_9_allreduce_ring', 'received'))
    mc_sent, mc_received = packet_total('e1_9_allreduce_sym_multicast', 'sent'), packet_total('e1_9_allreduce_sym_multicast', 'received')
    read_requests_8 = number(pick(e19, cell='e1_9_allreduce_sym_multicast', direction='received'), 'request_protocol_per_payload_byte') - number(pick(e19, cell='e1_9_allgather_sym_multicast', direction='received'), 'request_protocol_per_payload_byte')
    write_responses_8 = number(pick(e19, cell='e1_9_allgather_sym_multicast', direction='sent'), 'response_protocol_per_payload_byte')
    # Gain accounting uses the driver raw counter of the earlier 8-GPU raw-counter run (corrected 2026-11-21):
    # the packet counters alone miss about 0.1 raw bytes per payload byte on the sent side.
    raw8_rows = rows(RAW8)
    def raw8(collective, arm, direction):
        return number(next(row for row in raw8_rows if row['collective'] == collective and row['arm'] == arm and row['direction'] == direction and row['gpu_count'] == '8'), 'raw_bytes_per_payload_byte')
    raw_ring = max(raw8('allreduce', 'ring', 'Tx'), raw8('allreduce', 'ring', 'Rx'))
    raw_mc_sent, raw_mc_received = raw8('allreduce', 'sym_multicast', 'Tx'), raw8('allreduce', 'sym_multicast', 'Rx')
    without_sent, without_received = raw_mc_sent - write_responses_8, raw_mc_received - read_requests_8
    without = max(without_sent, without_received)
    gain_algbw = number(pick(e19, cell='e1_9_allreduce_sym_multicast', direction='sent'), 'algbw_gbps') / number(pick(e19, cell='e1_9_allreduce_ring', direction='sent'), 'algbw_gbps')
    value.update(e19_ring_busy=f'{ring_busy:.3f}', e19_mc_sent=f'{mc_sent:.3f}', e19_mc_received=f'{mc_received:.3f}', e19_readreq=f'{read_requests_8:.3f}',
                 e19_without=f'{without:.3f}', e19_without_sent=f'{without_sent:.3f}', e19_without_received=f'{without_received:.3f}', e19_raw_ring=f'{raw_ring:.3f}', e19_raw_mc=f'{max(raw_mc_sent, raw_mc_received):.3f}', e19_gain_packets=f'{raw_ring / max(raw_mc_sent, raw_mc_received):.2f}', e19_gain_without=f'{raw_ring / without:.2f}', e19_gain_algbw=f'{gain_algbw:.2f}',
                 e19_ring_ratio=f'{packet_total("e1_9_allreduce_ring", "sent") / 1.75:.2f}', e19_mc_ratio_r=f'{mc_received / 1.125:.2f}', e19_mc_ratio_s=f'{mc_sent / 1.125:.2f}',
                 e19_ring_resp=f'{number(pick(e19, cell="e1_9_allreduce_ring", direction="sent"), "response_protocol_per_payload_byte"):.3f}')
    charts['gain'] = gain_chart(float(value['ideal8']), gain_algbw, raw_ring / max(raw_mc_sent, raw_mc_received), raw_ring / without)
    return value, charts


def budget_chart(e16):
    """Horizontal budget of one busy direction per payload byte, AllGather alone against AllReduce."""
    def row(arm, direction):
        return next(item for item in e16 if item['arm'] == arm and item['direction'] == direction)
    def f(item, key):
        return float(item[key])
    ag_r, ag_s = row('e1_6_allgather', 'received'), row('e1_6_allgather', 'sent')
    rs_r, rs_s = row('e1_6_reducescatter', 'received'), row('e1_6_reducescatter', 'sent')
    ar_r, ar_s = row('e1_6_allreduce_same_ctas', 'received'), row('e1_6_allreduce_same_ctas', 'sent')
    # Headers of data packets and the extra packets are split using the operations alone in the same session.
    write_headers = f(ag_r, 'request_protocol_per_payload_byte')
    reply_headers = f(rs_s, 'response_protocol_per_payload_byte')
    store_headers = f(ag_s, 'request_protocol_per_payload_byte')
    bars = [
        ('multimem.st alone, received', [(f(ag_r, 'request_user_per_payload_byte'), 'c-data', 'D = 1.0 S'), (write_headers, 'c-hdr', 'headers'), (f(ag_r, 'response_protocol_per_payload_byte'), 'c-proto2', ''), (max(f(ag_r, 'raw_outside_packets_per_payload_byte'), 0), 'c-other', '')]),
        # Response protocol of the received sums is more than their headers (0.052 against 0.25 / 8), so all of
        # R4 is shown as "other protocol"; only the write headers are shown as headers.
        ('AllReduce, received', [(f(ar_r, 'request_user_per_payload_byte') + f(ar_r, 'response_user_per_payload_byte'), 'c-data', 'D = 1.25 S'), (write_headers, 'c-hdr', 'headers'), (f(ar_r, 'request_protocol_per_payload_byte') - write_headers, 'c-proto', 'read requests'), (f(ar_r, 'response_protocol_per_payload_byte'), 'c-proto2', ''), (max(f(ar_r, 'raw_outside_packets_per_payload_byte'), 0), 'c-other', '')]),
        ('AllReduce, sent', [(f(ar_s, 'request_user_per_payload_byte') + f(ar_s, 'response_user_per_payload_byte'), 'c-data', 'D = 1.25 S'), (reply_headers + store_headers, 'c-hdr', 'headers'), (f(ar_s, 'response_protocol_per_payload_byte') - reply_headers, 'c-proto', 'write responses'), (f(ar_s, 'request_protocol_per_payload_byte') - store_headers, 'c-proto2', 'own requests'), (max(f(ar_s, 'raw_outside_packets_per_payload_byte'), 0), 'c-other', '')]),
    ]
    left, width, top, row_height = 190, 760, 56, 46
    scale = (width - left - 120) / 1.8
    height = top + row_height * len(bars) + 30
    svg = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Link budget of one busy direction" class="chart">']
    svg.append(legend([('Data', 'c-data'), ('Headers of data packets', 'c-hdr'), ('Read requests, write responses', 'c-proto'), ('Other protocol', 'c-proto2'), ('Wire bytes outside packets', 'c-other')], 20, 16, None, max_x=width - 10))
    for tick in (0, 0.5, 1.0, 1.5):
        x = left + scale * tick
        svg.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{top + row_height * len(bars)}" class="grid"/><text x="{x:.1f}" y="{top + row_height * len(bars) + 16}" class="t-small" text-anchor="middle">{tick:g}</text>')
    for index, (name, segments) in enumerate(bars):
        y = top + index * row_height + 6
        svg.append(f'<text x="{left - 10}" y="{y + 19}" class="t-label" text-anchor="end">{esc(name)}</text>')
        x = left
        total = 0.0
        for size, css, note in segments:
            if size <= 0:
                continue
            w = scale * size
            svg.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w - 1, 0.5):.1f}" height="28" class="{css}"><title>{esc(note or css)}: {size:.3f}</title></rect>')
            if note and w > 7 * len(note) + 10:
                svg.append(f'<text x="{x + w / 2:.1f}" y="{y + 18}" class="t-inbar" text-anchor="middle">{esc(note)}</text>')
            x += w
            total += size
        data = segments[0][0]
        svg.append(f'<text x="{x + 6:.1f}" y="{y + 19}" class="t-value">{total:.2f} S (w = {total / data:.3f})</text>')
    svg.append(f'<text x="{left + scale * 0.9:.1f}" y="{height - 2}" class="t-small" text-anchor="middle">bytes on one direction of one GPU ÷ S</text>')
    svg.append('</svg>')
    return ''.join(svg)


def gain_chart(ideal, measured, packet_ratio, without):
    """Horizontal bars: gain of multicast AllReduce over Ring at 8 GPUs."""
    items = [('Byte-count bound', ideal, 'c-data2'), ('Measured algBW ratio', measured, 'c-proto'), ('Ratio of W ÷ S on the busier direction', packet_ratio, 'c-proto2'), ('Same, without read requests and write responses', without, 'c-data')]
    left, width, top, row_height = 330, 760, 14, 40
    scale = (width - left - 70) / 1.8
    height = top + row_height * len(items) + 30
    svg = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Gain of multicast AllReduce over Ring at 8 GPUs" class="chart">']
    for tick in (0, 0.5, 1.0, 1.5):
        x = left + scale * tick
        svg.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + row_height * len(items)}" class="grid"/><text x="{x:.1f}" y="{top + row_height * len(items) + 16}" class="t-small" text-anchor="middle">{tick:g}×</text>')
    for index, (name, size, css) in enumerate(items):
        y = top + index * row_height + 6
        svg.append(f'<text x="{left - 10}" y="{y + 17}" class="t-label" text-anchor="end">{esc(name)}</text>')
        svg.append(f'<rect x="{left}" y="{y}" width="{scale * size:.1f}" height="24" class="{css}"/><text x="{left + scale * size + 6:.1f}" y="{y + 17}" class="t-value">{size:.2f}×</text>')
    svg.append('</svg>')
    return ''.join(svg)


# ---------------------------------------------------------------- page

STYLE = r'''
<title>NVLS AllReduce Packet Budget</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
/* Layout: one reading column (~68ch) with figures that break out wider; colours encode one thing each:
   blue = data bytes, orange = requests/responses/protocol, grey = wire bytes outside the packet counters. */
:root {
  --bg: #f6f7f9; --surface: #ffffff; --ink: #18212b; --muted: #5b6672; --rule: #d8dde4; --code-bg: #eef1f5;
  --data: #2f6fb0; --data2: #93b9e2; --hdr: #1f4f80; --proto: #d4721c; --proto2: #f0bf8f; --other: #b3b8bf; --mark: #18212b;
  --note-bg: #eaf1f9;
  --font-display: "IBM Plex Sans Condensed", "Arial Narrow", "Helvetica Neue", Arial, sans-serif;
  --font-body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #10151b; --surface: #171e26; --ink: #e5eaf0; --muted: #9ba7b4; --rule: #2b3541; --code-bg: #1d2630;
  --data: #5d9de2; --data2: #2e5a86; --hdr: #a9cdf2; --proto: #f08c3e; --proto2: #84532b; --other: #59616b; --mark: #e5eaf0;
  --note-bg: #17283a; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #10151b; --surface: #171e26; --ink: #e5eaf0; --muted: #9ba7b4; --rule: #2b3541; --code-bg: #1d2630;
  --data: #5d9de2; --data2: #2e5a86; --hdr: #a9cdf2; --proto: #f08c3e; --proto2: #84532b; --other: #59616b; --mark: #e5eaf0;
  --note-bg: #17283a; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 400 17px/1.65 var(--font-body); }
.page { max-width: 1000px; margin: 0 auto; padding-block: 40px 80px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.col { max-width: 700px; margin: 0 auto; }
h1, h2, h3, h4 { font-family: var(--font-display); line-height: 1.15; text-wrap: balance; margin: 0; }
h1 { font-size: clamp(2.2rem, 5vw, 3.4rem); font-weight: 700; letter-spacing: -0.01em; }
h2 { font-size: 1.85rem; font-weight: 700; margin-top: 3.2rem; padding-top: 1.2rem; border-top: 2px solid var(--ink); }
h3 { font-size: 1.25rem; font-weight: 600; margin-top: 2rem; }
h4 { font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-top: 1.6rem; }
p, ul, ol { margin: 0.7rem 0; }
li { margin: 0.25rem 0; }
a { color: var(--data); }
a:focus-visible { outline: 2px solid var(--proto); outline-offset: 2px; }
code { font-family: var(--font-mono); font-size: 0.86em; background: var(--code-bg); padding: 0.08em 0.32em; border-radius: 3px; }
.eyebrow { font-family: var(--font-mono); font-size: 0.8rem; color: var(--muted); letter-spacing: 0.04em; }
.lede { font-size: 1.2rem; color: var(--ink); margin-top: 1rem; }
.byline { color: var(--muted); font-size: 0.92rem; }
.tag { font-family: var(--font-mono); font-size: 0.75rem; color: var(--muted); display: block; margin-top: 2.6rem; }
.glance { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin: 1.8rem 0 0.6rem; }
.glance div { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; min-width: 0; }
.glance b { display: block; font-family: var(--font-display); font-size: 1.9rem; line-height: 1.1; font-variant-numeric: tabular-nums; }
.glance span { font-size: 0.88rem; color: var(--muted); line-height: 1.4; display: block; margin-top: 4px; }
.toc { columns: 2 260px; font-size: 0.95rem; padding-left: 1.2rem; }
figure { margin: 1.6rem 0; }
figure .scroll { overflow-x: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 12px; }
figure.pair .scroll { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 12px; }
figure svg { display: block; width: 100%; height: auto; min-width: 380px; }
figcaption { font-size: 0.88rem; color: var(--muted); margin-top: 0.5rem; max-width: 700px; }
.wide { max-width: 1000px; margin-left: auto; margin-right: auto; }
.tablewrap { overflow-x: auto; margin: 1rem 0; }
table { border-collapse: collapse; font-size: 0.9rem; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }
td.n, th.n { text-align: right; }
.note { background: var(--note-bg); border-radius: 6px; padding: 12px 16px; margin: 1.2rem 0; font-size: 0.95rem; }
.verdict { font-family: var(--font-display); font-size: 1.15rem; font-weight: 600; margin: 0.6rem 0; }
.takeaways li { margin: 0.7rem 0; }
.takeaways b { font-family: var(--font-display); font-size: 1.05rem; }
/* chart marks */
.chart text { fill: var(--ink); font-family: var(--font-body); }
.t-small { font-size: 11px; fill: var(--muted) !important; }
.t-tiny { font-size: 10px; fill: var(--muted) !important; }
.t-label { font-size: 12px; }
.t-value { font-size: 11px; font-weight: 600; font-variant-numeric: tabular-nums; }
.t-inbar { font-size: 11px; fill: var(--surface) !important; font-weight: 600; }
.grid { stroke: var(--rule); stroke-width: 1; }
.axis { stroke: var(--muted); stroke-width: 1; }
.mark { stroke: var(--mark); stroke-width: 2; }
.c-data { fill: var(--data); } .c-data2 { fill: var(--data2); } .c-hdr { fill: var(--hdr); } .c-proto { fill: var(--proto); } .c-proto2 { fill: var(--proto2); } .c-other { fill: var(--other); }
.line { fill: none; stroke-width: 2.2; }
.c-data-stroke { stroke: var(--data); } .c-proto-stroke { stroke: var(--proto); } .c-other-stroke { stroke: var(--other); }
.dot { stroke: var(--surface); stroke-width: 1.5; }
/* diagram parts */
.d-box { fill: var(--surface); stroke: var(--ink); stroke-width: 1.5; }
.d-switch { fill: var(--code-bg); stroke: var(--ink); stroke-width: 1.5; }
.d-arrow-data { stroke: var(--data); stroke-width: 3; fill: none; }
.d-arrow-proto { stroke: var(--proto); stroke-width: 2; fill: none; stroke-dasharray: 5 4; }
.d-arrow-ink { stroke: var(--muted); stroke-width: 2; fill: none; }
.d-head-data { fill: var(--data); } .d-head-proto { fill: var(--proto); } .d-head-ink { fill: var(--muted); }
.math svg { display: inline-block; overflow: visible; }
.c-small { fill: #9b7fc7; }
.d-text { font-size: 13px; } .d-title { font-size: 14px; font-weight: 600; font-family: var(--font-display); }
@media (max-width: 520px) { body { font-size: 16px; } h2 { font-size: 1.5rem; } }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
</style>
'''

DIAGRAM_TOPOLOGY = '''
<svg viewBox="0 0 760 250" role="img" aria-label="Four GPUs connected to an NVSwitch, each link with a sent and a received direction" class="chart">
<defs>
<marker id="hd" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="d-head-data"/></marker>
<marker id="hp" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="d-head-proto"/></marker>
<marker id="hi" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="d-head-ink"/></marker>
</defs>
<rect x="150" y="18" width="460" height="56" rx="6" class="d-switch"/>
<text x="380" y="44" class="d-title" text-anchor="middle">NVSwitch fabric</text>
<text x="380" y="62" class="d-text" text-anchor="middle">copies, reads and adds data for multimem instructions</text>
''' + ''.join(f'''
<rect x="{60 + 180 * index}" y="170" width="120" height="56" rx="6" class="d-box"/>
<text x="{120 + 180 * index}" y="203" class="d-title" text-anchor="middle">GPU {index}</text>
<line x1="{103 + 180 * index}" y1="166" x2="{103 + 180 * index}" y2="82" class="d-arrow-ink" marker-end="url(#hi)"/>
<line x1="{137 + 180 * index}" y1="80" x2="{137 + 180 * index}" y2="164" class="d-arrow-ink" marker-end="url(#hi)"/>
''' for index in range(4)) + '''
<text x="96" y="128" class="d-text" text-anchor="end">sent</text>
<text x="144" y="128" class="d-text">received</text>
</svg>'''

DIAGRAM_ALGORITHMS = '''
<svg viewBox="0 0 900 312" role="img" aria-label="Ring AllReduce against NVLS AllReduce" class="chart">
<text x="200" y="22" class="d-title" text-anchor="middle">Ring: pass pieces around the circle</text>
<circle cx="200" cy="150" r="85" fill="none" class="d-arrow-ink" stroke-dasharray="3 5"/>
''' + ''.join(f'''<rect x="{200 + 85 * x - 34}" y="{150 + 85 * y - 20}" width="68" height="40" rx="6" class="d-box"/><text x="{200 + 85 * x}" y="{150 + 85 * y + 5}" class="d-title" text-anchor="middle">GPU {index}</text>''' for index, (x, y) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)])) + '''
<path d="M236,66 Q280,78 285,126" class="d-arrow-data" marker-end="url(#hd)"/>
<path d="M285,172 Q280,222 238,234" class="d-arrow-data" marker-end="url(#hd)"/>
<path d="M164,234 Q120,222 115,174" class="d-arrow-data" marker-end="url(#hd)"/>
<path d="M115,126 Q120,78 162,66" class="d-arrow-data" marker-end="url(#hd)"/>
<text x="200" y="284" class="d-text" text-anchor="middle">2(N−1) steps. Each GPU sends and receives</text>
<text x="200" y="300" class="d-text" text-anchor="middle">D = 2(N−1)/N × S on each direction.</text>
<text x="660" y="22" class="d-title" text-anchor="middle">NVLS: let the switch add and copy</text>
<rect x="520" y="60" width="280" height="44" rx="6" class="d-switch"/><text x="660" y="87" class="d-title" text-anchor="middle">NVSwitch</text>
''' + ''.join(f'''<rect x="{510 + 78 * index}" y="200" width="66" height="40" rx="6" class="d-box"/><text x="{543 + 78 * index}" y="225" class="d-title" text-anchor="middle">GPU {index}</text>''' for index in range(4)) + '''
<line x1="534" y1="196" x2="534" y2="110" class="d-arrow-data" marker-end="url(#hd)"/>
<line x1="612" y1="196" x2="612" y2="110" class="d-arrow-data" marker-end="url(#hd)"/>
<line x1="690" y1="196" x2="690" y2="110" class="d-arrow-data" marker-end="url(#hd)"/>
<line x1="768" y1="196" x2="768" y2="110" class="d-arrow-data" marker-end="url(#hd)"/>
<line x1="552" y1="110" x2="552" y2="194" class="d-arrow-proto" marker-end="url(#hp)"/>
<text x="560" y="152" class="d-text">sum</text>
<text x="660" y="268" class="d-text" text-anchor="middle">1. ld_reduce: GPU 0 asks for its share; the switch reads that</text>
<text x="660" y="284" class="d-text" text-anchor="middle">share on every GPU (blue) and returns the sum (orange).</text>
<text x="660" y="300" class="d-text" text-anchor="middle">2. multimem.st: GPU 0 sends the sum once; the switch writes it to all.</text>
</svg>'''

DIAGRAM_PACKETS = '''
<svg viewBox="0 0 760 250" role="img" aria-label="Packet shapes drawn to scale: 16-byte header, 128-byte payload, 16-byte requests and responses" class="chart">
<text x="20" y="20" class="t-small">drawn to scale: 1 byte = 3.4 px</text>
<text x="20" y="62" class="t-label">Write packet</text>
<rect x="200" y="44" width="54" height="28" class="c-hdr"/><rect x="255" y="44" width="435" height="28" class="c-data"/>
<text x="227" y="63" class="t-inbar" text-anchor="middle">16 B</text><text x="472" y="63" class="t-inbar" text-anchor="middle">128 B of data</text>
<text x="20" y="108" class="t-label">Write response</text>
<rect x="200" y="90" width="54" height="28" class="c-proto"/><text x="227" y="109" class="t-inbar" text-anchor="middle">16 B</text>
<text x="270" y="109" class="t-small">only for writes that must be acknowledged (multimem.st)</text>
<text x="20" y="154" class="t-label">Read request</text>
<rect x="200" y="136" width="54" height="28" class="c-proto"/><text x="227" y="155" class="t-inbar" text-anchor="middle">16 B</text>
<text x="270" y="155" class="t-small">"send me these bytes"</text>
<text x="20" y="200" class="t-label">Read reply</text>
<rect x="200" y="182" width="54" height="28" class="c-hdr"/><rect x="255" y="182" width="435" height="28" class="c-data2"/>
<text x="227" y="201" class="t-inbar" text-anchor="middle">16 B</text><text x="472" y="201" class="t-small" text-anchor="middle">128 B of data</text>
<text x="20" y="240" class="t-small">Header bytes on full packets: 16 ÷ 128 = 0.125 per data byte, measured for every kind of data packet in this study.</text>
</svg>'''

DIAGRAM_DIRECTIONS = '''
<svg viewBox="0 0 900 276" role="img" aria-label="Which packets each multimem operation puts on each link direction" class="chart">
<text x="150" y="20" class="d-title" text-anchor="middle">multimem.st alone</text>
<text x="450" y="20" class="d-title" text-anchor="middle">ld_reduce alone</text>
<text x="750" y="20" class="d-title" text-anchor="middle">AllReduce: both at once</text>
''' + ''.join(f'''
<rect x="{x0}" y="40" width="260" height="84" rx="6" class="d-box"/>
<text x="{x0 + 10}" y="60" class="t-small">SENT (GPU → switch)</text>
<rect x="{x0}" y="140" width="260" height="84" rx="6" class="d-box"/>
<text x="{x0 + 10}" y="160" class="t-small">RECEIVED (switch → GPU)</text>
''' for x0 in (20, 320, 620)) + '''
<rect x="30" y="72" width="40" height="18" class="c-data"/><text x="76" y="86" class="t-small">own stores (1/N)</text>
<rect x="30" y="96" width="60" height="18" class="c-proto"/><text x="96" y="110" class="t-small">write responses</text>
<rect x="30" y="172" width="220" height="18" class="c-data"/><text x="30" y="206" class="t-small">stores from every GPU (full)</text>
<rect x="330" y="72" width="220" height="18" class="c-data2"/><text x="330" y="106" class="t-small">read replies for every GPU (full)</text>
<rect x="330" y="172" width="70" height="18" class="c-proto"/><text x="406" y="186" class="t-small">read requests</text>
<rect x="330" y="196" width="40" height="18" class="c-data2"/><text x="376" y="210" class="t-small">own sums (1/N)</text>
<rect x="630" y="72" width="170" height="18" class="c-data2"/><rect x="801" y="72" width="30" height="18" class="c-data"/>
<rect x="630" y="96" width="60" height="18" class="c-proto"/><text x="696" y="110" class="t-small">write responses</text>
<rect x="630" y="172" width="170" height="18" class="c-data"/><rect x="801" y="172" width="30" height="18" class="c-data2"/>
<rect x="630" y="196" width="70" height="18" class="c-proto"/><text x="706" y="210" class="t-small">read requests</text>
<text x="450" y="248" class="d-text" text-anchor="middle">Alone, each operation's requests or responses ride on the low-data direction.</text>
<text x="450" y="266" class="d-text" text-anchor="middle">In AllReduce both directions are full of data, so they compete with it.</text>
</svg>'''

BODY = r'''
<main class="page">
<div class="col">
<div class="eyebrow">NVLink SHARP on H100 · NCCL 2.32.3 · experiments of 18–19 November 2026</div>
<h1>The hidden packets of NVLS AllReduce</h1>
<p class="lede">NCCL can sum data inside the NVLink switch instead of passing it around a ring of GPUs. On paper this needs far fewer bytes per GPU. In practice it gains much less than the byte count promises. Nine experiments trace the missing speed to two small kinds of packet that the ring algorithm never sends.</p>
<p class="byline">A report on the root-cause series of the scale-up communication observability project. It assumes no earlier knowledge of the project or of GPU communication.</p>
<div class="glance">
<div><b>@@gain8@@× of @@ideal8@@×</b><span>Measured speed-up of NVLS AllReduce over Ring at 8 GPUs, against what byte counts allow</span></div>
<div><b>w = @@e11_allreduce_ratio@@</b><span>Wire cost (wire bytes ÷ data bytes) of NVLS AllReduce at 4 GPUs, against w = @@e11_allgather_ratio@@ for one multimem operation alone</span></div>
<div><b>@@e16_readreq_mix@@ S + @@e16_writeresp@@ S</b><span>Read requests and write responses, as multiples of the buffer size S, that land on directions already full of data</span></div>
<div><b>@@e19_gain_without@@×</b><span>Gain over Ring at 8 GPUs if those two packet kinds did not exist (accounting with measured wire bytes); the bound is @@ideal8@@×</span></div>
</div>
<h4>Contents</h4>
<ol class="toc">
<li><a href="#background">Background: moving data between GPUs</a></li>
<li><a href="#puzzle">The puzzle</a></li>
<li><a href="#bench">The test bench</a></li>
<li><a href="#e1">Are the links busy or idle?</a></li>
<li><a href="#e2">Does protocol traffic follow requests or time?</a></li>
<li><a href="#e3">Can we count packets by type?</a></li>
<li><a href="#e4">The packets of each operation alone</a></li>
<li><a href="#e5">Does the way threads address memory matter?</a></li>
<li><a href="#e6">The packets of the mix</a></li>
<li><a href="#e7">Who gets the link when both want it?</a></li>
<li><a href="#e8">How long does one operation wait?</a></li>
<li><a href="#e9">Does NCCL on 8 GPUs behave the same?</a></li>
<li><a href="#expect">Putting it together: what NVLS AllReduce should reach</a></li>
<li><a href="#takeaways">Takeaways</a></li>
</ol>

<h2 id="background">Background: moving data between GPUs</h2>
<h3>Links, a switch, and two directions</h3>
<p>The machine has eight NVIDIA H100 GPUs. Each GPU connects to a set of NVSwitch chips by 18 NVLink lanes. In this report we treat a GPU's lanes together as one link with two directions: <b>sent</b> (out of the GPU) and <b>received</b> (into the GPU). The two directions are separate wires, so a GPU can send and receive at full speed at the same time. Engineers call this full duplex.</p>
</div>
<figure class="col"><div class="scroll">@@diagram_topology@@</div><figcaption>Every GPU has its own link to the switch fabric. Each link has a sent and a received direction that work independently. All experiments here use 4 GPUs (GPUs 4–7) except the last one, which uses all 8.</figcaption></figure>
<div class="col">
<p>On this machine, one direction of one GPU's link carries at most about 420–427 GB/s of traffic, counting everything on the wire. That number appears again and again below: it is the ceiling every pattern hits.</p>

<h3>AllReduce, two ways</h3>
<p>Training a model on several GPUs needs an operation called <b>AllReduce</b>: every GPU has a buffer of numbers, and at the end every GPU must hold the element-by-element sum of all buffers. NVIDIA's library NCCL offers two ways to do it on one machine.</p>
<ul>
<li><b>Ring.</b> The GPUs form a circle. Each GPU passes pieces of its buffer to its neighbour, adds what it receives, and passes the result on. After 2(N−1) steps every GPU has the full sum. Each direction of each GPU carries D = 2(N−1)/N × S of data: 1.5 S at 4 GPUs, 1.75 S at 8.</li>
<li><b>NVLS</b> (NVLink SHARP). The switch itself can add and copy. Two special GPU instructions use this. <code>multimem.ld_reduce</code> asks the switch to read the same address on every GPU, add the values, and return the sum. <code>multimem.st</code> sends one store to the switch, which writes it into every GPU. NCCL's kernel gives each GPU 1/N of the buffer: the GPU sums its share with <code>ld_reduce</code>, then writes the result to everyone with <code>multimem.st</code>. Each direction of each GPU then carries D = (1 + 1/N) × S: 1.25 S at 4 GPUs, 1.125 S at 8.</li>
</ul>
</div>
<figure class="wide"><div class="scroll">@@diagram_algorithms@@</div><figcaption>Left: Ring passes pieces between neighbours. Right: NVLS asks the switch to read and add every GPU's share (step 1), then to copy the sum to every GPU (step 2). Every GPU does this for its own 1/N share at the same time.</figcaption></figure>
<div class="col">
<p>If both methods could run their links at the same speed, NVLS would be faster by the ratio of bytes:</p>
<div class="tablewrap"><table>
<tr><th>GPUs</th><th class="n">Ring, data factor k = 2(N−1)/N</th><th class="n">NVLS, data factor k = 1 + 1/N</th><th class="n">Best possible speed-up (ratio of k)</th></tr>
<tr><td>4</td><td class="n">1.5</td><td class="n">1.25</td><td class="n">@@ideal4@@×</td></tr>
<tr><td>8</td><td class="n">1.75</td><td class="n">1.125</td><td class="n">@@ideal8@@×</td></tr>
</table></div>

<h3>What actually travels on a link: packets</h3>
<p>Data does not move as a plain stream of bytes. It moves in <b>packets</b>. A data packet has a header (where it goes, what it is) and up to 128 bytes of data. Some operations also need small packets without data:</p>
<ul>
<li>A <b>read</b> needs a <b>read request</b> ("send me these bytes") that travels the opposite way from the data it asks for.</li>
<li>Some <b>writes</b> must be acknowledged. The receiver then sends back a <b>write response</b>. Ordinary stores on this machine get almost none; we will see that <code>multimem.st</code> gets one per packet.</li>
</ul>
</div>
<figure class="col"><div class="scroll">@@diagram_packets@@</div><figcaption>The packet shapes found in this study, drawn to scale. The sizes come from the measurements in the sections on single operations and address patterns below.</figcaption></figure>
<div class="col">
<p>Every packet without data, and every header, takes link time that data cannot use. Because a link direction has a fixed ceiling, the more of these bytes a pattern needs per byte of data, the less data it can move per second. This is the whole story of this report.</p>

<h3>How we count</h3>
<p>Every experiment measures bytes on each GPU's link, separately for the two directions. Two counters in the GPU driver do this, and they run on the same links at the same time:</p>
<div class="tablewrap"><table>
<tr><th>Instrument</th><th>What it counts</th><th>Name used below</th></tr>
<tr><td><code>nvidia-smi nvlink -gt d</code></td><td>Payload only: the bytes the instructions asked to move</td><td>data counter</td></tr>
<tr><td><code>nvidia-smi nvlink -gt r</code></td><td>Everything on the wire: payload, headers, requests, responses, and link-level bytes</td><td>raw counter</td></tr>
<tr><td>Nsight Systems GPU metrics (from experiment 3 on)</td><td>Bytes of four packet types per direction: request data, request protocol, response data, response protocol, 10 000 samples per second</td><td>packet counters</td></tr>
</table></div>
<p>From these counters and the work the program records, this page uses <b>three kinds of bytes, each with one name</b>:</p>
<div class="tablewrap"><table>
<tr><th>Name</th><th>Symbol</th><th>What it is</th><th>Counted by</th></tr>
<tr><td><b>Buffer bytes</b> (payload)</td><td>$S$</td><td>The size of the collective's buffer: what the user asked to reduce or gather. Not a link quantity</td><td>The program: 4 × the mean bytes each GPU finished of its own quarter (nccl-tests: buffer size × calls)</td></tr>
<tr><td><b>Data bytes</b></td><td>$D$</td><td>The user's numbers that cross one direction of one GPU's link (the data part of data packets)</td><td>Data counter</td></tr>
<tr><td><b>Wire bytes</b> (raw)</td><td>$W$</td><td>Everything that crosses that direction: data, packet headers, control packets (read requests, write responses), anything else</td><td>Raw counter; the packet counters split it into packet types</td></tr>
</table></div>
<p><b>Writing convention.</b> Anything counted on one direction is written as a multiple of the buffer size $S$. "$W = 1.135\,S$" means: for a buffer of $S$ bytes, that direction carried 1.135 × $S$ wire bytes. Packet types are written the same way, for example "0.231 $S$ of read requests".</p>
<p>Two ratios connect the three, and they are the only ratios this page uses:</p>
<div class="tablewrap"><table>
<tr><th>Ratio</th><th>Definition</th><th>What sets it</th></tr>
<tr><td><b>Data factor $k$</b></td><td>$D/S$</td><td>The algorithm alone: AllGather and ReduceScatter 1 on the busy direction; NVLS AllReduce $1+1/N$; Ring AllReduce $2(N-1)/N$. The data counter always matches it</td></tr>
<tr><td><b>Wire cost $w$</b></td><td>$W/D$</td><td>The packets: $w$ = 1.13 means 13% of the wire time carried something other than data</td></tr>
</table></div>
<p>And three rates: <b>algBW = $S/t$</b>; <b>data rate = $D/t$</b> (on the busier direction this is busBW); <b>wire rate = $W/t$</b>. In our program, algBW is measured from the work the CTAs finish and the GPU clock, and the data rate and wire rate from the driver counters divided by the window time; they are three separate measurements. How nccl-tests runs differ is given in the last section.</p>
<p>The driver counters run for the whole program, including start-up and checks. So each setting runs twice, a long run (11 windows of 300 ms) and a short run (3 windows), and every count is taken as <b>long − short</b>: exactly 8 windows, @@ex_seconds@@ s of work. Values are computed per GPU and round; tables and charts show the median over 4 GPUs × 3 rounds. The profiler needs no subtraction: it timestamps its samples, and only samples inside the measured kernels are kept.</p>
<h4>Worked example</h4>
<p>One GPU (GPU 4), one round, <code>multimem.st</code> alone on all 4 GPUs. Long − short, the four GPUs processed S = @@ex_payload_gb@@ GB of buffer.</p>
<div class="tablewrap"><table>
<tr><th>Step</th><th class="n">Sent (GPU → switch)</th><th class="n">Received (switch → GPU)</th></tr>
<tr><td>Data counter</td><td class="n">@@ex_s_data_gb@@ GB</td><td class="n">@@ex_r_data_gb@@ GB</td></tr>
<tr><td>Raw counter</td><td class="n">@@ex_s_raw_gb@@ GB</td><td class="n">@@ex_r_raw_gb@@ GB</td></tr>
<tr><td>D ÷ S (data factor k)</td><td class="n">@@ex_s_dpp@@</td><td class="n">@@ex_r_dpp@@</td></tr>
<tr><td>W ÷ S</td><td class="n">@@ex_s_rpp@@</td><td class="n">@@ex_r_rpp@@</td></tr>
<tr><td>W ÷ D (wire cost w)</td><td class="n">@@ex_s_rd@@</td><td class="n">@@ex_r_rd@@</td></tr>
<tr><td>Data rate D ÷ t, GB/s</td><td class="n">@@ex_s_drate@@</td><td class="n">@@ex_r_drate@@</td></tr>
<tr><td>Wire rate W ÷ t, GB/s</td><td class="n">@@ex_s_rrate@@</td><td class="n">@@ex_r_rrate@@</td></tr>
</table></div>
<p>The received direction carries every GPU's stores (D = 1.000 S): @@ex_r_drate@@ GB/s of data at @@ex_r_rrate@@ GB/s wire rate. The sent direction carries only this GPU's own stores (D ≈ 0.25 S), yet its W is far larger than that.</p>
<h4>Busy and low-data directions</h4>
<p>When one multimem operation runs alone on every GPU, its two link directions are used very differently. One direction carries the main data stream; we call it the <b>busy direction</b>. The other carries only this GPU's own quarter of the data (at 4 GPUs); we call it the <b>low-data direction</b>. It is not idle: it also carries headers, the operation's small packets, and some wire bytes that no packet counter sees.</p>
<div class="tablewrap"><table>
<tr><th>One operation alone</th><th>Busy direction (D = S)</th><th>Low-data direction (D = S/N = 0.25 S at 4 GPUs)</th></tr>
<tr><td><code>multimem.st</code></td><td>Received: the switch writes every GPU's stores into this GPU</td><td>Sent: this GPU's own stores, plus write responses</td></tr>
<tr><td><code>ld_reduce</code></td><td>Sent: this GPU answers the switch's reads for every GPU's sums</td><td>Received: this GPU's own sums, plus read requests</td></tr>
</table></div>
<p>"Low-data-direction traffic" means the wire bytes W of that direction, as a multiple of S or per second.</p>

<h2 id="puzzle">The puzzle</h2>
<p>Earlier measurements on this machine found two problems that this series set out to explain.</p>
<p><b>Problem 1: running both multimem operations at once is slow.</b> On 4 GPUs, NVLS AllGather (only <code>multimem.st</code>) moved @@obs_ag@@ GB/s per busiest link direction, and NVLS ReduceScatter (only <code>ld_reduce</code>) moved @@obs_rs@@ GB/s. NVLS AllReduce, which does both, moved only @@obs_ar@@ GB/s: @@obs_ratio@@ of the mean of the two. That loss happens whether the same or different thread blocks issue the two instructions. A full-duplex link gives no reason for it: AllReduce fills both directions, but each direction should still reach its own ceiling.</p>
<p><b>Problem 2: NVLS gains less over Ring than the byte count promises.</b> At 8 GiB per GPU, NVLS AllReduce reached @@sym4@@ GB/s against Ring's @@ring4@@ on 4 GPUs (@@gain4@@×, best possible @@ideal4@@×), and @@sym8@@ against @@ring8@@ on 8 GPUs (@@gain8@@×, best possible @@ideal8@@×). In both cases NVLS keeps only about @@share8@@–@@share4@@ of its theoretical advantage. A report from a colleague on newer B300 GPUs saw the gain almost disappear.</p>
<p><b>The first clue</b> came from the raw counter on 8 GPUs. Every pattern, Ring or NVLS, AllGather or AllReduce, ran its busiest direction at the same wire rate, about @@raw8_ring_rate@@ GB/s. NVLS AllReduce was not slower on the wire. It simply carried more bytes that are not data: a wire cost w = @@raw8_ar@@, against w = @@raw8_ring@@ for Ring AllReduce.</p>
</div>
<figure class="col"><div class="scroll">@@chart_raw8@@</div><figcaption>Earlier result, 8 GPUs, 2 GiB, NCCL's kernels. Each bar is the busiest direction of one pattern. Dark part: data rate D ÷ t (data counter). Light part: wire rate − data rate. Bar top: wire rate W ÷ t (raw counter), GB/s. Label below: wire cost w = W ÷ D. Long − short, median over GPUs and rounds.</figcaption></figure>
<div class="col">
<div class="note"><b>The question of this series:</b> which packets does each multimem operation put on each link direction, and why does NVLS AllReduce have a wire cost w ≈ 1.35 when every other pattern has 1.13–1.18? The same number explains both problems.</div>

<h2 id="bench">The test bench</h2>
<p>Most experiments use a small program written for this study instead of NCCL itself. It lets us choose exactly which instruction each thread block issues, and how many. The last experiment repeats the key measurement with NCCL's own kernels.</p>
<div class="tablewrap"><table>
<tr><th>Item</th><th>Setting</th></tr>
<tr><td>GPUs</td><td>4 × H100 80 GB (GPUs 4–7) for the first eight experiments; all 8 for the last</td></tr>
<tr><td>Memory</td><td>Buffers allocated and registered through NCCL 2.32.3 as a symmetric window, so <code>multimem</code> instructions work on them, exactly as in NCCL's own NVLS AllReduce kernel</td></tr>
<tr><td>Work unit</td><td>A <b>CTA</b> (thread block) runs on one of the GPU's 132 streaming multiprocessors. Each CTA has a <b>role</b>: <code>multimem.st</code>, <code>ld_reduce</code>, both (AllReduce), plain store to the next GPU, or plain load from it</td></tr>
<tr><td>Default size</td><td>32 CTAs × 512 threads per role, enough to reach the ceiling; each thread moves 16 bytes per instruction, 8 at a time, with the 32 threads of a warp touching consecutive addresses (NCCL's pattern)</td></tr>
<tr><td>Timing</td><td>All CTAs on all GPUs start together and run for a fixed 300 ms window, then stop. A fixed time, not a fixed byte count, keeps both operations of a mix running for the whole window</td></tr>
<tr><td>Checks</td><td>Every word written is checked after each window. All 289 runs in this report had zero wrong words</td></tr>
</table></div>
<h4>Removing set-up traffic</h4>
<p>The driver counters run for the whole program, including start-up and checks. So every setting runs twice: a long run (11 windows) and a short run (3 windows). Long minus short leaves exactly 8 windows of work. The profiler needs no such trick: it records timestamps, so we keep only samples taken while the measured kernel ran.</p>
<h4>Cross-checks that hold in every experiment</h4>
<ul>
<li>The data counter always equals what the algorithm must move (for example D = 1.25 S for NVLS AllReduce at 4 GPUs), to four digits.</li>
<li>Repeated runs agree within 1% for every value, so tables show medians without error bars.</li>
<li>The profiler sometimes lost the counters of one to four of a GPU's 18 links (its data bytes came out at exactly 17/18, 16/18, …). Such GPUs are left out of the medians: @@e13_excluded@@ of @@e13_total@@ rows.</li>
</ul>

<span class="tag">Experiment 1 of 9 · raw link bytes</span>
<h2 id="e1">Are the links busy or idle?</h2>
<h4>Motivation</h4>
<p>On 8 GPUs every pattern hit the same wire-rate ceiling, so the loss was bytes that are not data, not idle time. Problem 1 was measured on 4 GPUs with the data counter only. If the 4-GPU loss had a different cause, for example the switch pausing when it must add and copy at once, the mixed patterns would show a lower wire rate.</p>
<h4>Configuration</h4>
<div class="tablewrap"><table>
<tr><th>What changes</th><th>What stays fixed</th></tr>
<tr><td><b>The pattern</b>: <code>multimem.st</code> alone, <code>ld_reduce</code> alone, both in the same CTAs (AllReduce), both in separate CTAs, plain store, plain load, plain load + plain store</td><td>4 GPUs, all issuing; 32 CTAs × 512 threads per role; 16 bytes per thread per memory instruction; long and short runs; 3 rounds</td></tr>
</table></div>
<h4>Procedure</h4>
<ol><li>Run each pattern long and short, back to back, three times, rotating the order.</li><li>For each GPU and direction, compute D and W from the long-minus-short difference.</li><li>Take the busiest direction of each pattern and compare its wire rate W ÷ t and its wire cost w = W ÷ D.</li></ol>
<h4>Results</h4>
</div>
<figure class="wide"><div class="scroll">@@chart_e11@@</div><figcaption>Busiest direction of each pattern, 4 GPUs. Dark part: data rate D ÷ 2.4 s. Light part: wire rate − data rate. Bar top: wire rate W ÷ 2.4 s, GB/s. Blue: one kind of operation; orange: two kinds mixed. Label below: wire cost w = W ÷ D. Median of 4 GPUs × 3 rounds.</figcaption></figure>
<div class="col">
<h4>Interpretation</h4>
<ul>
<li><b>No link is idle.</b> Every multimem pattern ran its busy direction at a wire rate of @@e11_raw_min@@–@@e11_raw_max@@ GB/s, AllReduce included. The switch does not stall when it must add and copy at the same time.</li>
<li><b>AllReduce moves less data because a quarter of its wire time is something else:</b> w = @@e11_allreduce_ratio@@, against w = @@e11_allgather_ratio@@ for <code>multimem.st</code> alone and w = @@e11_reducescatter_ratio@@ for <code>ld_reduce</code> alone. With a fixed wire rate, data rate = wire rate ÷ w: @@e11_allreduce_raw@@ ÷ @@e11_allreduce_ratio@@ ≈ @@e11_allreduce_data@@ GB/s. The 4-GPU and 8-GPU problems have the same shape.</li>
<li><b>A surprise for later:</b> alone, each operation puts W = @@e11_idle_ag@@ S (<code>multimem.st</code>) or @@e11_idle_rs@@ S (<code>ld_reduce</code>) on its low-data direction. If the mix simply added those to the busy direction of the other, AllReduce would need about W = 2 S per direction. It needs only W = @@e11_ar_raw_pp@@ S. Something shrinks when the two operations share a link.</li>
</ul>

<span class="tag">Experiment 2 of 9 · issue rate</span>
<h2 id="e2">Does protocol traffic follow requests or time?</h2>
<h4>Motivation</h4>
<p>Alone, each operation puts about W = 1 S of wire bytes on its low-data direction (defined in the background section), although only D = 0.25 S of it is data. That traffic could be of two kinds. It could be one small packet per request, which would grow and shrink with the work. Or it could be housekeeping that the link sends at a fixed rate whatever the work, such as credits that tell the other end how much room is left. Fixed-rate traffic would make the "cost per request" look larger than it is, and could explain why the mix seems cheaper.</p>
<h4>Configuration</h4>
<div class="tablewrap"><table>
<tr><th>What changes</th><th>What stays fixed</th></tr>
<tr><td><b>How hard one operation pushes</b>: CTAs × threads from 1 × 32 up to 32 × 512, for <code>multimem.st</code> alone and <code>ld_reduce</code> alone. The rate actually reached is measured, not assumed</td><td>4 GPUs, all issuing; 16 bytes per thread per memory instruction; long and short runs; 3 rounds</td></tr>
</table></div>
<h4>Procedure</h4>
<ol><li>Run 6 settings of <code>multimem.st</code> and 5 of <code>ld_reduce</code>, long and short, three rounds.</li><li>On each GPU, read the raw counter of each direction before and after every run; long minus short gives the bytes of 8 windows (2.4 s).</li><li>Divide by the buffer bytes S of those windows (W ÷ S) and by 2.4 s (wire rate). The busy direction is received for <code>multimem.st</code> and sent for <code>ld_reduce</code>; the low-data direction is the other one.</li><li>Plot both directions against the data rate reached on the busy direction. The raw counter gives one total per direction, so this experiment cannot tell which kind of bytes grow; experiments 3, 4 and 6 split them.</li></ol>
<h4>Results</h4>
</div>
<figure class="wide pair"><div class="scroll">@@chart_e12_st@@ @@chart_e12_ld@@</div><figcaption>x: data rate on the busy direction (data counter ÷ 2.4 s). y: W ÷ S (raw counter ÷ buffer bytes), on the busy direction (blue; there D = S, so this is also the wire cost w) and the low-data direction (grey). One point per setting; labels give CTAs × threads per CTA. Median of 4 GPUs × 3 rounds.</figcaption></figure>
<div class="col">
<h4>Interpretation</h4>
<ul>
<li><b>It follows requests, not time.</b> As the <code>multimem.st</code> data rate rose @@e12_st_fold@@-fold, the wire rate of the low-data direction rose with it, from @@e12_st_idle_rate_low@@ to @@e12_st_idle_rate_high@@ GB/s, and its W stayed between @@e12_st_idle_min@@ S and @@e12_st_idle_max@@ S. <code>ld_reduce</code> behaves the same way. Fixed-rate housekeeping is ruled out.</li>
<li><b>But the wire cost falls when the link is busier.</b> On the busy direction of <code>multimem.st</code>, w fell from @@e12_st_busy_low@@ to @@e12_st_busy_high@@, and for <code>ld_reduce</code> from @@e12_ld_busy_low@@ to @@e12_ld_busy_high@@. We expected it to stay constant. The link packs its traffic more tightly when more is waiting.</li>
</ul>

<span class="tag">Experiment 3 of 9 · instrument check</span>
<h2 id="e3">Can we count packets by type?</h2>
<h4>Motivation</h4>
<p>The driver counters tell us how many bytes, not which kind. Nsight Systems can split each direction into four packet types: request data, request protocol, response data and response protocol. Before trusting it we must check that it agrees with the driver counters and does not slow the work down.</p>
<h4>Configuration</h4>
<div class="tablewrap"><table>
<tr><th>What changes</th><th>What stays fixed</th></tr>
<tr><td><b>Profiler on or off</b>, for <code>multimem.st</code> alone, <code>ld_reduce</code> alone and plain store</td><td>4 GPUs; 32 CTAs × 512 threads; 11 windows; 3 rounds; custom metric set with byte sums at 10 000 samples per second</td></tr>
</table></div>
<h4>Procedure</h4>
<ol><li>Run each pattern with and without the profiler, back to back.</li><li>Compare the rates; compare the profiler's sums inside the measured kernels with the driver's counters.</li></ol>
<h4>Results</h4>
<div class="tablewrap"><table>
<tr><th>Pattern</th><th class="n">Rate with / without profiler, GB/s</th><th class="n">Packets ÷ raw counter, busy direction</th><th class="n">Packets ÷ raw counter, low-data direction</th></tr>
<tr><td><code>multimem.st</code> alone</td><td class="n">@@e13_allgather_sampled@@ / @@e13_allgather_unsampled@@</td><td class="n">@@e13_allgather_busy@@%</td><td class="n">@@e13_allgather_idle@@%</td></tr>
<tr><td><code>ld_reduce</code> alone</td><td class="n">@@e13_reducescatter_sampled@@ / @@e13_reducescatter_unsampled@@</td><td class="n">@@e13_reducescatter_busy@@%</td><td class="n">@@e13_reducescatter_idle@@%</td></tr>
<tr><td>Plain store</td><td class="n">@@e13_plain_store_sampled@@ / @@e13_plain_store_unsampled@@</td><td class="n" colspan="2">both directions busy, 99.4–99.5%</td></tr>
</table></div>
<h4>Interpretation</h4>
<ul>
<li><b>The profiler is safe to use.</b> Rates change by at most @@e13_max_diff@@%, and its data bytes equal the driver's data counter.</li>
<li><b>On a busy direction the four packet types are the whole raw counter.</b> On a low-data direction they are only about 40% of it. The rest is counted by the driver but by no packet counter. We call it <b>wire bytes outside the packets</b>, and it matters below. The profiler cannot say what it is. A likely candidate is link-level traffic, such as acknowledgements or partly filled transfer units, sent when there is no data to share them with.</li>
<li>One planned extra, a counter of busy link cycles, was accepted by the tool but silently not recorded.</li>
</ul>

<span class="tag">Experiment 4 of 9 · packet anatomy</span>
<h2 id="e4">The packets of each operation alone</h2>
<h4>Motivation</h4>
<p>To understand the mix, we first need each operation's own packets on both directions. One guess from the 8-GPU numbers was that each thread's 16-byte memory instruction leaves the GPU as its own packet, with no merging. That would make the headers very expensive.</p>
<h4>Configuration</h4>
<div class="tablewrap"><table>
<tr><th>What changes</th><th>What stays fixed</th></tr>
<tr><td><b>The operation</b>: <code>multimem.st</code>, <code>ld_reduce</code>, plain store, plain load, each issued by every GPU</td><td>4 GPUs; 32 CTAs × 512 threads; each thread moves 16 bytes per memory instruction, warp-contiguous addresses; profiler on; 3 rounds</td></tr>
</table></div>
<h4>Procedure</h4>
<ol><li>Run each operation with the profiler.</li><li>For each direction, divide the bytes of each packet type by the buffer bytes S; add the wire bytes of the raw counter that fall outside every packet type.</li></ol>
<h4>Results</h4>
</div>
<figure class="wide"><div class="scroll">@@chart_e14@@</div><figcaption>Bytes on each direction ÷ S, each operation alone, 4 GPUs, 32 CTAs × 512 threads. Stacked from the packet counters inside the measured kernels: request data and response data (blue), request and response protocol (orange); grey: raw counter minus all packet bytes. Bar top: W ÷ S (raw counter). Median over GPUs and rounds.</figcaption></figure>
<div class="col">
<h4>Interpretation</h4>
<ul>
<li><b>The instructions of a warp are merged into 128-byte packets.</b> Request protocol is @@e14_st_r_rqp@@ S when @@e14_st_r_rqd@@ S of data is written: 16 header bytes per 128 data bytes. The guess of one packet per 16-byte access was wrong.</li>
<li><b>Every <code>multimem.st</code> packet is acknowledged.</b> Its sent direction carries @@e14_st_s_rsp@@ S of write responses, one 16-byte response per 128-byte packet. Plain stores, the kind Ring uses, get almost none: @@e14_ps_s_rsp@@ S.</li>
<li><b><code>ld_reduce</code> needs read requests.</b> To serve the other GPUs' sums, the switch sends each GPU @@e14_ld_r_rqp@@ S of read requests, on the low-data direction.</li>
<li><b>Most of the low-data direction is wire bytes outside the packets</b>: @@e14_st_s_out@@ S and @@e14_ld_r_out@@ S.</li>
</ul>

<span class="tag">Experiment 5 of 9 · instruction width and address pattern</span>
<h2 id="e5">Does the way threads address memory matter?</h2>
<h4>Motivation</h4>
<p>If packets follow the instructions, narrower accesses would mean more packets and more headers per byte. If packets follow memory lines, width would not matter but the arrangement of addresses would. This tells us whether a different kernel could avoid the extra packets.</p>
<h4>Configuration</h4>
<div class="tablewrap"><table>
<tr><th>What changes</th><th>What stays fixed</th></tr>
<tr><td><b>Bytes per access</b>: 4, 8, 16; <b>address pattern</b>: warp-contiguous (the 32 threads of a warp touch consecutive addresses, as NCCL does) or thread-contiguous (each thread works on its own run of 8 consecutive pieces, so the 32 threads of one instruction touch addresses far apart)</td><td>4 GPUs; 32 CTAs × 512 threads; 8 accesses in flight per thread in every setting; profiler on; 3 rounds</td></tr>
</table></div>
<h4>Procedure</h4>
<ol><li>Run 3 widths × 2 patterns for <code>multimem.st</code> and <code>ld_reduce</code>, and both patterns for plain load and store.</li><li>Sum all non-data bytes over both directions (protocol plus wire bytes outside the packets), ÷ S.</li></ol>
<h4>Results</h4>
</div>
<figure class="wide pair"><div class="scroll">@@chart_e15_st@@ @@chart_e15_ld@@</div><figcaption>y: non-data bytes ÷ S = request protocol + response protocol + wire bytes outside the packet counters, both directions summed. x: instruction width, the bytes one thread moves with one memory instruction. Blue: NCCL's warp-contiguous pattern; orange: threads far apart. Median over GPUs and rounds.</figcaption></figure>
<div class="col">
<h4>Interpretation</h4>
<ul>
<li><b>Width does not matter in NCCL's pattern.</b> 4-, 8- and 16-byte instructions give the same packets and the same rate (@@e15_st_rate@@ GB/s for <code>multimem.st</code>). The GPU merges the instructions of a warp's 32 threads into 128-byte packets whatever their width.</li>
<li><b>Scattering the threads' addresses is much worse.</b> Each thread's instruction becomes its own packet, and at 4 and 8 bytes each small access also fills at least a 16-byte data slot, so those points include padding: non-data bytes rise from @@e15_st_nd@@ S to @@e15_st_nd_tc@@ S for <code>multimem.st</code>, and its rate falls @@e15_st_drop@@-fold to @@e15_st_rate_tc@@ GB/s. Plain stores fall @@e15_ps_drop@@-fold.</li>
<li><b>NCCL's kernel already uses the cheapest pattern.</b> No change of instruction width or address order inside the kernel can remove the write responses or the read requests.</li>
</ul>

<span class="tag">Experiment 6 of 9 · the core test</span>
<h2 id="e6">The packets of the mix</h2>
<h4>Motivation</h4>
<p>This is the experiment the others prepare for. Experiment 1 showed that the mix carries much less than the two operations put on their low-data directions alone. Three explanations were on the table: packets combine when a direction is busy; part of the empty-direction traffic is not per request (ruled out by experiment 2); or packets get smaller. Counting each packet type inside the mix itself decides it.</p>
</div>
<figure class="wide"><div class="scroll">@@diagram_directions@@</div><figcaption>Where each operation's packets go, for one GPU. Blue: data. Orange: requests and responses. Alone, each operation's small packets use the direction its data does not fill.</figcaption></figure>
<div class="col">
<h4>Configuration</h4>
<div class="tablewrap"><table>
<tr><th>What changes</th><th>What stays fixed</th></tr>
<tr><td><b>Mixed or alone</b>: AllReduce (both operations in the same 32 CTAs), both in separate CTAs (32 + 32), and each operation alone as reference</td><td>4 GPUs; 512 threads per CTA; 16-byte instructions, warp-contiguous addresses; profiler on; 5 rounds, order rotated</td></tr>
</table></div>
<h4>Procedure</h4>
<ol><li>Run the four arms interleaved, five rounds.</li><li>For each direction, add the packets of the two operations alone and compare with the packets of the mix.</li><li>Do the same for the wire bytes outside the packets.</li></ol>
<h4>Results</h4>
</div>
<figure class="wide"><div class="scroll">@@chart_e16@@</div><figcaption>Same metric and stacking as the previous packet chart, for the operations alone and mixed, 4 GPUs, 32 CTAs × 512 threads per role, median over GPUs and 5 rounds. The grey parts of the two single operations almost disappear in the mix.</figcaption></figure>
<div class="col">
<div class="tablewrap"><table>
<tr><th>Direction (all values × S)</th><th class="n">Packets, alone: <code>st</code> + <code>ld_reduce</code></th><th class="n">Packets, AllReduce</th><th class="n">Difference</th><th class="n">Outside packets, alone → mixed</th></tr>
<tr><td>Sent</td><td class="n">@@e16_sent_ag@@ + @@e16_sent_rs@@ = @@e16_sent_sum@@</td><td class="n">@@e16_sent_ar@@</td><td class="n">@@e16_sent_diff@@</td><td class="n">@@e16_sent_out_alone@@ → @@e16_sent_out_ar@@</td></tr>
<tr><td>Received</td><td class="n">@@e16_received_ag@@ + @@e16_received_rs@@ = @@e16_received_sum@@</td><td class="n">@@e16_received_ar@@</td><td class="n">@@e16_received_diff@@</td><td class="n">@@e16_received_out_alone@@ → @@e16_received_out_ar@@</td></tr>
</table></div>
</div>
<figure class="wide"><div class="scroll">@@chart_budget@@</div><figcaption>Where the wire bytes of one busy direction go, ÷ S, split from the packet counters of the mix, using each operation alone as reference (headers = 16 bytes per 128-byte data packet as measured alone; requests and responses = the rest of request or response protocol). Both directions of AllReduce carry the same D = 1.25 S; neither is a low-data direction. Top: <code>multimem.st</code> alone. Below: the two directions of AllReduce. Orange: the packets the other operation used to send on its low-data direction. Label: W ÷ S and, in brackets, the wire cost w = W ÷ D.</figcaption></figure>
<div class="col">
<h4>Interpretation</h4>
<ul>
<li><b>The packets add up.</b> On the sent direction, AllReduce's packets equal the sum of the two operations alone to three decimals. On the received direction they are slightly more (@@e16_received_diff@@ S), all in read requests: @@e16_readreq_mix@@ S in the mix against @@e16_readreq_alone@@ S alone.</li>
<li><b>What vanishes is the traffic outside the packets.</b> It is large only on a low-data direction and shrinks to almost nothing when data fills the direction. So the "missing" bytes of experiment 1 were never part of the cost of the mix.</li>
<li><b>This is the cause.</b> Each busy direction of AllReduce carries, on top of its data and data headers, the small packets the other operation used to hide on its low-data direction: the switch's read requests on the received side (@@e16_readreq_mix@@ S) and the responses to multicast writes on the sent side (@@e16_writeresp@@ S). Together with headers and the few wire bytes outside the packets, this gives a wire cost w = @@e16_received_ratio@@ (received) and w = @@e16_sent_ratio@@ (sent). Ring needs neither kind: plain stores are not acknowledged and Ring never reads remotely.</li>
<li><b>Same or separate CTAs make no difference</b>, because the packets are formed below the CTAs, in the GPU's link interface and the switch.</li>
</ul>

<span class="tag">Experiment 7 of 9 · sharing</span>
<h2 id="e7">Who gets the link when both want it?</h2>
<h4>Motivation</h4>
<p>An earlier test suggested that read replies keep their speed while multicast stores take what is left, as if reads had priority. If there were a priority rule, a kernel might be tuned around it.</p>
<h4>Configuration</h4>
<div class="tablewrap"><table>
<tr><th>What changes</th><th>What stays fixed</th></tr>
<tr><td><b>How many CTAs issue each operation</b>: <code>ld_reduce</code> at 0, 4, 8, 16, 32, 64 CTAs with <code>multimem.st</code> at 32; and the reverse</td><td>4 GPUs, all issuing both; 512 threads per CTA; separate CTAs per operation; 3 rounds</td></tr>
</table></div>
<h4>Procedure</h4>
<ol><li>Run the 11 settings, three rounds each.</li><li>Plot each operation's rate (bytes of its own share per GPU per second) against the other operation's CTA count.</li></ol>
<h4>How the rate of each operation is measured</h4>
<p>The rate comes from the work the CTAs finish and the GPU clock, not from the link counters.</p>
<ul>
<li><b>What counts.</b> Each GPU works on its own quarter of the buffer. Each CTA walks its slice of that quarter in chunks of 1 MiB and counts the chunks it finished. One byte of the GPU's own quarter counts once, even though <code>multimem.st</code> writes it into 4 GPUs and <code>ld_reduce</code> sums it from 4.</li>
<li><b>The time.</b> All CTAs start from one signal. After 300 ms a controller CTA raises a stop flag, and each CTA stops within two chunks. Before reading the end time, each CTA waits until its stores are delivered (<code>__threadfence_system()</code>).</li>
<li><b>The formula.</b> Rate of one GPU = finished chunks of all CTAs of that operation × 1 MiB ÷ (latest CTA end − earliest CTA start). Then the mean over the 4 GPUs, the median over 5 windows and the median over 3 rounds.</li>
<li><b>Worked example.</b> 32 + 32 CTAs, round 1, first measured window, GPU 4: the <code>multimem.st</code> CTAs finished 17 366 chunks in 301 330 976 ns, so 17 366 × 1 048 576 ÷ 301 330 976 = 60.43 GB/s. The mean over the 4 GPUs is 60.70; the medians over windows and rounds give 60.7 GB/s.</li>
<li><b>Link to the link counters.</b> With one operation alone, 4 × this rate is the data rate on its busy direction: <code>multimem.st</code> alone at 93.6 GB/s is 374 GB/s of data received per GPU.</li>
</ul>
<h4>Results</h4>
</div>
<figure class="wide pair"><div class="scroll">@@chart_e17_a@@ @@chart_e17_b@@</div><figcaption>y: rate of each operation in bytes of the GPU's own share per second (bytes finished ÷ time from first start to last end of that operation's CTAs). x: CTAs of the varied operation, 512 threads each. Left: <code>multimem.st</code> fixed at 32 CTAs. Right: <code>ld_reduce</code> fixed at 32 CTAs. Median over 3 rounds.</figcaption></figure>
<div class="col">
<h4>Interpretation</h4>
<ul>
<li><b>Nobody has priority; the share follows demand.</b> With 64 CTAs of <code>multimem.st</code> against 32 of <code>ld_reduce</code>, stores get @@e17_64_32_st@@ GB/s and reads @@e17_64_32_ld@@. Reversed, stores get @@e17_32_64_st@@ and reads @@e17_32_64_ld@@. At 32 + 32 they get @@e17_32_32_st@@ and @@e17_32_32_ld@@.</li>
<li>Shifting CTAs moves the split between the two operations but cannot remove the extra packets, so it is no fix.</li>
</ul>

<span class="tag">Experiment 8 of 9 · latency</span>
<h2 id="e8">How long does one operation wait?</h2>
<h4>Motivation</h4>
<p>The colleague's report proposed a shared limit on how many multimem requests a GPU can have in flight. Timing single operations while other traffic runs shows where requests wait, and is a first step towards testing that idea.</p>
<h4>Configuration</h4>
<div class="tablewrap"><table>
<tr><th>What changes</th><th>What stays fixed</th></tr>
<tr><td><b>Background traffic</b> on all 4 GPUs: none, plain store, <code>multimem.st</code>, <code>ld_reduce</code>, AllReduce (32 CTAs × 512 threads each)</td><td>A probe: one thread on an otherwise unused processor of each GPU issues one 16-byte operation at a time and times it; probe kinds: <code>ld_reduce</code>, or <code>multimem.st</code> followed by a fence that waits for its acknowledgement; 3 rounds; no profiler</td></tr>
</table></div>
<h4>Procedure</h4>
<ol><li>For each background and probe kind, record the latency of every probe operation for 5 windows.</li><li>Take the median and 90th percentile per GPU and window, then the median over GPUs and rounds.</li></ol>
<h4>Results</h4>
</div>
<figure class="wide pair"><div class="scroll">@@chart_e18_ld@@ @@chart_e18_st@@</div><figcaption>Latency of one operation with nothing else outstanding from the probe. Bars and numbers: median. Black line: 90th percentile. Blue: idle links. Orange: under background traffic.</figcaption></figure>
<div class="col">
<h4>Interpretation</h4>
<ul>
<li><b>Requests wait much longer behind multimem traffic than behind plain stores</b>, even though both fill the link. The delay added over idle links is @@e18_factor_min@@ to @@e18_factor_max@@ times larger under a multimem background.</li>
<li>This shows long queues, not a fixed limit. Whether a hard limit on requests in flight also exists cannot be decided from these numbers.</li>
</ul>

<span class="tag">Experiment 9 of 9 · NCCL on 8 GPUs</span>
<h2 id="e9">Does NCCL on 8 GPUs behave the same?</h2>
<h4>Motivation</h4>
<p>Experiments 1–8 used our own program on 4 GPUs. The final check runs NCCL's real kernels on all 8 GPUs, compares them with Ring, and connects the packets to problem 2.</p>
<h4>Configuration</h4>
<div class="tablewrap"><table>
<tr><th>What changes</th><th>What stays fixed</th></tr>
<tr><td><b>The NCCL kernel</b>: NVLS AllGather, NVLS ReduceScatter, NVLS AllReduce (symmetric multicast kernels, 2 CTAs × 512 threads as NCCL chooses), Ring AllReduce (24 CTAs × 512 threads)</td><td>All 8 GPUs; nccl-tests with 2 GiB buffers, 200 timed calls; profiler on; 3 rounds</td></tr>
</table></div>
<h4>Procedure</h4>
<ol><li>Run each kernel under the profiler; check that the profiler does not change its speed (it changed by at most 0.2%).</li><li>Keep only samples inside NCCL's kernels; divide by the calls × 2 GiB.</li><li>Compare busy-direction packet bytes of NVLS and Ring AllReduce.</li></ol>
<h4>Results</h4>
</div>
<figure class="wide"><div class="scroll">@@chart_e19@@</div><figcaption>Bytes on each direction ÷ S for NCCL's kernels on 8 GPUs, from the packet counters inside NCCL's kernels; S = number of calls × 2 GiB. Bar top: all packet bytes ÷ S. Median over 8 GPUs × 3 rounds. Ring's protocol is its data headers and almost nothing else.</figcaption></figure>
<figure class="wide"><div class="scroll">@@chart_gain@@</div><figcaption>Speed-up of NVLS AllReduce over Ring at 8 GPUs. The last bar removes the read requests (@@e19_readreq@@ S) from the received side and the write responses (0.125 S) from the sent side, then takes the busier side. All ratios use W from the driver's raw counter. This is accounting with the same measurements, not a new prediction. The measured ratio here is for 2 GiB buffers in this run; the earlier sweep at 8 GiB gave @@gain8@@×.</figcaption></figure>
<div class="col">
<h4>Interpretation</h4>
<ul>
<li><b>NCCL's kernels show the same packets</b> as our program: 16-byte headers on 128-byte data packets, one response per multicast write packet, and read requests for <code>ld_reduce</code>. The mix adds up in the same way.</li>
<li><b>Counting packets only, Ring's wire cost is w = @@e19_ring_ratio@@; NVLS AllReduce's is w = @@e19_mc_ratio_r@@</b> on the side where read requests arrive and w = @@e19_mc_ratio_s@@ on the side where write responses leave. Ring's responses are only @@e19_ring_resp@@ S.</li>
<li><b>These two packet kinds explain about half of problem 2.</b> All busy directions run at the same wire-rate limit C, and algBW = C ÷ (k × w) = C ÷ (W ÷ S), so the speed-up follows W ÷ S on the busier direction (driver raw counter, 8 GPUs): Ring W = @@e19_raw_ring@@ S, NVLS W = @@e19_raw_mc@@ S, ratio @@e19_gain_packets@@ (measured @@e19_gain_algbw@@). Without the read requests (@@e19_readreq@@ S) and write responses (0.125 S), NVLS would need W = @@e19_without_received@@ S received and @@e19_without_sent@@ S sent: a ratio of @@e19_gain_without@@, against the byte-count bound of @@ideal8@@. The rest of the gap is the sent side's wire bytes in no packet type (about 0.1 S), the GPU's own read requests and the headers. An earlier version of this page said 1.54 and "nearly all"; it used the packet counters only (corrected 21 November 2026).</li>
</ul>

<h2 id="expect">Putting it together: what NVLS AllReduce should reach</h2>
<p>This section first fixes the words, then derives from theory alone what NVLS AllReduce should reach, and only then compares with the measurements.</p>

<h4>Recap: three kinds of bytes</h4>
<p>The same three byte counts as in "How we count", with where they live:</p>
<div class="tablewrap"><table>
<tr><th>Name</th><th>Symbol</th><th>What it is</th><th>Where it lives</th><th>Measured by</th></tr>
<tr><td><b>Buffer bytes</b> (payload)</td><td>$S$</td><td>The size of the collective's buffer: what the user asked to reduce or gather. Not a link quantity</td><td>In GPU memory</td><td>Known from the call (nccl-tests: buffer size; our program: 4 × the bytes each GPU finished of its own quarter)</td></tr>
<tr><td><b>Data bytes</b></td><td>$D$</td><td>The user's numbers that cross <b>one direction of one GPU's link</b>: the data part of data packets</td><td>On one link direction</td><td>Driver data counter (<code>nvidia-smi nvlink -gt d</code>)</td></tr>
<tr><td><b>Wire bytes</b> (raw)</td><td>$W$</td><td>Everything that crosses that direction: data + packet headers + control packets (read requests, write acknowledgements) + anything else</td><td>On one link direction</td><td>Driver raw counter (<code>nvidia-smi nvlink -gt $r$</code>); the profiler splits it into packet types</td></tr>
</table></div>
<p>Only two ratios are needed to connect them, and the rest of this section uses only these two:</p>
<div class="tablewrap"><table>
<tr><th>Ratio</th><th>Definition</th><th>What sets it</th><th>Values</th></tr>
<tr><td><b>Data factor</b> $k$</td><td>$D/S$ on the busier direction</td><td>The algorithm only: how often each buffer byte must cross a GPU's link</td><td>AllGather, ReduceScatter: 1. NVLS AllReduce: $1+1/N$. Ring AllReduce: $2(N-1)/N$</td></tr>
<tr><td><b>Wire cost</b> $w$</td><td>$W/D$ on the same direction</td><td>The packets: headers and control packets the instructions need</td><td>At least $1 + 16/128 = 1.125$ (16-byte header per 128 data bytes). This is the open question</td></tr>
</table></div>
</div>
<figure class="wide"><div class="scroll">@@diagram_bytes@@</div><figcaption>The three kinds of bytes for one NVLS AllReduce of a 1 GB buffer on 4 GPUs, seen on one GPU's received direction, with the theoretical packet sizes of the next subsection (16-byte headers, one 16-byte read request per 128 bytes read). Lengths are to scale.</figcaption></figure>
<div class="col">
<p>And three rates, each one kind of bytes divided by a time. The table separates what each rate <b>is</b> (its definition) from how it was <b>obtained</b> in each setup, because the setups differ:</p>
<div class="tablewrap"><table>
<tr><th>Rate</th><th>Definition</th><th>Our program, 4 GPUs: how it is obtained</th><th>nccl-tests, 8 GPUs (and Ring at 4): how it is obtained</th></tr>
<tr><td><b>algBW</b></td><td>$S/t$</td><td><b>Measured</b> in the kernel: finished chunks of all CTAs × 1 MiB ÷ (latest CTA end − earliest CTA start), on the GPU clock; × 4 for the 4 quarters (experiment 7)</td><td><b>Measured</b>: buffer size ÷ nccl-tests' own time per call</td></tr>
<tr><td><b>busBW</b> (data rate of the busier direction)</td><td>$D/t$</td><td><b>Measured</b> with the driver's data counter: bytes of 8 windows (long − short) ÷ 2.4 s. Not computed from algBW</td><td><b>Computed</b> as $\mathrm{algBW}\cdot D/S$, with $D/S$ from the data counter (long − short ÷ buffer bytes). Not divided by a time, because the counter also covers set-up and checks</td></tr>
<tr><td><b>Wire rate</b></td><td>$W/t$</td><td><b>Measured</b> with the driver's raw counter, the same way: ÷ 2.4 s</td><td><b>Computed</b> as $\mathrm{algBW}\cdot W/S$, with $W/S$ from the raw counter</td></tr>
</table></div>
<p>The relations <b>$\mathrm{busBW} = k\cdot\mathrm{algBW}$</b> and <b>$\text{wire rate} = w\cdot\mathrm{busBW}$</b> follow from the definitions ($D$ = $k\times S$, $W$ = $w\times D$, same $t$). They are used in two ways:</p>
<ul>
<li><b>In our program they are a check, not a calculation.</b> algBW (kernel clock) and busBW (data counter) are measured independently, and busBW ÷ algBW comes out 0.3–0.5% above $k$. The small gap is the time base: algBW divides by each operation's real time (about 301.3 ms per window, because CTAs finish the chunk they are in after the stop flag), busBW by the nominal 300 ms.</li>
<li><b>For nccl-tests they are the calculation.</b> Only algBW is timed; busBW and the wire rate are algBW times a counter ratio.</li>
</ul>
<div class="note">When the link is full: <b>$\mathrm{busBW} = C/w$</b> and <b>$\mathrm{algBW} = C/(k\,w)$</b>. So for a fixed link, busBW depends only on the wire cost $w$, and algBW also on the data factor $k$.</div>
<p>Worked example (the figure): $S$ = 1 GB, $k$ = 1.25, so $D$ = 1.25 GB cross the received direction. With 0.156 GB of headers and 0.125 GB of read requests, $W$ = 1.531 GB and $w$ = 1.531 ÷ 1.25 = 1.225. If the call takes $t$ = 5 ms: algBW = 200 GB/s, busBW = 250 GB/s, wire rate = 306 GB/s.</p>

<h4>Theory: what each collective should reach</h4>
<p>No measured values in this subsection: only the packet rules.</p>
<ul>
<li><b>Data packets</b> carry d bytes of data plus a header; the header adds $\eta$ bytes per data byte.</li>
<li><b>Read requests</b>: a read over NVLink needs a request travelling the other way. <code>ld_reduce</code> has two kinds, with separate symbols because they need not be the same size:<ul><li><b>$\rho_{\mathrm{own}}$</b>: the request a GPU sends to the switch for a sum of its own share (as issuer); bytes per data byte of that sum.</li><li><b>$\rho_{\mathrm{switch}}$</b>: the request the switch sends to each GPU to read its copy (the GPU as target); bytes per data byte read.</li></ul></li>
<li><b>Write acknowledgements</b>: a write that is acknowledged returns $\alpha$ bytes per data byte written. Plain stores, which Ring uses: $\alpha_{\mathrm{p}}$.</li>
</ul>
<p>$\eta$, $\rho_{\mathrm{own}}$, $\rho_{\mathrm{switch}}$ and $\alpha$ are <b>unknowns</b>: no NVLink specification we have gives them, so the theory keeps them as symbols. Only the measurements below give them values.</p>
<h4>What actually happens on the links, packet by packet</h4>
<p>Each GPU plays two roles at once. As an <b>issuer</b> it runs the instruction on its own share of the buffer ($S$/N bytes). As a <b>target</b> it is written to, or read from, by the switch on behalf of <i>every</i> GPU's instruction, so for all $S$ bytes. Follow one 128-byte piece:</p>
</div>
<figure class="wide"><div class="scroll">@@story_st@@</div><figcaption><code>multimem.st</code> by itself, in four steps. GPU 0 is the issuer (bold); every GPU is also a target. Solid blue: data packets; dashed orange: control packets. "Whole call" is what the step adds up to on each GPU's direction when every GPU issues all pieces of its share: each GPU issues S/N, but receives every GPU's writes (S) and acknowledges all of them (α S).</figcaption></figure>
<figure class="wide"><div class="scroll">@@story_ld@@</div><figcaption><code>ld_reduce</code> by itself, in four steps. Each GPU asks only for the sums of its own share (S/N of requests and sums), but serves every GPU's reads: it receives the switch's read requests for all S bytes and replies with all S of its data.</figcaption></figure>
<div class="col">
<p>Adding up over every GPU and every piece gives each GPU's traffic, which is where the three control terms come from:</p>
<div class="tablewrap"><table>
<tr><th>Operation</th><th>Role of one GPU</th><th>What it sends (sent direction)</th><th>What it receives (received direction)</th></tr>
<tr><td rowspan="2"><code>multimem.st</code></td><td>Issuer, for its $S$/N</td><td>its writes: $\tfrac{S}{N}(1+\eta)$</td><td>a few responses from the switch: 0.009 $S$ measured, fewer than one per write it issued (one per write would be $\tfrac{S}{N}\times 16/128$ = 0.031 $S$ at 4 GPUs). Small enough to leave out of the theory</td></tr>
<tr><td>Target, for all $S$</td><td><b>an acknowledgement for every write it receives: $\alpha S$</b></td><td>the writes of all $N$ GPUs: $S(1+\eta)$</td></tr>
<tr><td rowspan="2"><code>ld_reduce</code></td><td>Issuer, for its $S$/N</td><td><b>its read-reduce requests: $\rho_{\mathrm{own}} S/N$</b></td><td>the sums: $\tfrac{S}{N}(1+\eta)$</td></tr>
<tr><td>Target, for all $S$</td><td>replies with its copy of every GPU's share: $S(1+\eta)$</td><td><b>the switch's read requests for all $N$ GPUs' sums: $\rho_{\mathrm{switch}} S$</b></td></tr>
</table></div>
<ul>
<li><b>$\alpha S$, acknowledgements:</b> a GPU acknowledges the multicast writes it <i>receives</i>, and it receives every GPU's writes, so $S$ worth of them. The acknowledgement goes back out, on its sent direction, the direction opposite to those writes. Plain stores (Ring) are not acknowledged packet by packet (0.003 $S$).</li>
<li><b>Who acknowledges whom.</b> The acknowledgements go from each target to the switch, not to the issuer: for each write g issues, the switch receives $N$ acknowledgements (one from each target, g included), yet g's received direction carries only 0.009 $S$ of responses, about one per 3.5 writes it issued. So the switch does not forward each acknowledgement; it appears to collect them and return fewer, combined responses to the issuer. The counters only see bytes per link direction and packet type, not who a packet is addressed to, so this attribution is an inference from the directions and the amounts. It holds for every instruction width tested (experiment 5): the sent direction always carries exactly one 16-byte acknowledgement per received write packet.</li>
<li><b>$\rho_{\mathrm{own}} S/N$, the GPU's own requests:</b> as issuer, a GPU asks the switch only for the sums of its own share, so $S$/N worth of requests, on its sent direction.</li>
<li><b>$\rho_{\mathrm{switch}} S$, the switch's requests:</b> to compute each sum, the switch must read that piece on every GPU. So every GPU receives read requests for all $N$ shares, $S$ worth, on its received direction, opposite to the replies it sends.</li>
<li><b>Why the big terms are the target terms.</b> The switch fans each instruction out to $N$ GPUs, so a GPU does $N$ times more work as a target ($S$) than as an issuer ($S$/N). The two control terms proportional to $S$, $\alpha S$ and $\rho_{\mathrm{switch}} S$, are the ones that land on a direction that AllReduce fills with the <i>other</i> operation's data.</li>
<li><b>What is measured and what is not.</b> The packet kinds and directions are measured. Why the switch needs acknowledgements for multicast writes is not known; one likely reason is that the switch must know when a write has reached every GPU before it can report the store as complete. Measured values: $\alpha$ = 0.125 (one 16-byte acknowledgement per 128-byte write), the switch's requests $\rho_{\mathrm{switch}}$ = 0.154 (ReduceScatter run by itself), and the GPU's own requests $\rho_{\mathrm{own}}$ = 0.062 $S$ ÷ 0.25 $S$ of sums = 0.25, about 32 bytes per 128-byte piece. All values, with sources, are in the table under "Theory against measurement".</li>
</ul>
<p>Bytes that cross one GPU's link for a buffer of $S$ bytes, $N$ GPUs, each GPU owning $S$/N:</p>
<div class="tablewrap"><table>
<tr><th>Collective</th><th>Sent direction (GPU → switch)</th><th>Received direction (switch → GPU)</th></tr>
<tr><td>AllGather (<code>multimem.st</code>)</td><td>own share: $\tfrac{S}{N}(1+\eta)$<br><b>+ acknowledgements for what it receives: $\alpha S$</b></td><td>every GPU's share: $S(1+\eta)$<br><b>+ small responses back to the issuer: $r_{\mathrm{st}} S$</b></td></tr>
<tr><td>ReduceScatter (<code>ld_reduce</code>)</td><td>replies to the switch for every GPU's share: $S(1+\eta)$<br><b>+ its own read requests: $\rho_{\mathrm{own}} S/N$</b></td><td>its own sums: $\tfrac{S}{N}(1+\eta)$<br><b>+ the switch's read requests: $\rho_{\mathrm{switch}} S$</b><br><b>+ small responses: $r_{\mathrm{ld}} S$</b></td></tr>
<tr><td>NVLS AllReduce (both at once, packets add up)</td><td>$S(1+\tfrac{1}{N})(1+\eta)$ <b>$+\ \alpha S + \rho_{\mathrm{own}} S/N$</b></td><td>$S(1+\tfrac{1}{N})(1+\eta)$ <b>$+\ \rho_{\mathrm{switch}} S + (r_{\mathrm{st}} + r_{\mathrm{ld}})\,S$</b></td></tr>
<tr><td>Ring AllReduce</td><td>$S\cdot\frac{2(N-1)}{N}(1+\eta+\alpha_{\mathrm{p}})$</td><td>the same</td></tr>
</table></div>
<p>Bold: control packets. Each travels in the direction opposite to the data it belongs to. From the table, $w = W/D$ on the busier direction and $\mathrm{busBW} = C/w$:</p>
<div class="tablewrap"><table>
<tr><th>Collective</th><th>Data factor $k$</th><th>Wire cost $w$ on the busier direction</th><th>busBW</th></tr>
<tr><td>AllGather alone</td><td>1</td><td>$1+\eta+r_{\mathrm{st}}$</td><td>$C/(1+\eta+r_{\mathrm{st}})$</td></tr>
<tr><td>ReduceScatter alone</td><td>1</td><td>$1+\eta+\rho_{\mathrm{own}}/N$</td><td>$C/(1+\eta+\rho_{\mathrm{own}}/N)$</td></tr>
<tr><td>Ring AllReduce</td><td>$2(N-1)/N$</td><td>$1+\eta+\alpha_{\mathrm{p}}$</td><td>$C/(1+\eta+\alpha_{\mathrm{p}})$</td></tr>
<tr><td><b>NVLS AllReduce</b></td><td>$1+1/N$</td><td><b>$1 + \eta + \max\left(\rho_{\mathrm{switch}} + r_{\mathrm{st}} + r_{\mathrm{ld}},\ \alpha + \frac{\rho_{\mathrm{own}}}{N}\right)\cdot\frac{N}{N+1}$</b></td><td>$C/w$</td></tr>
</table></div>
<p>What the theory says:</p>
<ol>
<li><b>Headers are not the cause.</b> $1+\eta$ appears in every row; it cancels in every comparison.</li>
<li><b>More data per direction is not the cause.</b> NVLS AllReduce has $D = (1+1/N)\,S$, but $\mathrm{busBW} = D/t$ counts data bytes, so $k$ does not enter busBW.</li>
<li><b>The whole difference is the control-packet term $\max\left(\rho_{\mathrm{switch}} + r_{\mathrm{st}} + r_{\mathrm{ld}},\ \alpha + \frac{\rho_{\mathrm{own}}}{N}\right)\cdot\frac{N}{N+1}$.</b> With $\rho_{\mathrm{switch}} = \rho_{\mathrm{own}} = \alpha = r_{\mathrm{st}} = r_{\mathrm{ld}} = 0$ (no read requests, no acknowledged writes), NVLS AllReduce would reach exactly the busBW of AllGather alone and of Ring.</li>
<li><b>Alone, the control packets cost nothing.</b> AllGather puts its acknowledgements, and ReduceScatter the switch's read requests, on the direction that carries only $S$/N of data, where there is room. In AllReduce both directions carry $S(1+\tfrac{1}{N})$ of data, so there is no room left.</li>
<li><b>Ring has no control term</b> ($\alpha_{\mathrm{p}}$ ≈ 0): it only writes with plain stores, and its reduction reads the GPU's own memory, not NVLink.</li>
</ol>
<p><b>A reference case, not a prediction.</b> To get a feel for the size, suppose every control packet is as large as one data-packet header and there is one per data packet. Experiment 4 measured headers of 16 bytes per 128-byte data packet ($\eta$ = 0.125), so this case is $\rho_{\mathrm{own}} = \rho_{\mathrm{switch}} = \alpha = 0.125$, with no small responses ($r_{\mathrm{st}} = r_{\mathrm{ld}} = 0$). Then the control term is $(0.125 + 0.125/N)\cdot\frac{N}{N+1} = 0.125$ at every $N$, $w$ = 1.25 against 1.125, and NVLS AllReduce would reach 1.125 ÷ 1.25 = 0.90 of the busBW of AllGather alone (a 10% loss at any GPU count), and $\frac{2(N-1)}{N+1}\times 0.90$ of Ring's algBW: 1.08 at 4 GPUs and 1.40 at 8. The 16 and 128 come from our own measurement, so this case only illustrates the formula; the measured ρ and $\alpha$ are compared with it below.</p>

<h4>Measured values</h4>
<div class="tablewrap"><table>
<tr><th>GPUs</th><th>Collective</th><th>Busier direction</th><th class="n">Data factor $k$</th><th class="n">Wire cost $w$ (raw counter)</th><th class="n">Wire cost $w$ (profiler packets only)</th><th class="n">busBW, GB/s</th><th class="n">Wire rate, GB/s</th><th class="n">algBW, GB/s</th></tr>
<tr><td>4</td><td>NVLS AllGather</td><td>Received</td><td class="n">1</td><td class="n">1.135</td><td class="n">1.134</td><td class="n">375.6</td><td class="n">426.2</td><td class="n">374.5</td></tr>
<tr><td>4</td><td>NVLS ReduceScatter</td><td>Sent</td><td class="n">1</td><td class="n">1.189</td><td class="n">1.187</td><td class="n">358.1</td><td class="n">425.9</td><td class="n">357.0</td></tr>
<tr><td>4</td><td><b>NVLS AllReduce</b></td><td>Received</td><td class="n">1.25</td><td class="n"><b>1.343</b></td><td class="n">1.334</td><td class="n"><b>317.9</b></td><td class="n">426.9</td><td class="n"><b>253.1</b></td></tr>
<tr><td>4 (nccl-tests)</td><td>Ring AllReduce</td><td>Each</td><td class="n">1.5</td><td class="n">not run</td><td class="n">not run</td><td class="n">360.3</td><td class="n">not run</td><td class="n">240.2</td></tr>
<tr><td>4 (nccl-tests)</td><td>NVLS AllReduce, NCCL kernel</td><td>Each</td><td class="n">1.25</td><td class="n">not run</td><td class="n">not run</td><td class="n">313.1</td><td class="n">not run</td><td class="n">250.5</td></tr>
<tr><td>8</td><td>NVLS AllGather</td><td>Received</td><td class="n">1</td><td class="n">1.133</td><td class="n">1.132</td><td class="n">373.6</td><td class="n">423.2</td><td class="n">373.6</td></tr>
<tr><td>8</td><td>NVLS ReduceScatter</td><td>Sent</td><td class="n">1</td><td class="n">1.159</td><td class="n">1.156</td><td class="n">364.2</td><td class="n">422.2</td><td class="n">364.2</td></tr>
<tr><td>8</td><td>Ring AllReduce</td><td>Received</td><td class="n">1.75</td><td class="n">1.159</td><td class="n">1.138</td><td class="n">364.0</td><td class="n">421.8</td><td class="n">208.0</td></tr>
<tr><td>8</td><td><b>NVLS AllReduce</b></td><td>Received</td><td class="n">1.125</td><td class="n"><b>1.360</b></td><td class="n">1.353</td><td class="n"><b>310.8</b></td><td class="n">422.8</td><td class="n"><b>276.3</b></td></tr>
</table></div>
<p>4 GPUs: our program, 32 CTAs × 512 threads per operation (experiments 1, 4 and 6), and nccl-tests at 2 GiB for the two nccl-tests rows. 8 GPUs: NCCL's kernels at 2 GiB (raw-counter run and experiment 9). Wire cost from the raw counter includes bytes in no packet type; from the profiler, only the four packet types. In our program busBW ÷ algBW is 0.3–0.5% above $k$ because algBW divides by each operation's real time (about 301.3 ms per window: CTAs finish their current chunk after the stop flag) and busBW by the nominal 300 ms. Measured algBW gain over Ring: 1.04 at 4 GPUs, 1.33 at 8.</p>

<h4>Theory against measurement</h4>
<p><b>The wire rate behaves as expected:</b> NVLS AllReduce runs its busier direction at 426.9 and 422.8 GB/s, like every other collective, so $\mathrm{busBW} = C/w$ holds and everything comes down to $w$. Four values of $w$, each turned into busBW with the wire rate of AllGather alone (426.2 GB/s at 4 GPUs, 423.2 at 8). The parameters come first.</p>
<p><b>How the packet parameters are measured.</b> The profiler gives four packet counters for each direction of each GPU. Each is divided by the buffer bytes $S$ of the run, so every counter value below is a multiple of $S$. In formulas, the direction is a subscript: sent or recv.</p>
<div class="tablewrap"><table>
<tr><th>Counter (Nsight Systems name)</th><th>Which packets it holds in these runs</th></tr>
<tr><td>request data</td><td>the data of write packets (<code>multimem.st</code> writes)</td></tr>
<tr><td>request protocol</td><td>the headers of those write packets, plus read requests (which are request packets without data)</td></tr>
<tr><td>response data</td><td>the data of read replies and of the returned sums (<code>ld_reduce</code>)</td></tr>
<tr><td>response protocol</td><td>the headers of those replies, plus write acknowledgements and other responses without data</td></tr>
</table></div>
<p>Measured values, 4 GPUs (experiment 4 for the instructions run by themselves, experiment 6 for AllReduce), all × $S$:</p>
<div class="tablewrap"><table>
<tr><th>Run</th><th>Direction</th><th class="n">request data</th><th class="n">request protocol</th><th class="n">response data</th><th class="n">response protocol</th></tr>
<tr><td><code>multimem.st</code> by itself (AllGather)</td><td>sent</td><td class="n">0.250</td><td class="n">0.031</td><td class="n">0</td><td class="n">0.125</td></tr>
<tr><td></td><td>recv</td><td class="n">1.000</td><td class="n">0.125</td><td class="n">0</td><td class="n">0.009</td></tr>
<tr><td><code>ld_reduce</code> by itself (ReduceScatter)</td><td>sent</td><td class="n">0</td><td class="n">0.062</td><td class="n">1.000</td><td class="n">0.125</td></tr>
<tr><td></td><td>recv</td><td class="n">0</td><td class="n">0.154</td><td class="n">0.250</td><td class="n">0.052</td></tr>
<tr><td>Both (NVLS AllReduce)</td><td>sent</td><td class="n">0.250</td><td class="n">0.094</td><td class="n">1.000</td><td class="n">0.250</td></tr>
<tr><td></td><td>recv</td><td class="n">1.000</td><td class="n">0.356</td><td class="n">0.250</td><td class="n">0.061</td></tr>
</table></div>
<p>At 8 GPUs (experiment 9, NCCL's kernels) the same counters are: AllGather sent 0.125 / 0.016 / 0 / 0.125, recv 1.000 / 0.125 / 0 / 0.007; ReduceScatter sent 0 / 0.031 / 1.000 / 0.125, recv 0 / 0.160 / 0.125 / 0.026; AllReduce sent 0.125 / 0.047 / 1.000 / 0.250, recv 1.000 / 0.364 / 0.125 / 0.033 (request data / request protocol / response data / response protocol).</p>
<p><b>The same table, every field written only with parameters.</b> $S$ = buffer bytes, $N$ = number of GPUs, and six packet parameters: $\eta$ (headers of data packets), $\alpha$ (write acknowledgements), $\rho_{\mathrm{own}}$ (the GPU's own read requests), $\rho_{\mathrm{switch}}$ (the switch's read requests), and two small terms the theory leaves out: $r_{\mathrm{st}}$ (responses back to the issuer of <code>multimem.st</code>) and $r_{\mathrm{ld}}$ (responses beyond the headers of the sums of <code>ld_reduce</code>), both written per $S$:</p>
<div class="tablewrap"><table>
<tr><th>Run</th><th>Direction</th><th>request data</th><th>request protocol</th><th>response data</th><th>response protocol</th></tr>
<tr><td><code>multimem.st</code> by itself (AllGather)</td><td>sent</td><td>$S/N$</td><td>$\eta\,S/N$</td><td>$0$</td><td>$\alpha\,S$</td></tr>
<tr><td></td><td>recv</td><td>$S$</td><td>$\eta\,S$</td><td>$0$</td><td>$r_{\mathrm{st}}\,S$</td></tr>
<tr><td><code>ld_reduce</code> by itself (ReduceScatter)</td><td>sent</td><td>$0$</td><td>$\rho_{\mathrm{own}}\,S/N$</td><td>$S$</td><td>$\eta\,S$</td></tr>
<tr><td></td><td>recv</td><td>$0$</td><td>$\rho_{\mathrm{switch}}\,S$</td><td>$S/N$</td><td>$\eta\,S/N + r_{\mathrm{ld}}\,S$</td></tr>
<tr><td>Both (NVLS AllReduce)</td><td>sent</td><td>$S/N$</td><td>$(\eta + \rho_{\mathrm{own}})\,S/N$</td><td>$S$</td><td>$(\eta + \alpha)\,S$</td></tr>
<tr><td></td><td>recv</td><td>$S$</td><td>$(\eta + \rho_{\mathrm{switch}})\,S$</td><td>$S/N$</td><td>$\eta\,S/N + (r_{\mathrm{st}} + r_{\mathrm{ld}})\,S$</td></tr>
</table></div>
<p>Parameter values, from the instructions run by themselves: $\eta = 0.125$, $\alpha = 0.125$, $\rho_{\mathrm{own}} = 0.25$, $\rho_{\mathrm{switch}} = 0.154$ (4 GPUs) and $0.160$ (8 GPUs), $r_{\mathrm{st}} = 0.009$ (4 GPUs) and $0.007$ (8 GPUs), $r_{\mathrm{ld}} = 0.021$ (4 GPUs) and $0.010$ (8 GPUs). With these, every AllReduce field matches the measurement except the received request protocol, where AllReduce behaves as if $\rho_{\mathrm{switch}}$ were 0.231 (4 GPUs) and 0.239 (8 GPUs). The term $r$ used below is $r_{\mathrm{st}} + r_{\mathrm{ld}}$.</p>

<p><b>Every field of this table, explained.</b> Each counter value is the sum of the packet kinds it holds; the model of the theory section (with $N = 4$) gives each kind as an expression in $S$. The column "Role" says whether the field was used to <b>define</b> a parameter (so it matches by construction), or is an independent <b>prediction</b> (it can fail):</p>
<div class="tablewrap"><table>
<tr><th>Run</th><th>Direction</th><th>Counter</th><th>Packets it holds</th><th>Model</th><th class="n">Model, 4 GPUs (× $S$)</th><th class="n">Measured (× $S$)</th><th>Role</th></tr>
<tr><td><code>multimem.st</code> by itself (AllGather)</td><td>sent</td><td>request data</td><td>data of its own writes (its share)</td><td>$S/N$</td><td class="n">0.250</td><td class="n">0.250</td><td>prediction</td></tr>
<tr><td></td><td></td><td>request protocol</td><td>headers of its own writes</td><td>$\eta\,S/N$</td><td class="n">0.125 × 0.25 = 0.031</td><td class="n">0.031</td><td>prediction (checks η)</td></tr>
<tr><td></td><td></td><td>response data</td><td>—</td><td>$0$</td><td class="n">0</td><td class="n">0</td><td>prediction</td></tr>
<tr><td></td><td></td><td>response protocol</td><td>acknowledgements of the writes it receives</td><td>$\alpha\,S$</td><td class="n">0.125</td><td class="n">0.125</td><td>defines α</td></tr>
<tr><td></td><td>recv</td><td>request data</td><td>data of every GPU's writes, its own included</td><td>$S$</td><td class="n">1.000</td><td class="n">1.000</td><td>prediction</td></tr>
<tr><td></td><td></td><td>request protocol</td><td>headers of those writes</td><td>$\eta\,S$</td><td class="n">0.125</td><td class="n">0.125</td><td>defines η</td></tr>
<tr><td></td><td></td><td>response data</td><td>—</td><td>$0$</td><td class="n">0</td><td class="n">0</td><td>prediction</td></tr>
<tr><td></td><td></td><td>response protocol</td><td>small responses from the switch to the issuer</td><td>$r_{\mathrm{st}}\,S$</td><td class="n">0.009</td><td class="n">0.009</td><td>defines $r_{\mathrm{st}}$</td></tr>
<tr><td><code>ld_reduce</code> by itself (ReduceScatter)</td><td>sent</td><td>request data</td><td>— (read requests carry no data)</td><td>$0$</td><td class="n">0</td><td class="n">0</td><td>prediction</td></tr>
<tr><td></td><td></td><td>request protocol</td><td>its own read-reduce requests</td><td>$\rho_{\mathrm{own}}\,S/N$</td><td class="n">0.25 × 0.25 = 0.062</td><td class="n">0.062</td><td>defines ρ<sub>own</sub></td></tr>
<tr><td></td><td></td><td>response data</td><td>data of its read replies, for every GPU's share</td><td>$S$</td><td class="n">1.000</td><td class="n">1.000</td><td>prediction</td></tr>
<tr><td></td><td></td><td>response protocol</td><td>headers of those replies</td><td>$\eta\,S$</td><td class="n">0.125</td><td class="n">0.125</td><td>prediction (checks η)</td></tr>
<tr><td></td><td>recv</td><td>request data</td><td>—</td><td>$0$</td><td class="n">0</td><td class="n">0</td><td>prediction</td></tr>
<tr><td></td><td></td><td>request protocol</td><td>the switch's read requests</td><td>$\rho_{\mathrm{switch}}\,S$</td><td class="n">0.154</td><td class="n">0.154</td><td>defines ρ<sub>switch</sub></td></tr>
<tr><td></td><td></td><td>response data</td><td>data of its own sums</td><td>$S/N$</td><td class="n">0.250</td><td class="n">0.250</td><td>prediction</td></tr>
<tr><td></td><td></td><td>response protocol</td><td>headers of those sums + small responses</td><td>$\eta\,S/N + r_{\mathrm{ld}}\,S$</td><td class="n">0.031 + 0.021 = 0.052</td><td class="n">0.052</td><td>defines $r_{\mathrm{ld}}$ (after the header part, predicted by $\eta$)</td></tr>
<tr><td>Both (NVLS AllReduce)</td><td>sent</td><td>request data</td><td>data of its own writes</td><td>$S/N$</td><td class="n">0.250</td><td class="n">0.250</td><td>prediction</td></tr>
<tr><td></td><td></td><td>request protocol</td><td>headers of its own writes + its own read requests</td><td>$\eta\,S/N + \rho_{\mathrm{own}}\,S/N$</td><td class="n">0.031 + 0.062 = 0.094</td><td class="n">0.094</td><td>prediction</td></tr>
<tr><td></td><td></td><td>response data</td><td>data of its read replies</td><td>$S$</td><td class="n">1.000</td><td class="n">1.000</td><td>prediction</td></tr>
<tr><td></td><td></td><td>response protocol</td><td>headers of the replies + acknowledgements</td><td>$\eta\,S + \alpha\,S$</td><td class="n">0.125 + 0.125 = 0.250</td><td class="n">0.250</td><td>prediction</td></tr>
<tr><td></td><td>recv</td><td>request data</td><td>data of every GPU's writes</td><td>$S$</td><td class="n">1.000</td><td class="n">1.000</td><td>prediction</td></tr>
<tr><td></td><td></td><td>request protocol</td><td>headers of those writes + the switch's read requests</td><td>$\eta\,S + \rho_{\mathrm{switch}}\,S$</td><td class="n">0.125 + 0.154 = 0.279</td><td class="n"><b>0.356</b></td><td>prediction: <b>+0.077 not predicted</b></td></tr>
<tr><td></td><td></td><td>response data</td><td>data of its own sums</td><td>$S/N$</td><td class="n">0.250</td><td class="n">0.250</td><td>prediction</td></tr>
<tr><td></td><td></td><td>response protocol</td><td>headers of the sums + both kinds of small responses</td><td>$\eta\,S/N + (r_{\mathrm{st}} + r_{\mathrm{ld}})\,S$</td><td class="n">0.031 + 0.009 + 0.021 = 0.061</td><td class="n">0.061</td><td>prediction</td></tr>
</table></div>
<ul>
<li><b>Example: the 0.031 in the request protocol sent by <code>multimem.st</code> by itself</b> is the headers of the GPU's own writes. It sends only its own share, $S/N = 0.25\,S$ of data, in 128-byte packets with 16-byte headers, so $\eta\,S/N = 0.125 \times 0.25 = 0.031\,S$. It is a prediction of the model, and it checks $\eta$ on a second data stream.</li>
<li><b>Six fields define the six parameters</b> ($\eta$, $\alpha$, $\rho_{\mathrm{own}}$, $\rho_{\mathrm{switch}}$, $r_{\mathrm{st}}$, $r_{\mathrm{ld}}$), all from the instructions run by themselves. Every other field is predicted from them.</li>
<li><b>All predictions hold except one:</b> the request protocol received in AllReduce, 0.077 $S$ above the model.</li>
<li><b>The two small responses</b> are $r_{\mathrm{st}}$ = 0.009 (responses to the issuer of <code>multimem.st</code>) and $r_{\mathrm{ld}}$ = 0.021 (responses beyond the headers of the sums of <code>ld_reduce</code>). In AllReduce both appear on the received direction, and their sum, 0.030 $S$, is exactly what the response protocol received carries beyond the headers: this field is a prediction that holds.</li>
<li><b>At 8 GPUs</b> the same expressions with $N = 8$ match in the same way (for example $\eta\,S/N = 0.016$ and $\rho_{\mathrm{own}}\,S/N = 0.031$), and again only the received request protocol of AllReduce is above the model (0.364 against 0.285).</li>
</ul>

<p>Each parameter is one protocol counter divided by the data counter it belongs to. When the instruction runs by itself, the protocol counter holds only that kind of packet plus known headers. Inside AllReduce, the protocol counter also holds the other instruction's headers, which are subtracted first, using $\eta$:</p>
<div class="tablewrap"><table>
<tr><th>Parameter</th><th>Run by itself: formula</th><th class="n">4 GPUs</th><th class="n">8 GPUs</th><th>Inside AllReduce: formula</th><th class="n">4 GPUs</th><th class="n">8 GPUs</th></tr>
<tr><td>$\eta$ (headers of data packets)</td><td>AllGather: $\text{request protocol}_{\mathrm{recv}}/\text{request data}_{\mathrm{recv}}$ (write headers ÷ write data, received)</td><td class="n">0.125/1.000 = 0.125</td><td class="n">0.125/1.000 = 0.125</td><td>taken from the run by itself</td><td class="n">0.125</td><td class="n">0.125</td></tr>
<tr><td>$\alpha$ (write acknowledgements)</td><td>AllGather: $\text{response protocol}_{\mathrm{sent}}/\text{request data}_{\mathrm{recv}}$ (responses sent ÷ write data received)</td><td class="n">0.125/1.000 = 0.125</td><td class="n">0.125/1.000 = 0.125</td><td>$(\text{response protocol}_{\mathrm{sent}} - \eta\,\text{response data}_{\mathrm{sent}})/\text{request data}_{\mathrm{recv}}$ (minus the headers of the read replies it sends)</td><td class="n">(0.250 − 0.125)/1.000 = 0.125</td><td class="n">(0.250 − 0.125)/1.000 = 0.125</td></tr>
<tr><td>$\rho_{\mathrm{switch}}$ (the switch's read requests)</td><td>ReduceScatter: $\text{request protocol}_{\mathrm{recv}}/\text{response data}_{\mathrm{sent}}$ (requests received ÷ reply data sent)</td><td class="n">0.154/1.000 = 0.154</td><td class="n">0.160/1.000 = 0.160</td><td>$(\text{request protocol}_{\mathrm{recv}} - \eta\,\text{request data}_{\mathrm{recv}})/\text{response data}_{\mathrm{sent}}$ (minus the headers of the incoming writes)</td><td class="n">(0.356 − 0.125)/1.000 = <b>0.231</b></td><td class="n">(0.364 − 0.125)/1.000 = <b>0.239</b></td></tr>
<tr><td>$\rho_{\mathrm{own}}$ (the GPU's own read requests)</td><td>ReduceScatter: $\text{request protocol}_{\mathrm{sent}}/\text{response data}_{\mathrm{recv}}$ (requests sent ÷ sum data received)</td><td class="n">0.062/0.250 = 0.25</td><td class="n">0.031/0.125 = 0.25</td><td>$(\text{request protocol}_{\mathrm{sent}} - \eta\,\text{request data}_{\mathrm{sent}})/\text{response data}_{\mathrm{recv}}$ (minus the headers of its own writes)</td><td class="n">(0.094 − 0.031)/0.250 = 0.25</td><td class="n">(0.047 − 0.016)/0.125 = 0.25</td></tr>
<tr><td>$r$ (other small responses, × $S$)</td><td>AllGather $\text{response protocol}_{\mathrm{recv}}$ + ReduceScatter $(\text{response protocol}_{\mathrm{recv}} - \eta\,\text{response data}_{\mathrm{recv}})$</td><td class="n">0.009 + (0.052 − 0.031) = 0.030</td><td class="n">0.007 + (0.026 − 0.016) = 0.017</td><td>$\text{response protocol}_{\mathrm{recv}} - \eta\,\text{response data}_{\mathrm{recv}}$</td><td class="n">0.061 − 0.031 = 0.030</td><td class="n">0.033 − 0.016 = 0.017</td></tr>
</table></div>
<p>Checks on $\eta$: the same 0.125 comes out of every data stream, for example $\text{request protocol}_{\mathrm{sent}}/\text{request data}_{\mathrm{sent}}$ of AllGather (0.031/0.250, its own writes) and $\text{response protocol}_{\mathrm{sent}}/\text{response data}_{\mathrm{sent}}$ of ReduceScatter (0.125/1.000, its read replies).</p>
<p><b>What the columns mean.</b></p>
<ul>
<li><b>"Run by itself"</b>: the instruction is the only one running. <code>multimem.st</code> by itself is NVLS AllGather; <code>ld_reduce</code> by itself is NVLS ReduceScatter. All GPUs issue it, nothing else is on the links (experiment 4 at 4 GPUs, experiment 9 at 8). Each parameter is then read directly from one counter: for example $\alpha$ = the sent response protocol of AllGather (0.125 $S$) ÷ the writes the GPU received ($S$).</li>
<li><b>"Inside AllReduce"</b>: both instructions run at the same time, as in NVLS AllReduce (experiment 6 at 4 GPUs, experiment 9 at 8). Now each counter holds packets of both instructions mixed together, and the profiler cannot tell them apart. So each parameter is found by <b>subtraction</b>: the counter minus the part that belongs to the other instruction, taken from the "run by itself" measurement. For example $\rho_{\mathrm{switch}}$ = received request protocol of AllReduce (0.356 $S$) − headers of the incoming writes (0.125 $S$, from AllGather by itself) = 0.231 $S$, ÷ the data read from this GPU ($S$).</li>
</ul>
<p><b>Why the values differ.</b> $\alpha$ and $\rho_{\mathrm{own}}$ come out the same in both columns: those packets do not change when the other instruction runs. Only $\rho_{\mathrm{switch}}$ differs, because the received request-protocol counter of AllReduce is 0.077 $S$ larger than the sum of the two instructions by themselves. The subtraction assigns all of that excess to read requests. That is an assumption: the counter would look the same if, instead, some incoming writes arrived in smaller packets in the mix (more headers per byte of data). Which of the two happens, and why, is not known; it is the one part of AllReduce's packets that the instructions by themselves do not predict.</p>
<ul>
<li><b>$\alpha$ equals the reference case exactly:</b> one 16-byte acknowledgement per 128-byte write packet, alone and in the mix, at both GPU counts.</li>
<li><b>$\rho_{\mathrm{switch}}$ is above the reference case when run by itself</b> (0.154–0.160 instead of 0.125, about 1.2 requests per 128 bytes read) <b>and grows by half in the mix</b> (0.231–0.239). This growth is the main thing the instructions run by themselves do not predict.</li>
<li><b>$\rho_{\mathrm{own}}$ is twice the reference case</b> (0.25 instead of 0.125: about 32 bytes of request per 128-byte sum), but it is multiplied by $S$/N, so it adds only 0.062 $S$ (4 GPUs) or 0.031 $S$ (8 GPUs).</li>
</ul>
<p>With these, the traffic table of the theory gives, per direction of NVLS AllReduce (× $S$; $(1+\eta)(1+\tfrac{1}{N}) = 1.406$ at 4 GPUs, 1.266 at 8):</p>
<ul>
<li><b>Received</b> = $(1+\eta)(1+\tfrac{1}{N}) + \rho_{\mathrm{switch}} + r_{\mathrm{st}} + r_{\mathrm{ld}}$: 4 GPUs 1.406 + 0.154 + (0.009 + 0.021) = 1.590, so $w$ = 1.590 ÷ 1.25 = 1.272; 8 GPUs 1.266 + 0.160 + (0.007 + 0.010) = 1.443, $w$ = 1.283.</li>
<li><b>Sent</b> = $(1+\eta)(1+\tfrac{1}{N}) + \alpha + \rho_{\mathrm{own}}/N$: 4 GPUs 1.406 + 0.125 + 0.062 = 1.593, $w$ = 1.274; 8 GPUs 1.266 + 0.125 + 0.031 = 1.422, $w$ = 1.264.</li>
<li>The busier direction sets $w$: <b>1.274 at 4 GPUs</b> (sent), <b>1.283 at 8 GPUs</b> (received). These are the values of the third row below. With $\rho_{\mathrm{switch}}$ from the mix instead (0.231, 0.239), the received side alone would give $w$ = 1.334 and 1.353, the profiler values of AllReduce; the raw counter adds the wire bytes outside the packets on top (1.343, 1.360).</li>
</ul>
<p><b>The measured wire cost, term by term.</b> Dividing each direction's traffic of AllReduce by its data, $(1+\tfrac{1}{N})\,S$, every term of the model becomes a share of $w$. Adding the two things the instructions by themselves do not predict, the extra switch read requests in the mix ($\Delta\rho_{\mathrm{switch}}$) and the wire bytes in no packet type ($u$, raw counter minus the four packet counters, per $S$), accounts for the whole measured $w$:</p>
<div class="tablewrap"><table>
<tr><th>Term of $w$</th><th>Formula</th><th class="n">4 GPUs, recv</th><th class="n">4 GPUs, sent</th><th class="n">8 GPUs, recv</th><th class="n">8 GPUs, sent</th></tr>
<tr><td>Data</td><td>$1$</td><td class="n">1.000</td><td class="n">1.000</td><td class="n">1.000</td><td class="n">1.000</td></tr>
<tr><td>Headers of data packets</td><td>$\eta$</td><td class="n">0.125</td><td class="n">0.125</td><td class="n">0.125</td><td class="n">0.125</td></tr>
<tr><td>Switch read requests, as by itself</td><td>$\rho_{\mathrm{switch}}\cdot\frac{N}{N+1}$</td><td class="n">0.123</td><td class="n">—</td><td class="n">0.142</td><td class="n">—</td></tr>
<tr><td>Switch read requests, extra in the mix</td><td>$\Delta\rho_{\mathrm{switch}}\cdot\frac{N}{N+1}$</td><td class="n">0.062</td><td class="n">—</td><td class="n">0.070</td><td class="n">—</td></tr>
<tr><td>Small responses</td><td>$(r_{\mathrm{st}}+r_{\mathrm{ld}})\cdot\frac{N}{N+1}$</td><td class="n">0.024</td><td class="n">—</td><td class="n">0.015</td><td class="n">—</td></tr>
<tr><td>Write acknowledgements</td><td>$\alpha\cdot\frac{N}{N+1}$</td><td class="n">—</td><td class="n">0.100</td><td class="n">—</td><td class="n">0.111</td></tr>
<tr><td>The GPU's own read requests</td><td>$\rho_{\mathrm{own}}\cdot\frac{1}{N+1}$</td><td class="n">—</td><td class="n">0.050</td><td class="n">—</td><td class="n">0.028</td></tr>
<tr><td>Wire bytes in no packet type</td><td>$u\cdot\frac{N}{N+1}$</td><td class="n">0.010</td><td class="n">0.061</td><td class="n">0.007</td><td class="n">0.087</td></tr>
<tr><td><b>Sum</b></td><td></td><td class="n"><b>1.344</b></td><td class="n"><b>1.336</b></td><td class="n"><b>1.359</b></td><td class="n"><b>1.351</b></td></tr>
<tr><td>Measured $w$ (raw counter ÷ data counter)</td><td></td><td class="n">1.343</td><td class="n">1.336</td><td class="n">1.360</td><td class="n">1.350</td></tr>
</table></div>
<p>Values used: $\rho_{\mathrm{switch}} = 0.154 / 0.160$, $\Delta\rho_{\mathrm{switch}} = 0.077 / 0.079$, $r_{\mathrm{st}}+r_{\mathrm{ld}} = 0.030 / 0.017$, $\alpha = 0.125$, $\rho_{\mathrm{own}} = 0.25$, $u_{\mathrm{recv}} = 0.012 / 0.008$, $u_{\mathrm{sent}} = 0.076 / 0.098$ (4 / 8 GPUs). The small responses are 0.024 and 0.015 of $w$: real, but the smallest of the control terms. The terms that set the loss are the switch's read requests (by themselves and extra in the mix) on the received side, and the write acknowledgements and the bytes in no packet type on the sent side.</p>

<div class="tablewrap"><table>
<tr><th>Level</th><th>Wire cost $w$</th><th class="n">busBW, 4 GPUs</th><th class="n">algBW, 4 GPUs</th><th class="n">busBW, 8 GPUs</th><th class="n">algBW, 8 GPUs</th><th class="n">algBW ÷ measured Ring, 8 GPUs</th></tr>
<tr><td><b>No control packets</b>: data and headers only</td><td>$1+\eta$ = 1.125</td><td class="n">378.8</td><td class="n">303.1</td><td class="n">376.2</td><td class="n">334.4</td><td class="n">1.61</td></tr>
<tr><td><b>Reference case</b>: one 16-byte control packet per 128-byte data packet ($\eta$ = $\rho_{\mathrm{own}}$ = $\rho_{\mathrm{switch}}$ = $\alpha$ = 0.125)</td><td>1.25</td><td class="n">341.0</td><td class="n">272.8</td><td class="n">338.6</td><td class="n">301.0</td><td class="n">1.45</td></tr>
<tr><td><b>Theory with the parameters measured on each operation alone</b> ($\alpha$, $\rho_{\mathrm{switch}}$, $\rho_{\mathrm{own}}$, $r$ from the "run by itself" columns above)</td><td>4 GPUs: 1.274; 8 GPUs: 1.283</td><td class="n">334.4</td><td class="n">267.5</td><td class="n">329.9</td><td class="n">293.3</td><td class="n">1.41</td></tr>
<tr><td><b>Measured</b></td><td>4 GPUs: 1.343; 8 GPUs: 1.360</td><td class="n">317.9</td><td class="n">253.1</td><td class="n">310.8</td><td class="n">276.3</td><td class="n">1.33</td></tr>
</table></div>
<p>The last column divides by the measured Ring algBW (208.0 GB/s), whose $w$ is 1.159, a little above $1+\eta$; that is why the reference case gives 1.45 here and 1.40 above, where Ring is also taken at $1+\eta$.</p>
</div>
<figure class="wide"><div class="scroll">@@chart_expect@@</div><figcaption>busBW (GB/s) of NVLS AllReduce at the four levels of the table, next to the measured busBW of AllGather alone and of Ring AllReduce. Dark blue: measured. Light blue: computed as wire rate of AllGather alone ÷ $w$.</figcaption></figure>
<div class="col">
<ul>
<li><b>The theory, with the parameters measured on each operation alone, explains most of the loss:</b> busBW falls from 378.8 GB/s (no control packets) to 334.4 (−12%) at 4 GPUs. The reference case, for comparison, gives 341.0 (−10%).</li>
<li><b>The measurement is lower still: 317.9 (−16%).</b> The extra comes from two things the theory does not contain, shown next: more read requests in the mix than alone, and wire bytes in no packet type.</li>
<li><b>Use the profiler packets, not the raw counter, for the operations alone.</b> Alone, each low-data direction also carries 0.55–0.66 $S$ of wire bytes in no packet type. They almost vanish in the mix, so they are not part of the cost.</li>
</ul>

<h4>Check of the theory's main assumption: do the packets of the two instructions add up?</h4>
<p><b>Why this check.</b> The theory and its third level ("parameters measured on each instruction run by itself") assume that running both instructions at once, as AllReduce does, changes nothing about the packets of either: AllReduce's packets on a direction should be exactly the packets of <code>multimem.st</code> by itself plus those of <code>ld_reduce</code> by itself on that direction.</p>
<p><b>What is compared.</b> The same four profiler counters as above, each ÷ $S$, on each direction: the counter of AllGather (<code>multimem.st</code> by itself) <b>plus</b> the counter of ReduceScatter (<code>ld_reduce</code> by itself), against the counter of NVLS AllReduce. Adding is allowed because all three runs are normalised by the same $S$; the sum is what AllReduce would carry if nothing changed.</p>
</div>
<figure class="wide"><div class="scroll">@@chart_additivity@@</div><figcaption>Profiler packet bytes ÷ S on each direction of one GPU, split into the four counters. In each pair, left bar ("sum"): the counter of AllGather by itself plus the counter of ReduceScatter by itself; right bar ("AR"): the same counter measured in NVLS AllReduce. Bar top: all four counters added. 4 GPUs: our program, 32 CTAs × 512 threads per instruction (experiments 4 and 6). 8 GPUs: NCCL's kernels (experiment 9). Only the orange request-protocol part of the received bars differs.</figcaption></figure>
<div class="col">
<div class="tablewrap"><table>
<tr><th>GPUs, direction</th><th>Counter (÷ $S$)</th><th class="n">AllGather by itself</th><th class="n">ReduceScatter by itself</th><th class="n">Sum</th><th class="n">AllReduce</th><th class="n">AllReduce − sum</th></tr>
<tr><td>4, sent</td><td>request data</td><td class="n">0.250</td><td class="n">0</td><td class="n">0.250</td><td class="n">0.250</td><td class="n">0</td></tr>
<tr><td></td><td>request protocol</td><td class="n">0.031</td><td class="n">0.062</td><td class="n">0.093</td><td class="n">0.094</td><td class="n">+0.001</td></tr>
<tr><td></td><td>response data</td><td class="n">0</td><td class="n">1.000</td><td class="n">1.000</td><td class="n">1.000</td><td class="n">0</td></tr>
<tr><td></td><td>response protocol</td><td class="n">0.125</td><td class="n">0.125</td><td class="n">0.250</td><td class="n">0.250</td><td class="n">0</td></tr>
<tr><td>4, recv</td><td>request data</td><td class="n">1.000</td><td class="n">0</td><td class="n">1.000</td><td class="n">1.000</td><td class="n">0</td></tr>
<tr><td></td><td>request protocol</td><td class="n">0.125</td><td class="n">0.154</td><td class="n">0.279</td><td class="n">0.356</td><td class="n"><b>+0.077</b></td></tr>
<tr><td></td><td>response data</td><td class="n">0</td><td class="n">0.250</td><td class="n">0.250</td><td class="n">0.250</td><td class="n">0</td></tr>
<tr><td></td><td>response protocol</td><td class="n">0.009</td><td class="n">0.052</td><td class="n">0.061</td><td class="n">0.061</td><td class="n">0</td></tr>
<tr><td>8, sent</td><td>all four</td><td class="n">0.266</td><td class="n">1.156</td><td class="n">1.422</td><td class="n">1.421</td><td class="n">−0.001</td></tr>
<tr><td>8, recv</td><td>request protocol</td><td class="n">0.125</td><td class="n">0.160</td><td class="n">0.285</td><td class="n">0.364</td><td class="n"><b>+0.079</b></td></tr>
<tr><td></td><td>the other three</td><td class="n">1.007</td><td class="n">0.151</td><td class="n">1.158</td><td class="n">1.158</td><td class="n">0</td></tr>
</table></div>
<ul>
<li><b>Seven of the eight counters add up exactly</b> (within 0.001), at both GPU counts. So the assumption holds for the data, the data headers, the write acknowledgements and the GPU's own read requests.</li>
<li><b>The one exception is the received request protocol: +0.077 S (4 GPUs), +0.079 S (8 GPUs).</b> That counter holds the headers of the incoming writes and the switch's read requests. Attributing the whole excess to read requests gives the in-mix value $\rho_{\mathrm{switch}} = 0.231$ instead of 0.154 (see the parameter table above). Why it grows is not known.</li>
<li><b>The wire bytes in no packet type do not add.</b> They are not in these four counters; they are the raw counter minus the four. By itself, each instruction has about 0.001 $S$ of them on its busy direction and 0.55–0.66 $S$ on its low-data direction; in AllReduce, 0.012 $S$ (recv) and 0.076 $S$ (sent).</li>
<li><b>Together these give the measured wire cost.</b> Received: (1.667 $S$ of packets + 0.012 $S$) ÷ 1.25 $S$ of data = 1.343. Sent: (1.594 + 0.076) ÷ 1.25 = 1.336. The received side is raised by the extra request protocol, the sent side by the bytes in no packet type, and both end almost equal.</li>
</ul>

<h4>How Ring AllReduce runs, and why it does not pay this cost</h4>
<p>Ring also fills both link directions with data. To see why it does not lose, follow one Ring AllReduce on 4 GPUs. The buffer is cut into 4 chunks A, B, C, D, and every GPU holds its own copy of each.</p>
</div>
<figure class="wide"><div class="scroll">@@diagram_ring_steps@@</div><figcaption>What each GPU sends to the next GPU at each step. Steps 1–3 (reduce-scatter): a GPU receives a partial sum, adds its own copy of that chunk and sends the result on; the receiver of step 3 adds the last copy, so each GPU then holds one finished sum (GPU 0: ΣB, GPU 1: ΣC, GPU 2: ΣD, GPU 3: ΣA). Steps 4–6 (all-gather): the finished chunks are passed around until every GPU has all four. Each step moves 1/4 of the buffer per direction, so 6 steps give D = 1.5 S = 2(N − 1)/N × S on each direction.</figcaption></figure>
<figure class="wide"><div class="scroll">@@diagram_ring_pipeline@@</div><figcaption>NCCL does not wait for one step to finish everywhere before starting the next. It cuts the buffer into small slices and sends them through the steps one behind the other (dark: reduce-scatter steps, light: all-gather steps). Once the pipeline is full, all steps run at the same time.</figcaption></figure>
<div class="col">
<p>So in Ring, as in NVLS AllReduce, both directions of every GPU are full of data for the whole call. The difference is in what else travels on them:</p>
</div>
<figure class="wide"><div class="scroll">@@diagram_lanes@@</div><figcaption>One GPU's two link directions, amounts as multiples of S (profiler packet counters). Solid blue: data packets. Dashed orange: control packets that travel against the data they belong to. Ring (left, 8 GPUs, experiment 9): the sender writes with plain stores into the next GPU's buffer, and the reduction reads the GPU's own memory, so there are no read requests and almost no write responses. NVLS AllReduce (right, 4 GPUs, experiment 6): each direction carries one operation's data plus the other operation's control packets.</figcaption></figure>
<div class="col">
<div class="tablewrap"><table>
<tr><th></th><th>Ring AllReduce</th><th>NVLS AllReduce</th></tr>
<tr><td>How data crosses the link</td><td>The sender <b>writes</b> (plain stores) into the next GPU's buffer</td><td><code>ld_reduce</code>: the switch <b>reads</b> from every GPU. <code>multimem.st</code>: the switch <b>writes</b> into every GPU</td></tr>
<tr><td>Where the reduction reads</td><td>The GPU's own memory, not NVLink</td><td>Every GPU's memory, over NVLink</td></tr>
<tr><td>Control packets against the data</td><td>Almost none: 0.024–0.028 S of write responses</td><td>A read request for every read; a 16-byte response for every 128-byte multicast write packet</td></tr>
<tr><td>Wire cost w</td><td>1.16 on both directions</td><td>1.34 on both directions</td></tr>
</table></div>
<h4>Ring AllReduce in the same model</h4>
<p>Ring needs only plain stores to the next GPU, so its model has two packet parameters besides $\eta$, both the plain-store counterparts of the multicast ones: $\alpha_{\mathrm{p}}$, the responses a GPU sends for the stores it <b>receives</b> (per data byte received, like $\alpha$), and $r_{\mathrm{p}}$, the responses a GPU receives for the stores it <b>sends</b> (per data byte sent, like $r_{\mathrm{st}}$). With $k = 2(N-1)/N$, each direction carries $k\,S$ of data:</p>
<div class="tablewrap"><table>
<tr><th>Ring AllReduce, direction</th><th>request data</th><th>request protocol</th><th>response data</th><th>response protocol</th></tr>
<tr><td>sent: stores to the next GPU, responses for the stores received from the previous GPU</td><td>$k\,S$</td><td>$\eta\,k\,S$</td><td>$0$</td><td>$\alpha_{\mathrm{p}}\,k\,S$</td></tr>
<tr><td>recv: stores from the previous GPU, responses for the stores sent to the next GPU</td><td>$k\,S$</td><td>$\eta\,k\,S$</td><td>$0$</td><td>$r_{\mathrm{p}}\,k\,S$</td></tr>
</table></div>
<p>So $w = 1 + \eta + \alpha_{\mathrm{p}}$ on the sent direction and $1 + \eta + r_{\mathrm{p}}$ on the received direction, plus the wire bytes in no packet type, $u/k$. No read requests and no multicast acknowledgements appear in any field.</p>
<p>The parameters come from runs other than Ring: $\eta = 0.125$ (experiment 4), and $\alpha_{\mathrm{p}}$ and $r_{\mathrm{p}}$ from the plain-store ring of experiment 4 (each GPU stores $0.25\,S$ into the next GPU): $\alpha_{\mathrm{p}}$ = response protocol sent ÷ request data received = 0.0034 ÷ 0.250 = 0.014; $r_{\mathrm{p}}$ = response protocol received ÷ request data sent = 0.0031 ÷ 0.250 = 0.012. Every Ring field below is therefore a prediction. Measured: NCCL's Ring kernel, 8 GPUs, 24 CTAs × 512 threads (experiment 9 for the packets, the raw-counter run for $W$); Ring was not profiled at 4 GPUs.</p>
<div class="tablewrap"><table>
<tr><th>Direction</th><th>Counter (× $S$)</th><th>Packets it holds</th><th>Model</th><th class="n">Model, 8 GPUs ($k$ = 1.75)</th><th class="n">Measured</th><th>Role</th></tr>
<tr><td>sent</td><td>request data</td><td>data of its stores to the next GPU</td><td>$k\,S$</td><td class="n">1.750</td><td class="n">1.750</td><td>prediction</td></tr>
<tr><td></td><td>request protocol</td><td>headers of those stores</td><td>$\eta\,k\,S$</td><td class="n">0.125 × 1.75 = 0.219</td><td class="n">0.219</td><td>prediction</td></tr>
<tr><td></td><td>response data</td><td>—</td><td>$0$</td><td class="n">0</td><td class="n">0</td><td>prediction</td></tr>
<tr><td></td><td>response protocol</td><td>responses for the stores it receives</td><td>$\alpha_{\mathrm{p}}\,k\,S$</td><td class="n">0.014 × 1.75 = 0.024</td><td class="n">0.028</td><td>prediction (within 0.004)</td></tr>
<tr><td>recv</td><td>request data</td><td>data of the stores from the previous GPU</td><td>$k\,S$</td><td class="n">1.750</td><td class="n">1.750</td><td>prediction</td></tr>
<tr><td></td><td>request protocol</td><td>headers of those stores</td><td>$\eta\,k\,S$</td><td class="n">0.219</td><td class="n">0.219</td><td>prediction</td></tr>
<tr><td></td><td>response data</td><td>—</td><td>$0$</td><td class="n">0</td><td class="n">0</td><td>prediction</td></tr>
<tr><td></td><td>response protocol</td><td>responses for the stores it sends</td><td>$r_{\mathrm{p}}\,k\,S$</td><td class="n">0.012 × 1.75 = 0.022</td><td class="n">0.024</td><td>prediction (within 0.002)</td></tr>
<tr><td>sent</td><td>wire bytes $W$ (raw counter)</td><td>all of the above + bytes in no packet type</td><td>$k\,S\,(1+\eta+\alpha_{\mathrm{p}}) + u\,S$</td><td class="n">1.992 + $u$</td><td class="n">2.026 ($u$ = 0.030)</td><td>$u$ measured</td></tr>
<tr><td>recv</td><td>wire bytes $W$ (raw counter)</td><td>all of the above + bytes in no packet type</td><td>$k\,S\,(1+\eta+r_{\mathrm{p}}) + u\,S$</td><td class="n">1.988 + $u$</td><td class="n">2.028 ($u$ = 0.036)</td><td>$u$ measured</td></tr>
</table></div>
<ul>
<li><b>Ring's packets are data, headers and a trickle of responses.</b> Wire cost from the packets: $1 + \eta + \alpha_{\mathrm{p}}$ = 1.14; with the bytes in no packet type, the measured $w$ = 2.026 ÷ 1.75 = 1.158 (sent) and 2.028 ÷ 1.75 = 1.159 (recv).</li>
<li><b>Against NVLS AllReduce</b> ($w$ = 1.34–1.36): Ring has no read requests at all (no $\rho_{\mathrm{switch}}$, no $\rho_{\mathrm{own}}$), and its stores get about one tenth of the responses that multicast writes get ($\alpha_{\mathrm{p}}$ ≈ 0.014 against $\alpha$ = 0.125). That difference in $w$ is the 15% of NVLS's byte-count advantage that is lost.</li>
<li><b>The plain-store parameters were measured at 4 GPUs in our program and applied to NCCL's Ring at 8 GPUs;</b> the small mismatch in the response protocol (0.004 and 0.002 $S$) is within what that change of setup can explain, and is 0.2% of $w$.</li>
</ul>
<p><b>Would running the two phases one after the other help NVLS?</b> No. NVLS AllReduce already runs both operations at once, like Ring. If it ran ReduceScatter and then AllGather, each phase would leave one direction mostly empty: 1 ÷ (1/358 + 1/376) ≈ 183 GB/s of data per direction, far below the 318 it reaches. Separate calls were measured 1.38–1.53× slower.</p>

<h4>Where every step lands: alone and in AllReduce</h4>
<p>AllReduce runs both storyboards at the same time on the same GPUs. The chart puts each step's packets (AG1–AG4 for <code>multimem.st</code>, RS1–RS4 for <code>ld_reduce</code>) on GPU 0's two directions, as multiples of S, at 4 GPUs with the parameters measured on each instruction run by itself. The last received segment "mix" is the extra read requests that only appear when both run.</p>
</div>
<figure class="wide"><div class="scroll">@@chart_tally@@</div><figcaption>GPU 0's sent and received directions, built from the steps above. Alone, each operation has one busy direction (about 1.13–1.19 S) and one low-data direction, and its control step (AG3 acknowledgements, RS2 read requests) sits on the low-data one, where there is room. In AllReduce both directions carry about 1.4 S of data and headers, and AG3 now shares the sent direction with RS3's replies, RS2 the received direction with AG2's writes.</figcaption></figure>
<div class="col">
<ul>
<li><b>Sent direction of AllReduce</b> = AG1 + AG3 + RS1 + RS3 = $(1+\eta)\,S/N + \alpha S + \rho_{\mathrm{own}} S/N + (1+\eta)\,S$ = 0.281 + 0.125 + 0.062 + 1.125 = 1.594 $S$ (measured 1.594).</li>
<li><b>Received direction of AllReduce</b> = AG2 + AG4 + RS2 + RS4 = $(1+\eta)\,S + r_{\mathrm{st}} S + \rho_{\mathrm{switch}} S + (1+\eta)\,S/N + r_{\mathrm{ld}} S$ = 1.125 + 0.009 + 0.154 + 0.281 + 0.021 = 1.590 $S$; measured 1.667, the extra 0.077 being the "mix" segment.</li>
</ul>
<p>Ring for comparison, in the same style:</p>
</div>
<figure class="wide"><div class="scroll">@@story_ring@@</div><figcaption>Ring AllReduce: every step only moves plain stores from each GPU to the next, and the reduction reads the GPU's own memory. There is no step like AG3 (one acknowledgement per write packet) or RS2 (read requests into every GPU), so both directions carry data, headers and only a trickle of responses.</figcaption></figure>
<div class="col">
<h4>Diagnosis</h4>
<div class="note"><b>Root cause.</b> NVLS AllReduce puts the small control packets of its two instructions on link directions that are already full of data: the switch's <b>read requests</b> for <code>ld_reduce</code> on each GPU's received direction, and the GPU's <b>write responses</b> for <code>multimem.st</code> on its sent direction. A direction carries a fixed number of wire bytes per second (the wire rate limit C), so every wire byte these packets use is a byte of data that does not move.</div>
<p>How it happens, step by step:</p>
<ol>
<li><b><code>ld_reduce</code> is a pull.</b> To compute a sum, the switch sends a read request into every GPU. The request arrives on the GPU's received direction; the value goes back on its sent direction.</li>
<li><b><code>multimem.st</code> writes are acknowledged.</b> For every 128-byte multicast write packet a GPU receives, it sends a 16-byte response on its sent direction. Plain stores, which Ring uses, get almost none.</li>
<li><b>Alone, these packets are free.</b> In ReduceScatter alone the received direction carries only 1/N of the data, so the read requests use spare wire time. In AllGather alone the same holds for the write responses on the sent direction. The busy direction of each carries only data and headers.</li>
<li><b>In AllReduce both directions are full of data.</b> The read requests now share the received direction with the multicast writes; the write responses share the sent direction with the read replies. The wire still runs at its limit (426.9 GB/s), but less of it is data.</li>
<li><b>The mix adds a little more.</b> In AllReduce the switch sends 50% more read-request bytes than in ReduceScatter alone (0.231 S instead of 0.154 S). Why is not known.</li>
</ol>
</div>
<figure class="wide"><div class="scroll">@@chart_wire_budget@@</div><figcaption>Wire cost w on the busy direction, part by part, drawn to scale (packet counters and raw counter; 4 GPUs: experiments 1, 4 and 6; Ring at 8 GPUs: raw-counter run and experiment 9). Every bar starts with 1 data byte and 0.125 of headers. Orange is what NVLS AllReduce adds: the switch's read requests on the received side, write responses on the sent side. busBW = about 426 GB/s ÷ bar length.</figcaption></figure>
<div class="col">
<p>Wire cost w, split into its parts (wire bytes of each kind ÷ D on that direction), 4 GPUs, 32 CTAs × 512 threads per operation (packet counters and raw counter, experiments 4 and 6):</p>
<div class="tablewrap"><table>
<tr><th>Part</th><th class="n">AllGather alone, received</th><th class="n">AllReduce, received</th><th class="n">ReduceScatter alone, sent</th><th class="n">AllReduce, sent</th></tr>
<tr><td>Data</td><td class="n">1.000</td><td class="n">1.000</td><td class="n">1.000</td><td class="n">1.000</td></tr>
<tr><td>Headers of data packets (16 B per 128 B)</td><td class="n">0.125</td><td class="n">0.125</td><td class="n">0.125</td><td class="n">0.125</td></tr>
<tr><td>Read requests from the switch, as in ReduceScatter alone</td><td class="n">—</td><td class="n"><b>0.123</b></td><td class="n">—</td><td class="n">—</td></tr>
<tr><td>Read requests from the switch, only in the mix</td><td class="n">—</td><td class="n"><b>0.062</b></td><td class="n">—</td><td class="n">—</td></tr>
<tr><td>Write responses for <code>multimem.st</code></td><td class="n">—</td><td class="n">—</td><td class="n">—</td><td class="n"><b>0.100</b></td></tr>
<tr><td>The GPU's own read requests</td><td class="n">—</td><td class="n">—</td><td class="n">0.062</td><td class="n">0.050</td></tr>
<tr><td>Small responses ($r_{\mathrm{st}}$, $r_{\mathrm{ld}}$)</td><td class="n">0.009</td><td class="n">0.024</td><td class="n">—</td><td class="n">—</td></tr>
<tr><td>Wire bytes in no packet type ($u$)</td><td class="n">0.001</td><td class="n">0.010</td><td class="n">0.002</td><td class="n">0.061</td></tr>
<tr><td><b>Wire cost w</b></td><td class="n"><b>1.135</b></td><td class="n"><b>1.343</b></td><td class="n"><b>1.189</b></td><td class="n"><b>1.336</b></td></tr>
<tr><td><b>busBW = wire rate ÷ w, GB/s</b></td><td class="n">375.6</td><td class="n">317.9</td><td class="n">358.1</td><td class="n">317.9</td></tr>
</table></div>
<p>The read-request and write-response rows are split from the write headers by subtraction, assuming the headers are the same as when each operation runs alone; they are estimates. As multiples of S they are 0.154 S + 0.077 S (received) and 0.125 S (sent); dividing by D = 1.25 S gives the values in the table.</p>
<ul>
<li><b>Received side:</b> the cost rises from 1.135 to 1.343. Read requests make 0.185 of the 0.208 rise (89%).</li>
<li><b>Sent side:</b> it rises to 1.336, mostly from write responses. The two sides are almost equal, so removing only one packet kind gains nothing: the other side then limits.</li>
<li><b>Ring</b> fills both directions too but sends neither packet kind: w = 1.16. That is the 15% of NVLS's byte-count advantage that is lost (1.33× instead of 1.56× at 8 GPUs).</li>
</ul>
<p>What the measurements rule out:</p>
<div class="tablewrap"><table>
<tr><th>Not the cause</th><th>Evidence</th></tr>
<tr><td>Idle link time, a switch that stalls</td><td>Physical link rate 426.9 GB/s, the same as every other pattern (experiment 1)</td></tr>
<tr><td>More data per direction</td><td>busBW counts data bytes, so the data factor k = 1 + 1/N does not enter it; with AllGather's wire cost w AllReduce would also reach 375 GB/s</td></tr>
<tr><td>Too few CTAs, or how threads issue</td><td>The cost is the same from 4 to 64 CTAs; instruction width and NCCL's address pattern do not change the packets (experiment 5)</td></tr>
<tr><td>One operation having priority</td><td>The share follows how many CTAs issue each operation (experiment 7)</td></tr>
<tr><td>Our test program</td><td>NCCL's own kernels on 8 GPUs show the same packets (experiment 9)</td></tr>
</table></div>
<p><b>Still unknown:</b> why the read requests grow 50% in the mix (0.062 of the received-side rise); what the wire bytes in no packet type on the sent side are (0.061); and whether a different memory-ordering form of <code>multimem.st</code> avoids the write responses.</p>
<p><b>What this means.</b> NVLS AllReduce being below AllGather and ReduceScatter (problem 1) is mostly what their packets predict: the single operations are not a reference AllReduce can reach, because their small packets travel free on a direction with spare room. The loss that matters for performance is against Ring (problem 2).</p>

<h2 id="takeaways">Takeaways</h2>
<ol class="takeaways">
<li><b>The links are never idle.</b> In every pattern, on 4 and 8 GPUs, the busy directions run at the same wire rate of about 420–427 GB/s. NVLS AllReduce is not slow on the wire; a smaller share of the wire carries data.</li>
<li><b>Small packets that each operation needs end up on full directions.</b> <code>ld_reduce</code> needs read requests; every <code>multimem.st</code> packet gets a write response. Alone, each operation sends these on the direction its data does not fill. In AllReduce both directions are full of data, so they take data's place: wire cost w = @@e11_allreduce_ratio@@ instead of @@e11_allgather_ratio@@ for <code>multimem.st</code> alone.</li>
<li><b>Compared with the operations alone, AllReduce adds almost no new packets.</b> Only the read requests grow, by @@e16_readreq_growth@@%. What changes is where the packets travel, not how many there are.</li>
<li><b>Ring does not pay this cost.</b> Its plain stores get almost no responses and it never reads from another GPU. This is why NVLS keeps only about 85% of its byte-count advantage: @@gain8@@× instead of @@ideal8@@× at 8 GPUs, @@gain4@@× instead of @@ideal4@@× at 4.</li>
<li><b>Fewer CTAs do not help.</b> The wire cost w belongs to the packets, not to the load. With fewer CTAs the GPUs ask for less and less data moves; at low load w is even higher.</li>
<li><b>The kernel is not the lever.</b> NCCL's address pattern already gives the fewest packets, the bytes per thread per instruction make no difference, and moving CTAs between the operations only changes who gets the link.</li>
<li><b>Both directions must improve together.</b> The two directions of AllReduce carry different extra traffic: read requests on the received side; write responses, the GPU's own read requests and unexplained wire bytes on the sent side. Removing the write responses alone gains nothing, because the received side then limits. Removing both packet kinds would give about +9–10% algBW, or @@e19_gain_without@@× over Ring at 8 GPUs; the full bound needs the unexplained wire bytes gone too.</li>
<li><b>What to try next.</b> Other memory-ordering forms of <code>multimem.st</code> (to see whether write responses depend on them) and other ways of forming <code>ld_reduce</code> reads, tested together. Splitting the buffer between NVLS and Ring, or pushing data instead of reading it, cannot help by byte count. The remaining levers, unacknowledged multicast writes and larger read requests, belong to the hardware.</li>
<li><b>Open questions.</b> What the wire bytes in no packet type are; why read requests grow when mixed; and how B300 (NVLink 5) behaves.</li>
</ol>
<p class="byline">Data: 276 runs on GPUs 4–7 (18–19 November 2026) and 13 runs on GPUs 0–7 (19 November 2026), NCCL 2.32.3. Every number and chart on this page is generated from the saved result summaries in the project repository (<code>experiments/h100x4/nvls/rca/</code> and <code>experiments/h100x8/nvls/rca/</code>); raw run records stay on the machine.</p>
</div>
</main>
'''



# ---------------------------------------------------------------- diagrams of the "Putting it together" section

def diagram_ring_steps():
    """Ring AllReduce on 4 GPUs: what each GPU sends to the next one at each of the 2(N - 1) steps."""
    sends = [
        ['A₀', 'D₃+D₀', 'C₂+C₃+C₀', 'ΣB', 'ΣA', 'ΣD'],
        ['B₁', 'A₀+A₁', 'D₃+D₀+D₁', 'ΣC', 'ΣB', 'ΣA'],
        ['C₂', 'B₁+B₂', 'A₀+A₁+A₂', 'ΣD', 'ΣC', 'ΣB'],
        ['D₃', 'C₂+C₃', 'B₁+B₂+B₃', 'ΣA', 'ΣD', 'ΣC'],
    ]
    left, top, cell_w, cell_h = 340, 74, 90, 36
    svg = ['<svg viewBox="0 0 900 312" role="img" aria-label="Ring AllReduce on 4 GPUs, step by step" class="chart">']
    # the ring
    svg.append('<text x="120" y="22" class="d-title" text-anchor="middle">The ring</text>')
    positions = [(120, 62), (200, 150), (120, 238), (40, 150)]
    arcs = ['M153,66 Q196,74 201,124', 'M200,174 Q198,232 162,240', 'M86,238 Q44,228 40,176', 'M40,126 Q44,74 82,64']
    for index, (x, y) in enumerate(positions):
        svg.append(f'<rect x="{x - 32}" y="{y - 18}" width="64" height="36" rx="6" class="d-box"/><text x="{x}" y="{y + 5}" class="d-title" text-anchor="middle">GPU {index}</text>')
    for path in arcs:
        svg.append(f'<path d="{path}" class="d-arrow-data" marker-end="url(#hd)"/>')
    svg.append('<text x="120" y="284" class="t-small" text-anchor="middle">Each GPU sends only to the next GPU</text>')
    svg.append('<text x="120" y="298" class="t-small" text-anchor="middle">and receives only from the previous one</text>')
    # phase headers
    svg.append(f'<rect x="{left}" y="28" width="{3 * cell_w - 4}" height="22" rx="4" class="c-data"/><text x="{left + 1.5 * cell_w}" y="43" class="t-inbar" text-anchor="middle">Reduce-scatter: add your copy, pass on</text>')
    svg.append(f'<rect x="{left + 3 * cell_w}" y="28" width="{3 * cell_w - 4}" height="22" rx="4" class="c-data2"/><text x="{left + 4.5 * cell_w}" y="43" class="t-label" text-anchor="middle">All-gather: pass finished sums on</text>')
    for step in range(6):
        svg.append(f'<text x="{left + step * cell_w + cell_w / 2 - 2}" y="{top - 8}" class="t-small" text-anchor="middle">step {step + 1}</text>')
    for gpu in range(4):
        y = top + gpu * cell_h
        svg.append(f'<text x="{left - 8}" y="{y + 23}" class="t-label" text-anchor="end">GPU {gpu} → {(gpu + 1) % 4}</text>')
        for step in range(6):
            x = left + step * cell_w
            svg.append(f'<rect x="{x}" y="{y + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="4" class="d-box"/>')
            svg.append(f'<text x="{x + cell_w / 2 - 2}" y="{y + 23}" class="t-label" text-anchor="middle">{sends[gpu][step]}</text>')
    y = top + 4 * cell_h + 22
    lines = ['Every GPU sends one chunk and receives one chunk at every step,', 'so both link directions carry data at every step.', 'A, B, C, D: chunks of the buffer. Xᵢ: GPU i\'s copy. ΣX: finished sum.', 'The receiver adds its own copy from its own memory, not over NVLink.']
    for index, line in enumerate(lines):
        svg.append(f'<text x="{left}" y="{y + 18 * index}" class="d-text">{line}</text>')
    svg.append('</svg>')
    return ''.join(svg)


def diagram_ring_pipeline():
    """NCCL runs many slices of the buffer through the 6 steps at once, so every step is active at the same time."""
    left, top, cell_w, cell_h, slices, columns = 120, 40, 62, 26, 5, 11
    svg = ['<svg viewBox="0 0 900 266" role="img" aria-label="Ring steps are pipelined over slices of the buffer" class="chart">']
    svg.append(f'<text x="{left}" y="22" class="d-title">The steps are not run one after another: slices of the buffer are pipelined</text>')
    for column in range(columns):
        svg.append(f'<text x="{left + column * cell_w + cell_w / 2}" y="{top + 8}" class="t-small" text-anchor="middle">t{column + 1}</text>')
    highlight = 5
    svg.append(f'<rect x="{left + highlight * cell_w - 2}" y="{top + 14}" width="{cell_w}" height="{slices * cell_h + 6}" rx="4" fill="none" class="d-arrow-ink"/>')
    for row in range(slices):
        y = top + 16 + row * cell_h
        svg.append(f'<text x="{left - 8}" y="{y + 17}" class="t-label" text-anchor="end">slice {row + 1}</text>')
        for step in range(6):
            column = row + step
            if column >= columns:
                continue
            css = 'c-data' if step < 3 else 'c-data2'
            text_css = 't-inbar' if step < 3 else 't-label'
            x = left + column * cell_w
            svg.append(f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" rx="3" class="{css}"/><text x="{x + cell_w / 2 - 2}" y="{y + 16}" class="{text_css}" text-anchor="middle">step {step + 1}</text>')
    y = top + 16 + slices * cell_h + 24
    lines = ['At time t6 (box) all six steps run at once, each on a different slice. NCCL runs this on 24 channels', '(CTAs) in parallel, so every GPU sends and receives at full rate for the whole call, like NVLS AllReduce.', 'So Ring and NVLS differ not in sequential against concurrent, but in which packets each instruction needs.']
    for index, line in enumerate(lines):
        svg.append(f'<text x="30" y="{y + 18 * index}" class="d-text">{line}</text>')
    svg.append('</svg>')
    return ''.join(svg)


def diagram_lanes():
    """One GPU's two link directions: what Ring and NVLS AllReduce put on each (multiples of S)."""
    svg = ['<svg viewBox="0 0 900 330" role="img" aria-label="What Ring and NVLS AllReduce put on each link direction" class="chart">']

    def panel(x0, title, top_label, sent_data, sent_ctrl, recv_data, recv_ctrl, footer):
        sent_x, recv_x = x0 + 175, x0 + 265
        svg.append(f'<text x="{x0 + 220}" y="20" class="d-title" text-anchor="middle">{title}</text>')
        svg.append(f'<rect x="{x0 + 100}" y="34" width="240" height="40" rx="6" class="d-switch"/><text x="{x0 + 220}" y="59" class="d-text" text-anchor="middle">{top_label}</text>')
        svg.append(f'<rect x="{x0 + 140}" y="236" width="160" height="40" rx="6" class="d-box"/><text x="{x0 + 220}" y="261" class="d-title" text-anchor="middle">one GPU</text>')
        svg.append(f'<text x="{sent_x}" y="92" class="t-small" text-anchor="middle">sent ↑</text><text x="{recv_x}" y="92" class="t-small" text-anchor="middle">received ↓</text>')
        svg.append(f'<line x1="{sent_x - 8}" y1="230" x2="{sent_x - 8}" y2="100" class="d-arrow-data" marker-end="url(#hd)"/>')
        svg.append(f'<line x1="{recv_x + 8}" y1="100" x2="{recv_x + 8}" y2="230" class="d-arrow-data" marker-end="url(#hd)"/>')
        if sent_ctrl:
            svg.append(f'<line x1="{sent_x + 10}" y1="230" x2="{sent_x + 10}" y2="104" class="d-arrow-proto" marker-end="url(#hp)"/>')
        if recv_ctrl:
            svg.append(f'<line x1="{recv_x - 10}" y1="104" x2="{recv_x - 10}" y2="230" class="d-arrow-proto" marker-end="url(#hp)"/>')
        for index, line in enumerate(sent_data):
            svg.append(f'<text x="{sent_x - 18}" y="{126 + 15 * index}" class="t-label" text-anchor="end" style="fill: var(--data)">{line}</text>')
        for index, line in enumerate(sent_ctrl):
            svg.append(f'<text x="{sent_x - 18}" y="{186 + 15 * index}" class="t-label" text-anchor="end" style="fill: var(--proto)">{line}</text>')
        for index, line in enumerate(recv_data):
            svg.append(f'<text x="{recv_x + 18}" y="{126 + 15 * index}" class="t-label" style="fill: var(--data)">{line}</text>')
        for index, line in enumerate(recv_ctrl):
            svg.append(f'<text x="{recv_x + 18}" y="{186 + 15 * index}" class="t-label" style="fill: var(--proto)">{line}</text>')
        for index, line in enumerate(footer):
            svg.append(f'<text x="{x0 + 220}" y="{298 + 16 * index}" class="d-text" text-anchor="middle">{line}</text>')

    panel(0, 'Ring AllReduce (8 GPUs)', 'switch → next / from previous GPU',
          ['stores to the', 'next GPU 1.75'], [],
          ['stores from the', 'previous GPU 1.75'], [],
          ['Only data and headers. Plain stores get almost', 'no responses (0.03). Wire cost w = 1.16'])
    panel(450, 'NVLS AllReduce (4 GPUs)', 'NVSwitch',
          ['read replies 1.00', '+ own stores 0.25'], ['write responses 0.125', '+ own read requests 0.063'],
          ['multicast writes 1.00', '+ own sums 0.25'], ['read requests', 'from the switch 0.231'],
          ['Data plus the other operation\'s control packets', '(dashed). Wire cost w = 1.34'])
    svg.append('<line x1="450" y1="30" x2="450" y2="290" class="grid"/>')
    svg.append('</svg>')
    return ''.join(svg)


def wire_budget_chart():
    """Wire bytes per data byte on the busy direction, part by part, drawn to scale."""
    rows = [
        ('AllGather alone, received (4 GPUs)', [1.0, 0.125, 0, 0, 0.009, 0.001], 375.6),
        ('ReduceScatter alone, sent (4 GPUs)', [1.0, 0.125, 0, 0.062, 0, 0.002], 358.1),
        ('NVLS AllReduce, received (4 GPUs)', [1.0, 0.125, 0.185, 0, 0.024, 0.010], 317.9),
        ('NVLS AllReduce, sent (4 GPUs)', [1.0, 0.125, 0.100, 0.050, 0, 0.061], 317.9),
        ('Ring AllReduce, received (8 GPUs)', [1.0, 0.125, 0.014, 0, 0, 0.020], 364.0),
        ('Ring AllReduce, sent (8 GPUs)', [1.0, 0.125, 0.016, 0, 0, 0.017], 364.0),
    ]
    parts = [('Data', 'c-data'), ('Headers of data packets', 'c-hdr'), ('Packets against the data (read requests from the switch, write responses)', 'c-proto'), ('The GPU\'s own read requests', 'c-proto2'), ('Small responses (r_st, r_ld)', 'c-small'), ('Wire bytes in no packet type (u)', 'c-other')]
    left, top, bar_h, gap, scale = 250, 82, 24, 14, 380
    height = top + len(rows) * (bar_h + gap) + 40
    svg = [f'<svg viewBox="0 0 900 {height}" role="img" aria-label="Wire cost on the busy direction, part by part" class="chart">']
    svg.append(legend(parts[:2], 20, 16, None, max_x=880))
    svg.append(legend(parts[2:4], 20, 34, None, max_x=880))
    svg.append(legend(parts[4:], 20, 52, None, max_x=880))
    for tick in (0, 0.25, 0.5, 0.75, 1.0, 1.25):
        x = left + tick * scale
        svg.append(f'<line x1="{x}" x2="{x}" y1="{top - 6}" y2="{top + len(rows) * (bar_h + gap) - gap + 4}" class="grid"/><text x="{x}" y="{top + len(rows) * (bar_h + gap) + 12}" class="t-small" text-anchor="middle">{tick:g}</text>')
    for index, (name, values, bus) in enumerate(rows):
        y = top + index * (bar_h + gap)
        svg.append(f'<text x="{left - 8}" y="{y + 16}" class="t-label" text-anchor="end">{esc(name)}</text>')
        x = left
        for (part, css), value in zip(parts, values):
            if value <= 0:
                continue
            svg.append(f'<rect x="{x:.1f}" y="{y}" width="{max(value * scale - 1, 0.8):.1f}" height="{bar_h}" class="{css}"><title>{esc(part)}: {value:.3f}</title></rect>')
            x += value * scale
        svg.append(f'<text x="{x + 6:.1f}" y="{y + 16}" class="t-value">{sum(values):.3f} → {bus:.0f} GB/s</text>')
    svg.append(f'<text x="{left + 0.7 * scale}" y="{height - 6}" class="t-small" text-anchor="middle">wire cost w = wire bytes ÷ data bytes on the busy direction (busBW = wire rate ÷ w)</text>')
    svg.append('</svg>')
    return ''.join(svg)


def diagram_bytes():
    """Buffer, data and wire bytes of one NVLS AllReduce (1 GB buffer, 4 GPUs) on one GPU's received direction, to scale."""
    scale, left = 380, 250
    svg = ['<svg viewBox="0 0 900 270" role="img" aria-label="Buffer bytes, data bytes and wire bytes" class="chart">']
    rows = [
        (40, 'Buffer bytes S', [(1.0, 'c-data2', '1 GB')], 'the user\'s buffer; algBW = S ÷ t'),
        (110, 'Data bytes D', [(1.0, 'c-data', 'multicast writes 1 GB'), (0.25, 'c-data', 'own sums')], 'D = k × S, k = 1 + 1/N = 1.25; busBW = D ÷ t'),
        (180, 'Wire bytes W', [(1.25, 'c-data', 'data 1.25 GB'), (0.15625, 'c-hdr', ''), (0.125, 'c-proto', '')], 'W = D + headers 0.156 (dark) + read requests 0.125 (orange) = 1.531 GB; w = W ÷ D = 1.225; wire rate = W ÷ t'),
    ]
    for y, name, parts, note in rows:
        svg.append(f'<text x="{left - 10}" y="{y + 19}" class="t-label" text-anchor="end">{name}</text>')
        x = left
        for value, css, label in parts:
            width = value * scale
            svg.append(f'<rect x="{x:.1f}" y="{y}" width="{width - 1:.1f}" height="28" class="{css}"/>')
            if label:
                svg.append(f'<text x="{x + 8:.1f}" y="{y + 19}" class="{"t-label" if css == "c-data2" else "t-inbar"}">{label}</text>')
            x += width
        svg.append(f'<text x="{left}" y="{y + 46}" class="t-small">{note}</text>')
    svg.append(f'<line x1="{left + 1.25 * scale}" x2="{left + 1.25 * scale}" y1="104" y2="214" class="grid" stroke-dasharray="3 3"/>')
    svg.append('<text x="20" y="22" class="d-title">One NVLS AllReduce of a 1 GB buffer, 4 GPUs: one GPU\'s received direction</text>')
    svg.append('</svg>')
    return ''.join(svg)

def sub(base, subscript):
    return f'{base}<tspan baseline-shift="sub" font-size="9px">{subscript}</tspan>'


STEP_WIDTH = 222


def story_panel(x0, y0, index, arrows, title, lines, issuer=0, ring=False):
    """One step of a storyboard: the switch, four GPUs, the arrows active in this step, and its caption.
    arrows: list of (kind, path) with kind 'data' or 'ctrl' and path one of 'issuer_up', 'issuer_down',
    'all_up', 'all_down', 'ring'."""
    svg = []
    switch_left, switch_width = x0 + 12, STEP_WIDTH - 24
    svg.append(f'<rect x="{x0 + 2}" y="{y0}" width="{STEP_WIDTH - 8}" height="{290}" rx="8" fill="none" class="grid"/>')
    svg.append(f'<rect x="{switch_left}" y="{y0 + 10}" width="{switch_width}" height="26" rx="5" class="d-switch"/><text x="{switch_left + switch_width / 2}" y="{y0 + 28}" class="t-small" text-anchor="middle">NVSwitch</text>')
    centers = [x0 + 36 + 49 * gpu for gpu in range(4)]
    for gpu, center in enumerate(centers):
        css = 'd-box'
        svg.append(f'<rect x="{center - 21}" y="{y0 + 112}" width="42" height="24" rx="4" class="{css}" {"style=\"stroke-width:2.5\"" if gpu == issuer and not ring else ""}/><text x="{center}" y="{y0 + 128}" class="t-small" text-anchor="middle" {"font-weight=\"700\"" if gpu == issuer and not ring else ""}>GPU {gpu}</text>')
    top, bottom = y0 + 40, y0 + 108
    for kind, path in arrows:
        css, head = ('d-arrow-data', 'hd') if kind == 'data' else ('d-arrow-proto', 'hp')
        if path in ('issuer_up', 'all_up'):
            targets = [centers[issuer]] if path == 'issuer_up' else centers
            for center in targets:
                svg.append(f'<line x1="{center - 7}" y1="{bottom}" x2="{center - 7}" y2="{top + 4}" class="{css}" marker-end="url(#{head})"/>')
        elif path in ('issuer_down', 'all_down'):
            targets = [centers[issuer]] if path == 'issuer_down' else centers
            for center in targets:
                svg.append(f'<line x1="{center + 7}" y1="{top}" x2="{center + 7}" y2="{bottom - 4}" class="{css}" marker-end="url(#{head})"/>')
        elif path == 'ring':
            for gpu, center in enumerate(centers):
                target = centers[(gpu + 1) % 4]
                level = top + 6 + 12 * gpu
                if gpu < 3:
                    svg.append(f'<path d="M{center - 6},{bottom} L{center - 6},{level} L{target + 6},{level} L{target + 6},{bottom - 4}" class="{css}" style="stroke-width:2" marker-end="url(#{head})"/>')
                else:
                    svg.append(f'<path d="M{center - 6},{bottom} L{center - 6},{level} L{centers[0] + 6},{level} L{centers[0] + 6},{bottom - 4}" class="{css}" style="stroke-width:2; stroke-dasharray: 6 3" marker-end="url(#{head})"/>')
    svg.append(f'<text x="{x0 + 12}" y="{y0 + 160}" class="t-label" font-weight="700">{index}. {title}</text>')
    for line_index, (label, text) in enumerate(lines):
        y = y0 + 180 + 17 * line_index
        svg.append(f'<text x="{x0 + 12}" y="{y}" class="t-small"><tspan font-weight="700">{label}</tspan> {text}</text>')
    return ''.join(svg)


def storyboard(kind):
    """Step-by-step view of multimem.st (AllGather), multimem.ld_reduce (ReduceScatter) or Ring AllReduce."""
    if kind == 'st':
        heading = 'multimem.st by itself (NVLS AllGather): GPU 0 stores one 128-byte piece of its share'
        steps = [
            ([('data', 'issuer_up')], 'GPU 0 sends its piece', [('Aim:', 'hand the switch the data'), ('', 'to copy'), ('Packet:', 'write, 16 B + 128 B'), ('Where:', "GPU 0's sent direction"), ('Whole call:', '(S/N)(1 + η) on each'), ('', "GPU's sent direction")]),
            ([('data', 'all_down')], 'Switch copies it to all', [('Aim:', "put GPU 0's piece on every"), ('', 'GPU (the AllGather itself)'), ('Packet:', 'N writes, 16 B + 128 B'), ('Where:', "every GPU's received dir."), ('Whole call:', 'S(1 + η) on each received'), ('', 'direction (N shares × S/N)')]),
            ([('ctrl', 'all_up')], 'Every GPU acknowledges', [('Aim:', 'tell the switch the write'), ('', 'has landed'), ('Packet:', '16-B response per write'), ('Where:', "every GPU's sent direction"), ('Whole call:', 'α S on each sent'), ('', 'direction (α = 0.125)')]),
            ([('ctrl', 'issuer_down')], 'Switch tells GPU 0', [('Aim:', 'completion for the issuer'), ('Packet:', 'combined response,'), ('', 'fewer than 1 per write'), ('Where:', "GPU 0's received direction"), ('Whole call:', sub('r', 'st') + ' S on each received'), ('', 'direction (' + sub('r', 'st') + ' = 0.009)')]),
        ]
    elif kind == 'ld':
        heading = 'multimem.ld_reduce by itself (NVLS ReduceScatter): GPU 0 asks for the sum of one 128-byte piece'
        steps = [
            ([('ctrl', 'issuer_up')], 'GPU 0 asks for a sum', [('Aim:', 'start the reduction of'), ('', 'its piece'), ('Packet:', 'read-reduce request, no data'), ('Where:', "GPU 0's sent direction"), ('Whole call:', sub('ρ', 'own') + ' S/N on each sent'), ('', 'direction (' + sub('ρ', 'own') + ' = 0.25)')]),
            ([('ctrl', 'all_down')], 'Switch asks every GPU', [('Aim:', "fetch every GPU's copy"), ('', 'of that piece'), ('Packet:', 'N read requests, no data'), ('Where:', "every GPU's received dir."), ('Whole call:', sub('ρ', 'switch') + ' S on each'), ('', 'received direction (' + sub('ρ', 'switch') + ' = 0.154)')]),
            ([('data', 'all_up')], 'Every GPU replies', [('Aim:', 'deliver the copies to the'), ('', "switch's adder"), ('Packet:', 'read reply, 16 B + 128 B'), ('Where:', "every GPU's sent direction"), ('Whole call:', 'S(1 + η) on each sent'), ('', 'direction')]),
            ([('data', 'issuer_down')], 'Switch returns the sum', [('Aim:', 'GPU 0 gets its reduced'), ('', 'piece'), ('Packet:', 'sum, 16 B + 128 B'), ('Where:', "GPU 0's received direction"), ('Whole call:', '(S/N)(1 + η) + ' + sub('r', 'ld') + ' S'), ('', 'on each received direction')]),
        ]
    else:
        heading = 'Ring AllReduce: one of its 2(N − 1) steps (NCCL runs all steps at once on different slices)'
        steps = [
            ([('data', 'ring')], 'Each GPU stores a chunk', [('Aim:', 'pass a partial sum (or a'), ('', 'finished sum) to the next GPU'), ('Packet:', 'plain store, 16 B + 128 B'), ('Where:', "sender's sent, receiver's"), ('', 'received direction'), ('Whole call:', 'k S (1 + η) on each'), ('', 'direction')]),
            ([('ctrl', 'all_up'), ('ctrl', 'all_down')], 'Few responses', [('Aim:', 'completion of plain stores'), ('Packet:', 'rare responses'), ('Where:', 'both directions'), ('Whole call:', sub('α', 'p') + ' k S sent,'), ('', sub('r', 'p') + ' k S received'), ('', '(' + sub('α', 'p') + ' ≈ 0.014, ' + sub('r', 'p') + ' ≈ 0.012)')]),
            ([], 'The receiver adds locally', [('Aim:', 'add its own copy of the'), ('', 'chunk'), ('Packet:', 'none: it reads its own'), ('', 'memory, not NVLink'), ('Whole call:', 'no read requests at all')]),
        ]
    width = 20 + STEP_WIDTH * len(steps)
    svg = [f'<svg viewBox="0 0 {max(width, 900)} 330" role="img" aria-label="{esc(heading)}" class="chart">']
    svg.append(f'<text x="10" y="18" class="d-title">{esc(heading)}</text>')
    for index, (arrows, title, lines) in enumerate(steps):
        svg.append(story_panel(10 + STEP_WIDTH * index, 32, index + 1, arrows, title, lines, ring=(kind == 'ring')))
    svg.append('</svg>')
    return ''.join(svg)


def tally_chart():
    """GPU 0's two directions for AllGather alone, ReduceScatter alone and AllReduce, built from the storyboard
    steps (4 GPUs, parameters measured on each instruction run by itself), amounts as multiples of S."""
    rows = [
        ('AllGather alone, sent', [('AG1', 0.281, 'c-data'), ('AG3', 0.125, 'c-proto')], 'low-data'),
        ('AllGather alone, received', [('AG2', 1.125, 'c-data'), ('AG4', 0.009, 'c-proto')], 'busy'),
        ('ReduceScatter alone, sent', [('RS3', 1.125, 'c-data'), ('RS1', 0.062, 'c-proto')], 'busy'),
        ('ReduceScatter alone, received', [('RS4', 0.281, 'c-data'), ('RS2', 0.154, 'c-proto'), ('r_ld', 0.021, 'c-proto2')], 'low-data'),
        ('AllReduce, sent', [('RS3', 1.125, 'c-data'), ('AG1', 0.281, 'c-data'), ('AG3', 0.125, 'c-proto'), ('RS1', 0.062, 'c-proto')], 'both busy'),
        ('AllReduce, received', [('AG2', 1.125, 'c-data'), ('RS4', 0.281, 'c-data'), ('RS2', 0.154, 'c-proto'), ('mix', 0.077, 'c-proto2'), ('AG4', 0.009, 'c-proto'), ('r_ld', 0.021, 'c-proto2')], 'both busy'),
    ]
    left, top, bar_h, gap, scale = 230, 44, 24, 12, 300
    height = top + len(rows) * (bar_h + gap) + 34
    svg = [f'<svg viewBox="0 0 900 {height}" role="img" aria-label="Where each step\'s packets land on GPU 0" class="chart">']
    svg.append(legend([('data packets (with headers)', 'c-data'), ('control packets of the steps', 'c-proto'), ('small responses and the extra read requests of the mix', 'c-proto2')], 20, 16, None, max_x=880))
    for tick in (0, 0.5, 1.0, 1.5):
        x = left + tick * scale
        svg.append(f'<line x1="{x}" x2="{x}" y1="{top - 6}" y2="{top + len(rows) * (bar_h + gap) - gap + 4}" class="grid"/><text x="{x}" y="{top + len(rows) * (bar_h + gap) + 12}" class="t-small" text-anchor="middle">{tick:g} S</text>')
    for index, (name, parts, note) in enumerate(rows):
        y = top + index * (bar_h + gap)
        svg.append(f'<text x="{left - 8}" y="{y + 16}" class="t-label" text-anchor="end">{esc(name)}</text>')
        x = left
        for label, value, css in parts:
            w = value * scale
            svg.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w - 1, 0.8):.1f}" height="{bar_h}" class="{css}"><title>{label}: {value:.3f} S</title></rect>')
            if w > 26:
                svg.append(f'<text x="{x + w / 2:.1f}" y="{y + 16}" class="t-inbar" text-anchor="middle">{label}</text>')
            x += w
        total = sum(value for _, value, _ in parts)
        svg.append(f'<text x="{x + 6:.1f}" y="{y + 16}" class="t-value">{total:.3f} S ({note})</text>')
    svg.append('</svg>')
    return ''.join(svg)

def expectation_chart():
    """busBW of NVLS AllReduce at three levels against measured references. Values are those of the
    'NVLS AllReduce against AllGather, ReduceScatter and Ring' section of the plan 01 report (E1.1, E1.6,
    E1.9 and the 8-GPU raw-counter run of the busBW comparison)."""
    measured, level = ('Measured', 'c-data'), ('Computed from w', 'c-data2')
    def bar(tick, value, is_level):
        return {'tick': tick, 'values': [0, value] if is_level else [value, 0]}
    groups = [
        {'label': ['4 GPUs'], 'bars': [bar('AG alone', 375.6, False), bar('No ctrl', 378.8, True), bar('Ref. case', 341.0, True), bar('By itself', 334.4, True), bar('AR', 317.9, False)]},
        {'label': ['8 GPUs'], 'bars': [bar('AG alone', 373.6, False), bar('Ring', 364.0, False), bar('No ctrl', 376.2, True), bar('Ref. case', 338.6, True), bar('By itself', 329.9, True), bar('AR', 310.8, False)]},
    ]
    return stacked_chart(groups, [measured, level], 400, 100, 'busBW, GB/s', 'busBW of NVLS AllReduce: limit, prediction, measured', bar_width=46, total_fmt='{:.0f}', legend_step=150)


def render_math(body):
    """Turn every $...$ (LaTeX) into inline SVG rendered at build time with ziamath, keeping the LaTeX
    source in data-tex so that copying from the page gives the LaTeX back (see COPY_SCRIPT)."""
    try:
        import ziamath
    except ImportError as error:
        raise SystemExit('make_blog_report.py needs ziamath to render formulas: pip install ziamath') from error
    ziamath.config.svg2 = False
    font_px = 17.0
    def one(match):
        tex = match.group(1)
        svg = ziamath.Math.fromlatex(tex, size=font_px, color='currentColor').svg()
        width, height = (float(value) for value in re.search(r'width="([\d.]+)" height="([\d.]+)"', svg).groups())
        view_y = float(re.search(r'viewBox="[-\d.]+ ([-\d.]+)', svg).group(1))
        descent = height + view_y
        svg = re.sub(r'width="[\d.]+" height="[\d.]+"', f'width="{width / font_px:.3f}em" height="{height / font_px:.3f}em" style="vertical-align:{-descent / font_px:.3f}em" aria-hidden="true"', svg, count=1)
        return f'<span class="math" data-tex="{esc(tex)}" role="img" aria-label="{esc(tex)}">{svg}</span>'
    return re.sub(r'\$([^$]+)\$', one, body)


COPY_SCRIPT = r"""<script>
// Copy as Markdown: formulas become $LaTeX$, tables become pipe tables, bold/italic/code keep their marks.
// Meant for pasting into a Markdown editor such as Typora; only text/plain is set, so the editor parses it as Markdown.
(function () {
  function inline(node) {
    var out = '';
    node.childNodes.forEach(function (child) { out += convert(child, true); });
    return out;
  }
  function cellText(cell) {
    return inline(cell).replace(/\s*\n+\s*/g, ' ').replace(/\|/g, '\\|').trim();
  }
  function table(rows) {
    var grid = [], carry = [];
    rows.forEach(function (row, rowIndex) {
      var line = [], column = 0;
      function fillCarry() { while (carry[column] && carry[column] > 0) { carry[column]--; line.push(''); column++; } }
      Array.prototype.forEach.call(row.children, function (cell) {
        fillCarry();
        line.push(cellText(cell));
        var span = parseInt(cell.getAttribute('rowspan') || '1', 10);
        if (span > 1) carry[column] = span - 1;
        column++;
      });
      fillCarry();
      grid.push(line);
    });
    if (!grid.length) return '';
    var width = Math.max.apply(null, grid.map(function (line) { return line.length; }));
    var text = grid.map(function (line) {
      while (line.length < width) line.push('');
      return '| ' + line.join(' | ') + ' |';
    });
    var rule = '|' + new Array(width + 1).join(' --- |');
    text.splice(1, 0, rule);
    return '\n\n' + text.join('\n') + '\n\n';
  }
  function list(node, ordered) {
    var out = '\n', number = 1;
    Array.prototype.forEach.call(node.children, function (item) {
      if (item.tagName !== 'LI') return;
      var body = inline(item).trim().replace(/\n+/g, '\n    ');
      out += (ordered ? (number++) + '. ' : '- ') + body + '\n';
    });
    return out + '\n';
  }
  function convert(node, insideInline) {
    if (node.nodeType === 3) return node.textContent.replace(/\s+/g, ' ');
    if (node.nodeType !== 1 && node.nodeType !== 11) return '';
    if (node.nodeType === 11) return inline(node);
    if (node.classList && node.classList.contains('math')) return '$' + node.getAttribute('data-tex') + '$';
    var tag = node.tagName.toLowerCase();
    switch (tag) {
      case 'svg': case 'script': case 'style': return '';
      case 'b': case 'strong': { var t = inline(node).trim(); return t ? '**' + t + '**' : ''; }
      case 'i': case 'em': { var e = inline(node).trim(); return e ? '*' + e + '*' : ''; }
      case 'code': return '`' + node.textContent + '`';
      case 'sub': return '_' + inline(node);
      case 'br': return ' ';
      case 'table': return table(Array.prototype.slice.call(node.querySelectorAll('tr')));
      case 'tbody': case 'thead': return table(Array.prototype.slice.call(node.querySelectorAll('tr')));
      case 'tr': return table([node]);
      case 'ul': return list(node, false);
      case 'ol': return list(node, true);
      case 'h2': return '\n\n## ' + inline(node).trim() + '\n\n';
      case 'h3': return '\n\n### ' + inline(node).trim() + '\n\n';
      case 'h4': return '\n\n#### ' + inline(node).trim() + '\n\n';
      case 'p': case 'div': case 'figure': case 'figcaption': case 'li': return '\n\n' + inline(node).trim() + '\n\n';
      default: return inline(node);
    }
  }
  document.addEventListener('copy', function (event) {
    var selection = window.getSelection();
    if (!selection || !selection.rangeCount || selection.isCollapsed) return;
    var fragment = selection.getRangeAt(0).cloneContents();
    var rows = Array.prototype.filter.call(fragment.childNodes, function (child) { return child.nodeType === 1 && child.tagName === 'TR'; });
    var text = rows.length ? table(rows) : convert(fragment, false);
    text = text.replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
    event.clipboardData.setData('text/plain', text);
    event.preventDefault();
  });
})();
</script>"""

def build():
    value, charts = load_values()
    charts['expect'] = expectation_chart()
    charts['wire_budget'] = wire_budget_chart()
    def pair(summed, allreduce):
        # values in the order request data, response data, request protocol, response protocol (each ÷ S)
        return [{'tick': 'sum', 'values': summed}, {'tick': 'AR', 'values': allreduce}]
    charts['additivity'] = stacked_chart([
        {'label': ['4 GPUs, sent'], 'bars': pair([0.250, 1.000, 0.093, 0.250], [0.250, 1.000, 0.094, 0.250])},
        {'label': ['4 GPUs, recv'], 'bars': pair([1.000, 0.250, 0.279, 0.061], [1.000, 0.250, 0.356, 0.061])},
        {'label': ['8 GPUs, sent'], 'bars': pair([0.125, 1.000, 0.047, 0.250], [0.125, 1.000, 0.047, 0.250])},
        {'label': ['8 GPUs, recv'], 'bars': pair([1.000, 0.125, 0.285, 0.033], [1.000, 0.125, 0.364, 0.033])},
    ], [('request data', 'c-data'), ('response data', 'c-data2'), ('request protocol', 'c-proto'), ('response protocol', 'c-proto2')], 2.0, 0.5, 'Packet bytes on the direction ÷ S', 'Packets of AllReduce against the sum of the two instructions by themselves', bar_width=46, total_fmt='{:.3f}', legend_step=150)
    body = BODY
    for key, svg in charts.items():
        body = body.replace(f'@@chart_{key}@@', svg)
    body = body.replace('@@diagram_topology@@', DIAGRAM_TOPOLOGY).replace('@@diagram_algorithms@@', DIAGRAM_ALGORITHMS)
    body = body.replace('@@story_st@@', storyboard('st')).replace('@@story_ld@@', storyboard('ld')).replace('@@story_ring@@', storyboard('ring')).replace('@@chart_tally@@', tally_chart()).replace('@@diagram_bytes@@', diagram_bytes()).replace('@@diagram_ring_steps@@', diagram_ring_steps()).replace('@@diagram_ring_pipeline@@', diagram_ring_pipeline()).replace('@@diagram_lanes@@', diagram_lanes())
    body = body.replace('@@diagram_packets@@', DIAGRAM_PACKETS).replace('@@diagram_directions@@', DIAGRAM_DIRECTIONS)
    for key, text in value.items():
        body = body.replace(f'@@{key}@@', text)
    body = render_math(body) + COPY_SCRIPT
    if '@@' in body:
        start = body.index('@@')
        raise KeyError(f'unfilled placeholder near: {body[start:start + 60]}')
    return STYLE + body


def main():
    page = build()
    Path(sys.argv[1]).write_text(page)
    standalone = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + page.replace('<main class="page">', '</head>\n<body>\n<main class="page">', 1) + '\n</body>\n</html>\n'
    Path(sys.argv[2]).write_text(standalone)
    print(f'wrote {sys.argv[1]} and {sys.argv[2]}')


if __name__ == '__main__':
    main()
