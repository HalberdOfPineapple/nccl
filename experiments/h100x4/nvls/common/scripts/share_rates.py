#!/usr/bin/env python3
"""Role byte rates (N1) of RCA plan 01 E1.7: how st_multicast and ld_reduce share the link.

Run names are e1_7_multicast_c<M>_reduce_c<R>_R<k>. The sweep with st_multicast fixed at 32 CTAs varies
ld_reduce ("reduce_varied"); the sweep with ld_reduce fixed at 32 varies st_multicast ("multicast_varied").
The 32 + 32 setting belongs to both sweeps.

Usage: share_rates.py <raw directory> <output csv>
Column names are an interface for rca_figures.py.
"""
import csv
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mm_summary import summarize_run  # noqa: E402

RUN_NAME = re.compile(r'^e1_7_multicast_c(?P<multicast>\d+)_reduce_c(?P<reduce>\d+)_R(?P<round>\d+)$')


def main():
    raw_directory = Path(sys.argv[1])
    output = Path(sys.argv[2])
    per_setting = {}
    for path in sorted(raw_directory.glob('e1_7_*_R*.json')):
        match = RUN_NAME.match(path.stem)
        if not match:
            continue
        summary = summarize_run(path)
        key = (int(match['multicast']), int(match['reduce']))
        for role, values in summary['roles'].items():
            per_setting.setdefault(key + (role,), []).append(values['rate_per_issuing_gpu_gbps'])
    rows = []
    for (multicast_ctas, reduce_ctas, role), rates in sorted(per_setting.items()):
        sweeps = []
        if multicast_ctas == 32:
            sweeps.append(('reduce_varied', reduce_ctas))
        if reduce_ctas == 32:
            sweeps.append(('multicast_varied', multicast_ctas))
        for sweep, varied_ctas in sweeps:
            rows.append({
                'sweep': sweep,
                'varied_ctas': varied_ctas,
                'multicast_ctas': multicast_ctas,
                'reduce_ctas': reduce_ctas,
                'threads_per_cta': 512,
                'role': role,
                'rounds': len(rates),
                'rate_gbps': statistics.median(rates),  # N1, role bytes of the GPU's own part per ns
                'rate_min_gbps': min(rates),
                'rate_max_gbps': max(rates),
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['sweep']:<17} multicast {row['multicast_ctas']:>2} reduce {row['reduce_ctas']:>2} {row['role']:<10} {row['rate_gbps']:6.1f} GB/s [{row['rate_min_gbps']:.1f}, {row['rate_max_gbps']:.1f}]")


if __name__ == '__main__':
    main()
