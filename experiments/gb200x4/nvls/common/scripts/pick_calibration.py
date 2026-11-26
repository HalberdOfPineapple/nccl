#!/usr/bin/env python3
"""Pick the calibrated CTA count per arm from E1.2 medians (configs.csv of summarize_nccl.py).

Stopping rule (plan 01 E1.2, stated before the runs): the plateau starts at the first CTA count
after which two successive increases each add less than 1% of busBW. The calibrated count is the
smallest count on the plateau. If no count qualifies, the arm is 'not saturated' and the count with
the highest median busBW is used, marked as such. A drop of more than 2% after the best point is
reported as a downturn. Arms without a CTA knob (copy engine) get null.

Usage: pick_calibration.py <configs.csv> --block e1_2 --output <calibrated_ctas.json>
The output is reviewed by a person before batch jobs use it; the 'details' field says why.
"""

import argparse
import csv
import json
from pathlib import Path

IMPROVEMENT_LIMIT = 0.01
DOWNTURN_LIMIT = 0.02
ARMS_WITHOUT_CTAS = ('AG-CE',)


def pickArm(points):
    """points: sorted [(ctas, busbw_median, busbw_iqr)] -> (chosen ctas, detail dict)."""
    for index in range(len(points) - 2):
        first, second, third = points[index], points[index + 1], points[index + 2]
        gainOne = (second[1] - first[1]) / first[1]
        gainTwo = (third[1] - second[1]) / second[1]
        if gainOne < IMPROVEMENT_LIMIT and gainTwo < IMPROVEMENT_LIMIT:
            chosen = first
            status = 'plateau'
            break
    else:
        chosen = max(points, key=lambda point: point[1])
        status = 'not_saturated'
    best = max(points, key=lambda point: point[1])
    after = [point for point in points if point[0] > best[0]]
    downturn = bool(after) and min(point[1] for point in after) < best[1] * (1 - DOWNTURN_LIMIT)
    detail = {
        'status': status, 'chosen_ctas': chosen[0], 'chosen_busbw_gbps': chosen[1],
        'chosen_iqr_gbps': chosen[2], 'best_ctas': best[0], 'best_busbw_gbps': best[1],
        'downturn_after_best': downturn,
        'curve': [{'ctas': point[0], 'busbw_gbps': point[1], 'iqr_gbps': point[2]} for point in points],
    }
    return chosen[0], detail


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configs', type=Path)
    parser.add_argument('--block', default='e1_2')
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()

    curves = {}
    with arguments.configs.open() as handle:
        for row in csv.DictReader(handle):
            if row['block'] != arguments.block or row['ctas_requested'] == 'default' or not row['busbw_median_gbps']:
                continue
            curves.setdefault(row['arm'], []).append(
                (int(row['ctas_requested']), float(row['busbw_median_gbps']), float(row['busbw_iqr_gbps'] or 0)))

    calibrated = {arm: None for arm in ARMS_WITHOUT_CTAS}
    details = {}
    for arm, points in sorted(curves.items()):
        calibrated[arm], details[arm] = pickArm(sorted(points))
    arguments.output.write_text(json.dumps({
        'source': str(arguments.configs), 'block': arguments.block,
        'rule': 'first CTA count after which two successive increases each add < 1% busBW',
        'ctas': calibrated, 'details': details}, indent=1) + '\n')
    for arm, detail in details.items():
        print(f"{arm:8s} {detail['status']:14s} ctas={detail['chosen_ctas']:>3} busbw={detail['chosen_busbw_gbps']:7.1f} "
              f"best={detail['best_ctas']}@{detail['best_busbw_gbps']:.1f} downturn={detail['downturn_after_best']}")


if __name__ == '__main__':
    main()
