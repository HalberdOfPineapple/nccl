#!/usr/bin/env python3
"""Session files for the AllGather and ReduceScatter scalability series (run_allreduce.py with a `collective` per cell).

Run from experiments/h100-scale/allgather_reducescatter/scalability/symmetric_multicast/. One call per GPU count:

  make_configs_agrs.py N                      selection, calibration (full at 4 and 8 GPUs), scaling, payload, link files
  make_configs_agrs.py N --short-calibration SELECTION_JSONL
                                              at 2 and 6 GPUs: calibration with NCCL's choice and twice it, read from
                                              the selection runs of that GPU count

Cell ids: <step>_n<N>_<collective>_<arm>_<auto|c<CTAs>>[_long|_short].
"""
import argparse
import json
from pathlib import Path

COLLECTIVES = ['allgather', 'reducescatter']
ARMS = ['ring', 'nvls', 'sym_multicast', 'sym_plain']
GPU_SETS = {2: '4,5', 4: '4,5,6,7', 6: '1,2,3,4,5,6', 8: '0,1,2,3,4,5,6,7'}
FULL_CALIBRATION_CTAS = [None, 2, 4, 8, 16, 32, 64]
LINK_ARMS = {8: ARMS, 4: ['ring', 'sym_multicast']}
LINK_ITERS = {'long': 200, 'short': 20}
# Added 2026-11-18 after the 8-GPU calibration: NCCL's choice for plain load/store gave 0.82 (AllGather) and 0.93
# (ReduceScatter) of the best; 32 CTAs is within 0.5% of the best for both. Run at every GPU count so curves stay matched.
FIXED_PLAIN_CTAS = 32
# Added 2026-11-18 after the 4-, 6- and 2-GPU calibration: NCCL's choice gave 0.83-0.94 of the best for Ring at 2 and 4 GPUs
# and 0.89 for classic NVLS AllGather at 2 GPUs. 64 CTAs (NCCL's upper limit) is the best or near-best value measured at 4 and 8 GPUs.
FIXED_MAX_CTAS = 64


def arms_at(gpu_count):
    # Symmetric multicast is off at 2 GPUs (sym_kernels.cc:195).
    return [arm for arm in ARMS if not (arm == 'sym_multicast' and gpu_count == 2)]


def cell(step, gpu_count, collective, arm, ctas=None, min_size='2G', max_size='2G'):
    entry = {'id': f"{step}_n{gpu_count}_{collective}_{arm}_{'auto' if ctas is None else f'c{ctas}'}", 'collective': collective,
             'arm': arm, 'devices': GPU_SETS[gpu_count], 'min_size': min_size, 'max_size': max_size}
    if ctas:
        entry['ctas'] = ctas
    return entry


def write(path, rounds, cells):
    path.write_text(json.dumps({'rounds': rounds, 'cells': cells}, indent=1) + '\n')
    print(f'{path}: {len(cells)} cells x {rounds} rounds')


def ctas_chosen(selection_jsonl, gpu_count):
    """NCCL's CTA choice per (collective, arm) from the selection runs."""
    chosen = {}
    for line in Path(selection_jsonl).read_text().splitlines():
        record = json.loads(line)
        if record['gpu_count'] != gpu_count or not record['valid']:
            continue
        if record['sym_channel_values']:
            value = record['sym_channel_values'][0]
        else:
            low, high = record['algo_channel_ranges'][0].split('..')
            value = int(high) - int(low) + 1
        chosen[(record['collective'], record['arm'])] = value
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu_count', type=int)
    parser.add_argument('--short-calibration', metavar='SELECTION_JSONL')
    arguments = parser.parse_args()
    gpu_count = arguments.gpu_count
    configs = Path('configs')
    configs.mkdir(exist_ok=True)
    pairs = [(collective, arm) for collective in COLLECTIVES for arm in arms_at(gpu_count)]
    if arguments.short_calibration:
        chosen = ctas_chosen(arguments.short_calibration, gpu_count)
        cells = []
        for collective, arm in pairs:
            cells.append(cell('calibration', gpu_count, collective, arm))
            cells.append(cell('calibration', gpu_count, collective, arm, 2 * chosen[(collective, arm)]))
        write(configs / f'calibration_n{gpu_count}.json', 3, cells)
        return
    write(configs / f'selection_n{gpu_count}.json', 1, [cell('selection', gpu_count, collective, arm) for collective, arm in pairs])
    if gpu_count in (4, 8):
        write(configs / f'calibration_n{gpu_count}.json', 3,
              [cell('calibration', gpu_count, collective, arm, ctas) for collective, arm in pairs for ctas in FULL_CALIBRATION_CTAS])
    fixed = [(collective, 'sym_plain', FIXED_PLAIN_CTAS) for collective in COLLECTIVES]
    fixed += [(collective, arm, FIXED_MAX_CTAS) for collective in COLLECTIVES for arm in ('ring', 'nvls')]
    write(configs / f'scaling_n{gpu_count}.json', 5,
          [cell('scaling', gpu_count, collective, arm) for collective, arm in pairs] + [cell('scaling', gpu_count, c, a, n) for c, a, n in fixed])
    write(configs / f'payload_n{gpu_count}.json', 5,
          [cell('payload', gpu_count, collective, arm, min_size='32M', max_size='8G') for collective, arm in pairs]
          + [cell('payload', gpu_count, c, a, n, min_size='32M', max_size='8G') for c, a, n in fixed])
    if gpu_count in LINK_ARMS:
        cells = []
        for collective in COLLECTIVES:
            for arm in LINK_ARMS[gpu_count]:
                for length, iters in LINK_ITERS.items():
                    entry = cell('link', gpu_count, collective, arm)
                    entry['id'] += f'_{length}'
                    entry['iters'] = iters
                    cells.append(entry)
        write(configs / f'link_n{gpu_count}.json', 3, cells)


if __name__ == '__main__':
    main()
