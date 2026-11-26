#!/usr/bin/env python3
"""Plan 06 session files (configs/*.json) for run_custom_mc.py. Run from scalability/custom_mc_kernels/.

check_n<N>.json      E6.0: all four kernels, 4 CTAs x 1024 threads/CTA, 32 MiB to 2 GiB.
cta_n<N>.json        E6.1: one cell per CTA count (1 to 128), 32 MiB to 2 GiB, 3 rounds.
E6.2 and E6.3 files are written by make_configs_custom_mc.py --flat after E6.1 gives the flat CTA counts.
"""
import argparse
import json
from pathlib import Path

GPU_SETS = {2: '4,5', 4: '4,5,6,7', 6: '1,2,3,4,5,6', 8: '0,1,2,3,4,5,6,7'}
SIZE_32_MIB = 32 * 2**20
# E6.2 starts at 16 MiB and discards it: in the CTA sweep the NCCL column was sometimes slow at the first size of a run.
SIZE_16_MIB = 16 * 2**20
SIZE_2_GIB = 2 * 2**30
# 8 GPUs ran 1-128 (cta_n8.json) and 256 was added after (cta_n8_ext256.json); later GPU counts include 256 from the start.
CTA_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def write(path, session):
    path.write_text(json.dumps(session, indent=1) + '\n')
    print(path)


def check_and_cta_sessions(configs, gpu_counts):
    for gpu_count in gpu_counts:
        devices = GPU_SETS[gpu_count]
        write(configs / f'check_n{gpu_count}.json', {'rounds': 1, 'cells': [
            {'id': f'check_n{gpu_count}', 'devices': devices, 'min_ctas': 4, 'max_ctas': 4, 'min_size': SIZE_32_MIB, 'max_size': SIZE_2_GIB}]})
        write(configs / f'cta_n{gpu_count}.json', {'rounds': 3, 'cells': [
            {'id': f'cta_n{gpu_count}_c{ctas}', 'devices': devices, 'min_ctas': ctas, 'max_ctas': ctas, 'min_size': SIZE_32_MIB, 'max_size': SIZE_2_GIB}
            for ctas in CTA_COUNTS]})


def flat_sessions(configs, gpu_count, flat_ctas):
    """flat_ctas: {'oneshot': c, 'twoshot': c, 'poison': c}. E6.2: each kernel at its flat count with the NCCL column, 5 rounds.
    E6.3: one-shot and two-shot alone at 2 GiB, long (100 timed calls) and short (10) runs, 3 rounds."""
    devices = GPU_SETS[gpu_count]
    write(configs / f'flat_n{gpu_count}.json', {'rounds': 5, 'cells': [
        {'id': f'flat_n{gpu_count}_{kernel}_c{ctas}', 'devices': devices, 'min_ctas': ctas, 'max_ctas': ctas,
         'min_size': SIZE_16_MIB, 'max_size': SIZE_2_GIB, 'kernels': f'{kernel},nccl'} for kernel, ctas in flat_ctas.items()]})
    cells = []
    for kernel in ('oneshot', 'twoshot'):
        for length, iters in (('long', 100), ('short', 10)):
            cells.append({'id': f'link_n{gpu_count}_{kernel}_c{flat_ctas[kernel]}_{length}', 'devices': devices,
                          'min_ctas': flat_ctas[kernel], 'max_ctas': flat_ctas[kernel], 'min_size': SIZE_2_GIB, 'max_size': SIZE_2_GIB,
                          'kernels': kernel, 'timed_iters': iters})
    write(configs / f'link_n{gpu_count}.json', {'rounds': 3, 'cells': cells})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, default=[2, 4, 6, 8], help='GPU counts for the check and CTA-sweep files')
    parser.add_argument('--flat', nargs=4, metavar=('GPU_COUNT', 'ONESHOT', 'TWOSHOT', 'POISON'), type=int)
    arguments = parser.parse_args()
    configs = Path('configs')
    configs.mkdir(exist_ok=True)
    if arguments.flat:
        gpu_count, oneshot, twoshot, poison = arguments.flat
        flat_sessions(configs, gpu_count, {'oneshot': oneshot, 'twoshot': twoshot, 'poison': poison})
    else:
        check_and_cta_sessions(configs, arguments.gpus)


if __name__ == '__main__':
    main()
