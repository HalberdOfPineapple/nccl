#!/usr/bin/env python3
"""Writes the session configs of the scalability series for 2, 4 and 6 GPUs.

Every GPU count gets the same cells as the 8-GPU sessions, so scaling curves join matched settings:
NCCL's own CTA choice for every arm, plus plain load/store at 16 and 48 CTAs. The 8-GPU configs were
written before this script and are left as they are. Run from the `scalability` directory.
"""
import json

ARMS = ['ring', 'nvls', 'sym_multicast', 'sym_plain']
MAIN_SETS = {2: '4,5', 4: '4,5,6,7', 6: '1,2,3,4,5,6'}
CONTROL_SETS = {2: ['0,1', '3,4'], 4: ['0,1,2,3', '2,3,4,5'], 6: ['0,1,2,3,4,5']}
CALIBRATION_CTAS = [None, 2, 4, 8, 16, 24, 32, 48, 64]


def cell(cell_id, arm, devices, ctas=None):
    entry = {'id': cell_id, 'arm': arm, 'devices': devices, 'min_size': '2G', 'max_size': '2G'}
    if ctas:
        entry['ctas'] = ctas
    return entry


def write(path, rounds, cells):
    with open(path, 'w') as handle:
        json.dump({'rounds': rounds, 'cells': cells}, handle, indent=1)


def main():
    for gpu_count, devices in MAIN_SETS.items():
        # Multimem is off for 2-GPU symmetric kernels (sym_kernels.cc), so that arm is not run there.
        arms = [arm for arm in ARMS if not (gpu_count == 2 and arm == 'sym_multicast')]
        calibration = []
        for ctas in CALIBRATION_CTAS:
            for arm in arms:
                calibration.append(cell(f"n{gpu_count}_{arm}_{'auto' if ctas is None else f'c{ctas}'}", arm, devices, ctas))
        write(f'selection_calibration/configs/calibration_n{gpu_count}.json', 3, calibration)
        scaling = [cell(f'n{gpu_count}_{arm}_auto', arm, devices) for arm in arms]
        scaling += [cell(f'n{gpu_count}_sym_plain_c{ctas}', 'sym_plain', devices, ctas) for ctas in (16, 48)]
        write(f'gpu_count_scaling/configs/scaling_n{gpu_count}.json', 5, scaling)
        control = []
        for gpu_set in [devices] + CONTROL_SETS[gpu_count]:
            for arm in arms:
                control.append(cell(f"n{gpu_count}_g{gpu_set.replace(',', '')}_{arm}_auto", arm, gpu_set))
        write(f'gpu_count_scaling/configs/control_n{gpu_count}.json', 3, control)
    write_six_gpu_followup()
    print('configs written')


def write_six_gpu_followup():
    """Plan 02 follow-up E2.3 (added 2026-11-16 after the data): is the 6-GPU drop of the symmetric kernels
    caused by the GPU set, or by 2 GiB not splitting evenly into 6 parts? 2 GiB - 128 B = 6 x 357913920 B,
    and 357913920 is a multiple of 16 B, so every rank gets the same whole number of 16-byte packs."""
    gpu_sets = ['0,1,2,3,4,5', '1,2,3,4,5,6', '2,3,4,5,6,7', '0,1,2,5,6,7']
    sizes = {'s2g': '2147483648', 's2gm128': '2147483520'}
    cells = []
    for gpu_set in gpu_sets:
        for size_tag, size in sizes.items():
            for arm, ctas in (('ring', None), ('nvls', None), ('sym_multicast', None), ('sym_plain', 16)):
                entry = cell(f"n6_g{gpu_set.replace(',', '')}_{size_tag}_{arm}_{'auto' if ctas is None else f'c{ctas}'}", arm, gpu_set, ctas)
                entry['min_size'] = size
                entry['max_size'] = size
                cells.append(entry)
    write('gpu_count_scaling/configs/followup_n6_split.json', 3, cells)


if __name__ == '__main__':
    main()
