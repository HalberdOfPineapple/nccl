#!/usr/bin/env python3
"""Baseline sweeps, pressure calibration, paired comparisons, and standalone controls."""
from pathlib import Path
import subprocess

COMMON = Path(__file__).resolve().parents[1]
EXPERIMENT = COMMON.parent / 'observation/shared_resources'
CONFIGS = EXPERIMENT / 'configs'
RESULTS = EXPERIMENT / 'results/2026-10-27'


def run_one(config, label):
    output = RESULTS / f'{label}.json'
    if output.exists():
        return
    subprocess.run(['python3', str(COMMON/'scripts/run_one.py'), str(CONFIGS/f'{config}.cfg'), str(output)], check=True)


def main():
    # Five separate launches per baseline setting, each with warm-up and 12 retained transfers.
    baseline_configs = [f'baseline_ctas_{value}' for value in [4,8,16,32]]
    baseline_configs += [f'baseline_threads_{value}' for value in [64,128,256,512]]
    baseline_configs += ['baseline_global_stride']
    for repetition in range(1,6):
        for config in baseline_configs:
            run_one(config, f'{config}_R_{repetition}')
    for name in ['compute','source_memory','destination_memory']:
        for count in [8,16,32,64,100]:
            run_one(f'{name}_{count}', f'calibration_{name}_{count}')
        for count in [8,32,100]:
            subprocess.run(['bash', str(COMMON/'scripts/run_pair.sh'), str(RESULTS), '5', str(CONFIGS/'healthy.cfg'), str(CONFIGS/f'{name}_{count}.cfg'), f'{name}_{count}'], check=True)
            run_one(f'{name}_{count}_solo', f'{name}_{count}_solo')
    for name in ['source_memory','destination_memory']:
        for pacing in [0,64,256,1024]:
            run_one(f'{name}_100_pacing_{pacing}',f'pacing_{name}_{pacing}')
    subprocess.run(['bash',str(COMMON/'scripts/run_pair.sh'),str(RESULTS),'5',str(CONFIGS/'healthy_minimal.cfg'),str(CONFIGS/'healthy.cfg'),'instrumentation_overhead'],check=True)
    for name in ['compute','source_memory','destination_memory']:
        run_one(f'{name}_100_minimal',f'minimal_{name}_100')
    run_one('healthy','return_to_baseline')
    subprocess.run(['python3',str(COMMON/'scripts/collect.py'),str(RESULTS)],check=True)

if __name__ == '__main__':
    main()
