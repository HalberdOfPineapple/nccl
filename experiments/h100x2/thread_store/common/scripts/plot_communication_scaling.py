#!/usr/bin/env python3
"""Show the complete recorded scaling progression without joining different builds."""
import csv
from collections import defaultdict
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COMMON = Path(__file__).resolve().parents[1]
EXPERIMENT = COMMON.parent / 'observation/shared_resources'
CALIBRATION = EXPERIMENT / 'results/2026-10-27-saturation-gpu6-7'


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def draw(axis, rows, xfield, ratefield, label, color, marker, linestyle):
    groups = defaultdict(list)
    for row in rows:
        groups[int(float(row[xfield]))].append(float(row[ratefield]))
    positions = sorted(groups)
    medians = [statistics.median(groups[position]) for position in positions]
    lower = [median-min(groups[position]) for position, median in zip(positions, medians)]
    upper = [max(groups[position])-median for position, median in zip(positions, medians)]
    axis.errorbar(positions, medians, yerr=[lower,upper], label=label, color=color,
                  marker=marker, linestyle=linestyle, linewidth=1.3, markersize=6, capsize=3)
    return positions, medians


def main():
    baseline = [row for row in read_rows(EXPERIMENT/'results/2026-10-27-gpu6-7/runs.csv')
                if row['run'].startswith('baseline_') and '_g0_' in row['run']]
    screen = [row for row in read_rows(CALIBRATION/'runs.csv')
              if row['phase']=='concurrency' and row['run'].startswith('screen_') and row['traversal']=='0']
    final = [row for row in read_rows(CALIBRATION/'runs.csv')
             if row['phase']=='batch' and row['run'].startswith('confirm_')
             and row['payload_mib']=='2048' and row['batch']=='8']
    assert {int(row['threads']) for row in final if row['ctas']=='128'} == {512,768,1024}
    figure, axes = plt.subplots(1,2,figsize=(15,7.5),sharey=True)
    for threads, color in [(256,'#2878b5'),(512,'#54a4cf')]:
        draw(axes[0], [row for row in baseline if int(float(row['threads']))==threads], 'ctas','throughput_gbps',
             f'Initial build: T={threads}; 256 MiB; batch 1',color,'o','-')
    for threads, color in [(512,'#e6a23c'),(1024,'#c86a19')]:
        draw(axes[0], [row for row in screen if int(row['threads'])==threads], 'ctas','gbps',
             f'Expanded build: T={threads}; 256 MiB; batch 1',color,'s','--')
    draw(axes[0], [row for row in final if row['threads']=='1024'], 'ctas','gbps',
         'Batch build: T=1024; 2048 MiB; batch 8','#8057b3','D','-')
    draw(axes[0], [row for row in final if row['threads']=='512'], 'ctas','gbps',
         'Batch build: T=512; 2048 MiB; batch 8','#218557','*','none')
    draw(axes[1], [row for row in baseline if int(float(row['ctas']))==32], 'threads','throughput_gbps',
         'Initial build: C=32; 256 MiB; batch 1','#2878b5','o','-')
    draw(axes[1], [row for row in screen if row['ctas']=='128'], 'threads','gbps',
         'Expanded build: C=128; 256 MiB; batch 1','#c86a19','s','--')
    positions, rates = draw(axes[1], [row for row in final if row['ctas']=='128'], 'threads','gbps',
         'Batch build: C=128; 2048 MiB; batch 8','#218557','D','-')
    for threads, rate in zip(positions,rates):
        axes[1].annotate(f'128 × {threads}\n{rate:.2f} GB/s',(threads,rate),
                         xytext=(0,13),textcoords='offset points',ha='center',fontsize=9)
    axes[0].set(title='Vary communication CTA count C at fixed T',xlabel='Communication CTAs C',
                xticks=[4,8,16,32,64,96,128,132],xlim=(0,139))
    # Keep nearby endpoint ticks readable while retaining both measured counts.
    axes[0].tick_params(axis='x',labelrotation=45)
    axes[1].set(title='Vary threads per communication CTA T at fixed C',xlabel='Threads per communication CTA T',
                xticks=[64,128,256,512,768,1024],xlim=(0,1120))
    for axis in axes:
        axis.set_ylim(0,450)
        axis.set_ylabel('Completed communication throughput (GB/s)')
        axis.grid(axis='y',alpha=.2)
        axis.spines[['top','right']].set_visible(False)
        axis.legend(loc='upper left',bbox_to_anchor=(0,-.20),fontsize=9,frameon=False)
    figure.suptitle('Communication throughput scaling: initial sweep through final confirmation\nGPU 6 → 7; CTA-local stride; no co-runner; destination control 1 × T',fontsize=14)
    figure.subplots_adjust(top=.84,bottom=.32,left=.07,right=.97,wspace=.13)
    figure.text(.07,.015,'C × T = CTAs × threads/CTA. Lines connect only matching build, payload, batch, and fixed launch dimension.\nInitial and final points: median of five invocation medians, whiskers min–max. Expanded screen: one invocation per point.',fontsize=9)
    figure.savefig(CALIBRATION/'figures/scaling_overview.png',dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    main()
