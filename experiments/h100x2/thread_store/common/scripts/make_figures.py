#!/usr/bin/env python3
"""Render figures from derived tables; raw data remain unchanged."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import set_zero_based_y, zero_based_upper

BLUE = '#2878b5'
ORANGE = '#e1812c'


def median(rows, field):
    return statistics.median(float(row[field]) for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    arguments = parser.parse_args()
    rows = list(csv.DictReader((arguments.directory/'runs.csv').open()))
    pressure_names = ['compute', 'source_memory', 'destination_memory', 'vector_source_memory', 'vector_destination_memory']
    pressure_rows = [row for row in rows if any(row['run'].startswith(name + '_') for name in pressure_names) and ('_H_' in row['run'] or '_P_' in row['run'])]
    # Comparable pressure panels use the same full-scale bounds, including across binaries.
    pressure_limits = {
        field: zero_based_upper([float(row[field]) for row in pressure_rows if row[field] != ''])
        for field in ['throughput_gbps', 'issue_us', 'ordering_us']
    }
    arithmetic_limit = zero_based_upper([float(row['co_rate']) for row in rows if float(row['kind']) == 1])
    read_limit = zero_based_upper([float(row['co_rate']) for row in rows if float(row['kind']) in [2, 3]])
    figures = arguments.directory/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1,'figure.dpi':150})
    figure,axes = plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    for axis,prefix,values,xlabel in [(axes[0],'baseline_ctas_', [4,8,16,32], 'Communication CTA count\n(256 threads/CTA; CTA-local stride)'),(axes[1],'baseline_threads_', [64,128,256,512], 'Threads per communication CTA\n(32 communication CTAs; CTA-local stride)')]:
        medians = []
        for value in values:
            samples = [row for row in rows if row['run'].startswith(f'{prefix}{value}_R_')]
            if not samples:
                continue
            axis.scatter([value]*len(samples),[float(row['throughput_gbps']) for row in samples],color=BLUE,s=12,alpha=.45)
            medians.append((value,median(samples,'throughput_gbps')))
        axis.plot([entry[0] for entry in medians],[entry[1] for entry in medians],color=BLUE,marker='o',markersize=3)
        axis.set(xlabel=xlabel,ylabel='Completed-transfer throughput (GB/s)',xticks=values)
    traversal = [('CTA-local stride','baseline_ctas_32_R_'),('Global stride','baseline_global_stride_R_')]
    for index,(label,prefix) in enumerate(traversal):
        samples=[float(row['throughput_gbps']) for row in rows if row['run'].startswith(prefix)]
        if samples:
            axes[2].scatter([index]*len(samples),samples,color=BLUE,s=14,alpha=.6)
            axes[2].plot([index-.15,index+.15],[statistics.median(samples)]*2,color=BLUE)
    axes[2].set(xticks=[0,1],xticklabels=[entry[0] for entry in traversal],ylabel='Completed-transfer throughput (GB/s)',xlabel='Communication: 32 CTAs × 256 threads/CTA')
    baseline_limit = zero_based_upper([float(row['throughput_gbps']) for row in rows if row['run'].startswith('baseline_')])
    for axis in axes:
        set_zero_based_y(axis, baseline_limit)
    figure.suptitle('Communication-only reference: concurrency and traversal')
    figure.savefig(figures/'baseline.png')
    plt.close(figure)
    for name,title in [('compute','Arithmetic on separate source SMs'),('source_memory','Streaming reads on source SMs'),('destination_memory','Scalar streaming reads on destination SMs'),('vector_source_memory','Vector streaming reads on source SMs'),('vector_destination_memory','Vector streaming reads on destination SMs')]:
        figure,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        observations=[('throughput_gbps','Completed-transfer throughput (GB/s)'),('issue_us','Median CTA issue interval (µs)'),('ordering_us','Median CTA ordering interval (µs)'),('co_rate','Arithmetic rate (TFLOP/s)' if name=='compute' else 'Logical read rate (GB/s)')]
        for axis,(field,ylabel) in zip(axes.flat,observations):
            for side,color,label in [('H',BLUE,'Healthy reference'),('P',ORANGE,'Perturbed')]:
                if field=='co_rate' and side=='H':
                    continue
                medians=[]
                for count in ([8,32,100,131] if name=='vector_destination_memory' else [8,32,100]):
                    selected=[row for row in rows if row['run'].startswith(f'{name}_{count}_{side}_')]
                    if not selected:
                        continue
                    axis.scatter([count]*len(selected),[float(row[field]) for row in selected],color=color,s=15,alpha=.5)
                    medians.append((count,median(selected,field)))
                if medians:
                    axis.plot([item[0] for item in medians],[item[1] for item in medians],color=color,marker='o',markersize=3,label=label)
            axis.set(xlabel='Co-runner CTA count (256 threads/CTA)',ylabel=ylabel,xticks=([8,32,100,131] if name=='vector_destination_memory' else [8,32,100]))
            upper = (arithmetic_limit if name == 'compute' else read_limit) if field == 'co_rate' else pressure_limits[field]
            set_zero_based_y(axis, upper)
            if field!='co_rate' and axis.get_legend_handles_labels()[0]:
                axis.legend(frameon=False)
        figure.suptitle(title+'\nCommunication: 32 CTAs × 256 threads/CTA; CTA-local stride\nDots: invocation medians; lines: medians across five invocations')
        figure.savefig(figures/f'{name}.png')
        plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for axis,name in zip(axes,['source_memory','destination_memory']):
        for count,color,label in [(8,BLUE,'8 × 256'),(32,ORANGE,'32 × 256'),(100,'#555555','100 × 256')]:
            selected=[row for row in rows if row['run'].startswith(f'{name}_{count}_P_')]
            solo=[row for row in rows if row['run']==f'{name}_{count}_solo']
            if selected:
                axis.scatter([median(selected,'co_rate')],[median(selected,'throughput_gbps')],color=color,label=label,s=30)
            if solo:
                # Standalone rate is context, not plotted as a communication data point.
                axis.axvline(median(solo,'co_rate'),color=color,alpha=.35,linewidth=.7,linestyle=':')
        axis.set(xlabel='Logical co-runner read rate (GB/s)',ylabel='Completed-transfer throughput (GB/s)',title='Source reads' if name=='source_memory' else 'Destination reads')
        set_zero_based_y(axis, pressure_limits['throughput_gbps'])
        if axis.get_legend_handles_labels()[0]:
            axis.legend(frameon=False,title='Read CTAs × threads/CTA')
    figure.suptitle('Achieved pressure versus communication progress\nCommunication: 32 CTAs × 256 threads/CTA; CTA-local stride\nDotted vertical lines: standalone co-runner read rates')
    figure.savefig(figures/'memory_comparison.png')
    plt.close(figure)
    for figure_name,scenarios in [
        ('scalar_issue_progress',[('compute','Source arithmetic'),('source_memory','Source scalar reads'),('destination_memory','Destination scalar reads')]),
        ('vector_issue_progress',[('vector_source_memory','Source vector reads'),('vector_destination_memory','Destination vector reads')])]:
        figure,axes=plt.subplots(1,len(scenarios),figsize=(5*len(scenarios),3.8),layout='constrained',squeeze=False)
        any_data=False
        for axis,(name,title) in zip(axes.flat,scenarios):
            for side,color,label in [('H',BLUE,'Healthy reference'),('P',ORANGE,'Perturbed')]:
                curves=[]
                for row in rows:
                    if not row['run'].startswith(f'{name}_100_{side}_'):
                        continue
                    payload=json.loads((arguments.directory/(row['run']+'.json')).read_text())
                    iteration_curves=[]
                    for records in payload['ranks'][0]['iterations']:
                        communication=[record for record in records if record['role']==0]
                        iteration_curves.append([statistics.median((record['checkpoints_ns'][checkpoint]-record['start_ns'])/1e6 for record in communication) for checkpoint in range(8)])
                    curves.append([statistics.median(curve[checkpoint] for curve in iteration_curves) for checkpoint in range(8)])
                if curves:
                    any_data=True
                    for curve in curves:
                        axis.plot(range(1,9),curve,color=color,alpha=.25,linewidth=.6)
                    axis.plot(range(1,9),[statistics.median(curve[checkpoint] for curve in curves) for checkpoint in range(8)],color=color,label=label)
            axis.set(xlabel='Software issue progress per CTA (MiB)',ylabel='Elapsed since CTA start (ms)',title=title,xticks=[2,4,6,8])
            set_zero_based_y(axis, pressure_limits['issue_us'] / 1000)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(frameon=False)
        figure.suptitle('Issue progress: communication 32 CTAs × 256 threads/CTA; CTA-local stride\nCo-runner: 100 CTAs × 256 threads/CTA\nThin lines: invocation medians; no remote-arrival timing')
        if any_data:
            figure.savefig(figures/f'{figure_name}.png')
        plt.close(figure)
    overhead=[row for row in rows if row['run'].startswith('instrumentation_overhead_')]
    if overhead:
        figure,axis=plt.subplots(figsize=(5,3.5),layout='constrained')
        for index,(side,label,color) in enumerate([('H','Minimal boundary probes',BLUE),('P','Detailed phase/checkpoint probes',ORANGE)]):
            values=[float(row['throughput_gbps']) for row in overhead if f'_{side}_' in row['run']]
            axis.scatter([index]*len(values),values,color=color,s=20,label=label)
        axis.set(xticks=[0,1],xticklabels=['Minimal','Detailed'],ylabel='Completed-transfer throughput (GB/s)',title='Measurement overhead\nCommunication: 32 CTAs × 256 threads/CTA')
        set_zero_based_y(axis, pressure_limits['throughput_gbps'])
        axis.legend(frameon=False,fontsize=8)
        figure.savefig(figures/'instrumentation_overhead.png')
        plt.close(figure)
    print(figures)

if __name__=='__main__':
    main()
