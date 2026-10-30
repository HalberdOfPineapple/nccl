#!/usr/bin/env python3
"""Present communication throughput first, followed by secondary read-progress controls."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collect import percentile,summarize,write_csv
from plot_style import set_zero_based_y, zero_based_upper


def plot_communication_throughput(directory):
    """Use paired copy-only/copy-with-reads runs; a no-copy arm is not a copy reference."""
    with (directory/'runs.csv').open() as source:
        runs=list(csv.DictReader(source))
    comparisons=[]
    figure,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    upper=zero_based_upper([float(row['throughput_gbps']) for row in runs if row['run'].startswith('vector_destination_memory_')])
    for axis,count in zip(axes,[100,131]):
        samples={}
        for side in ['H','P']:
            samples[side]={int(row['run'].rsplit('_',1)[1]):float(row['throughput_gbps']) for row in runs if row['run'].startswith(f'vector_destination_memory_{count}_{side}_')}
            assert len(samples[side])==5,(count,side)
            values=list(samples[side].values())
            index=0 if side=='H' else 1
            color='#2878b5' if side=='H' else '#e1812c'
            axis.scatter([index]*len(values),values,color=color,s=25,alpha=.65,label='Copy alone' if side=='H' else 'Copy with destination reads')
            axis.plot([index-.15,index+.15],[statistics.median(values)]*2,color=color,linewidth=1)
        assert samples['H'].keys()==samples['P'].keys()
        healthy=statistics.median(samples['H'].values())
        perturbed=statistics.median(samples['P'].values())
        healthy_iqr=percentile(list(samples['H'].values()),.75)-percentile(list(samples['H'].values()),.25)
        comparisons.append({'destination_read_ctas':count,'threads_per_cta':256,'communication_ctas':32,'pairs':5,'copy_alone_gbps':healthy,'copy_with_reads_gbps':perturbed,'paired_change_percent':statistics.median(100*(samples['P'][pair]/samples['H'][pair]-1) for pair in samples['H']),'effect_iqr':(perturbed-healthy)/healthy_iqr if healthy_iqr else None})
        axis.set(xticks=[0,1],xticklabels=['No destination reads','With destination reads'],ylabel='Completed-transfer throughput (GB/s)',title=f'Destination reads when enabled:\n{count} CTAs × 256 threads/CTA')
        set_zero_based_y(axis,upper)
        axis.spines[['top','right']].set_visible(False)
        axis.legend(frameon=False,fontsize=8)
    figure.suptitle('Primary outcome: communication throughput\nCommunication: 32 CTAs × 256 threads/CTA; CTA-local stride\nFive original interleaved copy-only / copy-with-reads pairs per configuration')
    figure.savefig(directory/'figures/mutual_communication.png',dpi=150)
    plt.close(figure)
    (directory/'mutual_interference/communication_summary.json').write_text(json.dumps(comparisons,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    arguments=parser.parse_args()
    plot_communication_throughput(arguments.directory)
    directory=arguments.directory/'mutual_interference'
    rows=[]
    for path in sorted(directory.glob('*.json')):
        rows.extend(summarize(path))
    write_csv(directory/'transfers.csv',rows)
    results=[]
    read_limit = zero_based_upper([row['co_rate'] for row in rows])
    figure,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for axis,count in zip(axes,[100,131]):
        values={'H':[],'P':[]}
        active={'H':[],'P':[]}
        throughput=[]
        for side in ['H','P']:
            for pair in range(1,6):
                selected=[row for row in rows if row['run']==f'destination_workload_{count}_{side}_{pair}']
                assert len(selected)==12,(count,side,pair,len(selected))
                values[side].append(statistics.median(row['co_rate'] for row in selected))
                active[side].append(statistics.median(row['co_active_us'] for row in selected))
                if side=='P':
                    throughput.append(statistics.median(row['throughput_gbps'] for row in selected))
            index=0 if side=='H' else 1
            color='#2878b5' if side=='H' else '#e1812c'
            axis.scatter([index]*5,values[side],color=color,s=25,alpha=.65,label='Reads alone' if side=='H' else 'Reads with incoming stores')
            axis.plot([index-.15,index+.15],[statistics.median(values[side])]*2,color=color,linewidth=1)
        healthy_median=statistics.median(values['H'])
        perturbed_median=statistics.median(values['P'])
        healthy_iqr=percentile(values['H'],.75)-percentile(values['H'],.25)
        results.append({'co_ctas':count,'pairs':5,'read_alone_gbps':healthy_median,'with_copy_gbps':perturbed_median,'paired_change_percent':statistics.median(100*(perturbed/healthy-1) for healthy,perturbed in zip(values['H'],values['P'])),'effect_iqr':(perturbed_median-healthy_median)/healthy_iqr if healthy_iqr else None,'read_alone_active_us':statistics.median(active['H']),'with_copy_active_us':statistics.median(active['P']),'copy_throughput_gbps':statistics.median(throughput),'read_alone_range':values['H'],'with_copy_range':values['P']})
        axis.set(xticks=[0,1],xticklabels=['Reads alone','With incoming stores'],ylabel='Logical destination read rate (GB/s)',title=f'Destination reads: {count} CTAs × 256 threads/CTA',ylim=(0,None))
        set_zero_based_y(axis, read_limit)
        axis.spines[['top','right']].set_visible(False)
        axis.legend(frameon=False,fontsize=8)
    figure.suptitle('Secondary observation: destination read progress\nCopy when enabled: 32 CTAs × 256 threads/CTA; CTA-local stride\nFive interleaved pairs; standalone control duration: 3.3 ms')
    figure.savefig(arguments.directory/'figures/mutual_interference.png',dpi=150)
    (directory/'summary.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
