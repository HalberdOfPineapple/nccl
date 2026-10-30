#!/usr/bin/env python3
"""Summarize the GPU 6 -> 7 campaign without pooling independent configurations."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collect import summarize, percentile, write_csv
from plot_style import set_zero_based_y, zero_based_upper

BLUE = '#2878b5'
ORANGE = '#e1812c'


def median(values):
    return statistics.median(values)


def paired_summary(references, pressures, metric):
    healthy = [row[metric] for row in references]
    perturbed = [row[metric] for row in pressures]
    reference_iqr = percentile(healthy, .75) - percentile(healthy, .25)
    return dict(reference=median(healthy), pressure=median(perturbed),
                paired_percent=median([(pressure / reference - 1) * 100 for reference, pressure in zip(healthy, perturbed)]),
                reference_iqr=reference_iqr,
                effect_iqr=(median(perturbed)-median(healthy))/reference_iqr if reference_iqr else None,
                pairs=len(healthy))


def run_summary(path):
    rows = summarize(path)
    result = dict(rows[0])
    result.pop("iteration", None)
    for key, value in result.copy().items():
        if isinstance(value, (float, int)):
            result[key] = median([row[key] for row in rows])
    result['completed_max_us'] = max(row['completed_us'] for row in rows)
    result['overlap_min'] = min(row['minimum_overlap_fraction'] for row in rows)
    return result


def prettify(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', alpha=.15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    arguments = parser.parse_args()
    directory = arguments.directory
    raw = directory/'raw'
    figures = directory/'figures'
    figures.mkdir(exist_ok=True)
    paths = [path for path in sorted(raw.glob('*.json')) if not any(path.name.endswith(suffix) for suffix in ('.before.json','.after.json','.telemetry.json'))]
    runs = {path.stem: run_summary(path) for path in paths}
    write_csv(directory/'runs.csv', list(runs.values()))
    baseline = []
    for count, threads, traversal in [(count,256,0) for count in (4,8,16,32,64)]+[(32,threads,0) for threads in (64,128,512)]+[(16,512,0),(64,512,0),(32,256,1)]:
        samples = [runs[f'baseline_c{count}_t{threads}_g{traversal}_R{repetition}'] for repetition in range(1,6)]
        baseline.append(dict(ctas=count, threads=threads, global_stride=traversal,
                             gbps=median([row['throughput_gbps'] for row in samples]),
                             minimum_gbps=min(row['throughput_gbps'] for row in samples),
                             maximum_gbps=max(row['throughput_gbps'] for row in samples)))
    comparisons = []
    for threads in (256,512):
        for count in (16,32,64):
            for memory in (8,32,100,131):
                prefix=f'mixed_c{count}_t{threads}_m{memory}'
                healthy=[runs[f'{prefix}_H{repetition}'] for repetition in range(1,6)]
                pressure=[runs[f'{prefix}_P{repetition}'] for repetition in range(1,6)]
                entry=dict(ctas=count,threads=threads,read_ctas=memory,
                           **paired_summary(healthy,pressure,'throughput_gbps'))
                for metric in ('completed_us','issue_us','ordering_us','ack_wait_us','issue_spread_us','co_rate','co_active_us','completed_max_us'):
                    entry[f'reference_{metric}']=median([row[metric] for row in healthy])
                    entry[f'pressure_{metric}']=median([row[metric] for row in pressure])
                entry['reference_max_transfer_us']=max(row['completed_max_us'] for row in healthy)
                entry['pressure_max_transfer_us']=max(row['completed_max_us'] for row in pressure)
                entry['overlap_min']=min(row['overlap_min'] for row in pressure)
                entry['slower_pairs']=sum(pressure_row['throughput_gbps']<healthy_row['throughput_gbps'] for healthy_row,pressure_row in zip(healthy,pressure))
                comparisons.append(entry)
    # Keep phase observations separate: medians across CTAs are not additive.
    figure,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    strongest_rows=[row for row in comparisons if row['read_ctas']==131]
    for axis,(metric,label) in zip(axes.flat,[('issue_us','Median CTA issue duration (µs)'),('ordering_us','Median CTA ordering duration (µs)'),('ack_wait_us','Median CTA acknowledgement wait (µs)'),('issue_spread_us','CTA issue-end spread (µs)')]):
        for prefix,color,title,shift in [('reference',BLUE,'Copy only',-.18),('pressure',ORANGE,'Copy + reads',.18)]:
            axis.bar([position+shift for position in range(len(strongest_rows))],[row[f'{prefix}_{metric}'] for row in strongest_rows],width=.36,color=color,label=title)
        axis.set(xticks=range(len(strongest_rows)),xticklabels=[f"{row['ctas']}×{row['threads']}" for row in strongest_rows],xlabel='Communication CTAs × threads/CTA',ylabel=label)
        set_zero_based_y(axis,zero_based_upper([row[f'{prefix}_{metric}'] for row in strongest_rows for prefix in ('reference','pressure')]))
        prettify(axis)
    figure.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
    figure.suptitle('Communication timing with destination reads: 131 CTAs × matching group threads\nSource-local timings; phase medians must not be summed')
    figure.savefig(figures/'communication_phases.png',dpi=150)
    plt.close(figure)
    checkpoint_rows=[]
    figure,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for axis,(count,threads) in zip(axes,[(16,256),(64,512)]):
        for arm,color,title in [('H',BLUE,'Copy only'),('P',ORANGE,'Copy + reads')]:
            segment_invocations=[[] for _ in range(8)]
            for repetition in range(1,6):
                data=json.loads((raw/f'mixed_c{count}_t{threads}_m131_{arm}{repetition}.json').read_text())
                segments=[[] for _ in range(8)]
                for transfer in data['ranks'][0]['iterations']:
                    for record in transfer:
                        if record['role']!=0:
                            continue
                        previous=record['start_ns']
                        for index,checkpoint in enumerate(record['checkpoints_ns']):
                            if checkpoint<=previous or checkpoint>record['issue_end_ns']:
                                raise RuntimeError('Invalid software issue checkpoint')
                            segments[index].append((checkpoint-previous)/1000)
                            previous=checkpoint
                for index,values in enumerate(segments):
                    segment_invocations[index].append(median(values))
            for index,values in enumerate(segment_invocations):
                checkpoint_rows.append(dict(ctas=count,threads=threads,read_ctas=0 if arm=='H' else 131,segment=index+1,bytes_per_cta_segment=268435456/count/8,median_us=median(values),invocation_min_us=min(values),invocation_max_us=max(values)))
            axis.plot(range(1,9),[median(values) for values in segment_invocations],'o-',color=color,label=title,linewidth=1)
        axis.set(title=f'Communication: {count} CTAs × {threads} threads/CTA',xlabel='Equal-work issue segment (1–8)',ylabel='Median segment duration (µs)',xticks=range(1,9))
        prettify(axis)
    upper=zero_based_upper([row['invocation_max_us'] for row in checkpoint_rows])
    for axis in axes:
        set_zero_based_y(axis,upper)
    figure.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
    figure.suptitle('Software issue progress; destination reads: 131 CTAs × group threads\nSegments measure issuing-loop progress, not remote arrival')
    figure.savefig(figures/'issue_progress.png',dpi=150)
    plt.close(figure)
    write_csv(directory/'checkpoint_summary.csv',checkpoint_rows)
    solo=[]
    for threads in (256,512):
        for memory in (8,32,100,131):
            samples=[runs[f'solo_t{threads}_m{memory}_R{repetition}']['co_rate'] for repetition in range(1,6)]
            solo.append(dict(threads=threads,read_ctas=memory,gbps=median(samples),minimum_gbps=min(samples),maximum_gbps=max(samples)))
    mutual=[]
    for count,threads,memory,duration in json.loads((directory/'followup_selection_summary.json').read_text()):
        prefix=f'mutual_c{count}_t{threads}_m{memory}'
        healthy=[runs[f'{prefix}_H{repetition}'] for repetition in range(1,6)]
        pressure=[runs[f'{prefix}_P{repetition}'] for repetition in range(1,6)]
        mutual.append(dict(ctas=count,threads=threads,read_ctas=memory,solo_ns=duration,
                           **paired_summary(healthy,pressure,'co_rate'),
                           reference_active_us=median([row['co_active_us'] for row in healthy]),
                           pressure_active_us=median([row['co_active_us'] for row in pressure]),
                           copy_gbps=median([row['throughput_gbps'] for row in pressure])))
    probes=[]
    for count,threads in ((16,256),(64,512)):
        minimal=[runs[f'probe_c{count}_t{threads}_d0_R{repetition}'] for repetition in range(1,6)]
        detailed=[runs[f'probe_c{count}_t{threads}_d1_R{repetition}'] for repetition in range(1,6)]
        probes.append(dict(ctas=count,threads=threads,read_ctas=131,**paired_summary(minimal,detailed,'throughput_gbps')))
    summary=dict(baseline=baseline,communication=comparisons,reads_alone=solo,destination_progress=mutual,probe_cost=probes,
                 invocations=len(runs),retained_transfers=sum(12 for row in runs.values() if row['communication']),
                 standalone_windows=sum(5 if name.startswith('solo_') else 12 for name,row in runs.items() if not row['communication']))
    (directory/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name,rows in [('communication',comparisons),('baseline',baseline),('reads_alone',solo),('destination_progress',mutual),('probe_cost',probes)]:
        write_csv(directory/f'{name}.csv',rows)
    # Baseline points are invocation medians; ranges show min/max across five invocations.
    figure,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    upper=zero_based_upper([row['maximum_gbps'] for row in baseline])
    for axis,threads in zip(axes,(256,512)):
        selected=sorted([row for row in baseline if row['threads']==threads and not row['global_stride']],key=lambda row:row['ctas'])
        axis.errorbar([row['ctas'] for row in selected],[row['gbps'] for row in selected],yerr=[[row['gbps']-row['minimum_gbps'] for row in selected],[row['maximum_gbps']-row['gbps'] for row in selected]],fmt='o-',color=BLUE,capsize=3)
        axis.set(title=f'Communication: C CTAs × {threads} threads/CTA',xlabel='Communication CTA count C',xticks=[row['ctas'] for row in selected],ylabel='Completed communication throughput (GB/s)')
        set_zero_based_y(axis,upper)
        prettify(axis)
    figure.suptitle('Communication alone: fixed 256 MiB, CTA-local stride; GPU 6 → 7')
    figure.savefig(figures/'communication_alone.png',dpi=150)
    plt.close(figure)
    figure,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    upper=zero_based_upper([row[key] for row in comparisons for key in ('reference','pressure')])
    for axis,(count,threads) in zip(axes.flat,[(count,threads) for threads in (256,512) for count in (16,32,64)]):
        selected=[row for row in comparisons if row['ctas']==count and row['threads']==threads]
        for key,color,label in [('reference',BLUE,'Copy only (fresh paired reference)'),('pressure',ORANGE,'Copy + destination reads')]:
            axis.plot([row['read_ctas'] for row in selected],[row[key] for row in selected],marker='o' if key=='reference' else 'x',markerfacecolor='none',markersize=7 if key=='reference' else 5,color=color,label=label,linewidth=1)
        axis.set(title=f'Communication: {count} CTAs × {threads} threads/CTA',xlabel=f'Destination read CTAs (× {threads} threads/CTA)',xticks=[row['read_ctas'] for row in selected],ylabel='Communication throughput (GB/s)')
        set_zero_based_y(axis,upper)
        prettify(axis)
    figure.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
    figure.suptitle('Primary outcome: communication under destination read pressure\n256 MiB; CTA-local stride; control: 1 CTA × the group thread count')
    figure.savefig(figures/'communication_with_reads.png',dpi=150)
    plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    upper=zero_based_upper([row['maximum_gbps'] for row in solo])
    for axis,threads in zip(axes,(256,512)):
        selected=[row for row in solo if row['threads']==threads]
        axis.errorbar([row['read_ctas'] for row in selected],[row['gbps'] for row in selected],yerr=[[row['gbps']-row['minimum_gbps'] for row in selected],[row['maximum_gbps']-row['gbps'] for row in selected]],fmt='o-',color=BLUE,capsize=3)
        axis.set(title=f'Destination reads: M CTAs × {threads} threads/CTA',xlabel='Read CTA count M',xticks=[row['read_ctas'] for row in selected],ylabel='Logical read rate (GB/s)')
        set_zero_based_y(axis,upper)
        prettify(axis)
    figure.suptitle('Destination reads alone: 1 GiB buffer; 20 ms control window\nSource control: 32 CTAs × group threads; destination control: 1 CTA × group threads; copy disabled')
    figure.savefig(figures/'reads_alone.png',dpi=150)
    plt.close(figure)
    figure,axis=plt.subplots(figsize=(max(10,len(mutual)*.65),5),layout='constrained')
    positions=list(range(len(mutual)))
    axis.bar([position-.18 for position in positions],[row['reference'] for row in mutual],width=.36,color=BLUE,label='Reads only (matched duration)')
    axis.bar([position+.18 for position in positions],[row['pressure'] for row in mutual],width=.36,color=ORANGE,label='Reads + copy')
    axis.set(xticks=positions,xticklabels=[f"Copy {row['ctas']}×{row['threads']}\nReads {row['read_ctas']}×{row['threads']}" for row in mutual],ylabel='Logical destination read rate (GB/s)',title='Secondary outcome: destination read progress\nLabels give CTAs × threads/CTA; copy disabled in the reads-only arm')
    axis.tick_params(axis='x',labelrotation=45)
    set_zero_based_y(axis,zero_based_upper([row[key] for row in mutual for key in ('reference','pressure')]))
    axis.legend()
    prettify(axis)
    figure.savefig(figures/'destination_progress.png',dpi=150)
    plt.close(figure)
    figure,axis=plt.subplots(figsize=(7,4),layout='constrained')
    for position,row in enumerate(probes):
        axis.bar(position-.18,row['reference'],width=.36,color=BLUE,label='Minimal probes' if position==0 else None)
        axis.bar(position+.18,row['pressure'],width=.36,color=ORANGE,label='Detailed probes' if position==0 else None)
    axis.set(xticks=range(len(probes)),xticklabels=[f"Copy {row['ctas']}×{row['threads']}; reads 131×{row['threads']}" for row in probes],ylabel='Communication throughput (GB/s)',title='Probe comparison: CTAs × threads/CTA')
    set_zero_based_y(axis,zero_based_upper([row[key] for row in probes for key in ('reference','pressure')]))
    axis.legend()
    prettify(axis)
    figure.savefig(figures/'probe_cost.png',dpi=150)
    print(json.dumps({'invocations':len(runs),'communication_change_range_percent':[min(row['paired_percent'] for row in comparisons),max(row['paired_percent'] for row in comparisons)],'followup_conditions':len(mutual)},indent=2))


if __name__=='__main__':
    main()
