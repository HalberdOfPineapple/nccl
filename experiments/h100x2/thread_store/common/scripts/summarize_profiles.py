#!/usr/bin/env python3
"""Validate timeline capture against NVTX bounds and produce a cross-GPU activity figure."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collect import summarize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    arguments=parser.parse_args()
    profiles=arguments.directory/'profiles'
    entries=[]
    traces=[]
    cases=[('healthy','Healthy reference'),('compute','Source arithmetic'),('source_memory','Source scalar reads'),('destination_memory','Destination scalar reads'),('vector_source_memory','Source vector reads'),('vector_destination_memory','Destination vector reads')]
    for name,title in cases:
        prefix=profiles/f'nsys_range_{name}'
        with sqlite3.connect(prefix.with_suffix('.sqlite')) as connection:
            ranges=connection.execute('select n.start,n.end,coalesce(n.text,s.value) from NVTX_EVENTS n left join StringIds s on n.textId=s.id where coalesce(n.text,s.value)=?',('measurement',)).fetchall()
            assert len(ranges)==1,(name,ranges)
            range_start,range_end,_=ranges[0]
            kernels=connection.execute('select k.deviceId,k.start,k.end,k.gridX,k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where s.value like ? and k.start>=? and k.end<=? order by k.deviceId',('%fusedPush%',range_start,range_end)).fetchall()
            assert len(kernels)==2,(name,kernels)
            assert {kernel[0] for kernel in kernels}=={4,5}
        observations=summarize(prefix.with_suffix('.json'))
        assert len(observations)==1
        observed=observations[0]
        entry={'case':name,'source_kernel_ms':(kernels[0][2]-kernels[0][1])/1e6,'destination_kernel_ms':(kernels[1][2]-kernels[1][1])/1e6,'device_completed_us':observed['completed_us'],'throughput_gbps':observed['throughput_gbps'],'co_rate':observed['co_rate'],'minimum_overlap_fraction':observed['minimum_overlap_fraction'],'nvtx_range_ms':(range_end-range_start)/1e6,'source_grid':kernels[0][3],'destination_grid':kernels[1][3]}
        entries.append(entry)
        traces.append((name,title,range_start,kernels))
    with (profiles/'timeline_summary.csv').open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=list(entries[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(entries)
    figure,axes=plt.subplots(3,2,figsize=(11,8),layout='constrained')
    for axis,(name,title,range_start,kernels) in zip(axes.flat,traces):
        color='#2878b5' if name=='healthy' else '#e1812c'
        for row,(device,started,ended,grid,threads) in enumerate(kernels):
            axis.barh(row,(ended-started)/1e6,left=(started-range_start)/1e6,height=.35,color=color)
        axis.set(yticks=[0,1],yticklabels=['Source GPU 4','Destination GPU 5'],xlabel='Time since host measurement range start (ms)',title=title,xlim=(0,None))
        axis.spines[['top','right']].set_visible(False)
    figure.suptitle('Nsight Systems: measured fused kernels overlap across GPUs\nCommunication: 32 CTAs × 256 threads/CTA; co-runner when enabled: 100 CTAs × 256 threads/CTA\nDestination control: 1 CTA × 256 threads/CTA; CUPTI timeline clock')
    figure.legend(handles=[Patch(color='#2878b5',label='Healthy reference'),Patch(color='#e1812c',label='Pressure active')],loc='outside lower center',ncol=2,frameon=False)
    figures=arguments.directory/'figures'
    figures.mkdir(exist_ok=True)
    figure.savefig(figures/'timelines.png',dpi=150)
    print(json.dumps(entries,indent=2))

if __name__=='__main__':
    main()
