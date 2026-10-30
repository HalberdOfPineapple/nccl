#!/usr/bin/env python3
"""Capture endpoint timelines after primary measurements; preserve GPU 6/7 overlap."""
import argparse
import fcntl
import hashlib
from contextlib import ExitStack
import json
from pathlib import Path
import sqlite3
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collect import summarize
from plot_style import zero_based_upper
from run_one import ensure_idle
from run_saturated_contention import COMMON, CONFIGS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    arguments=parser.parse_args()
    directory=arguments.directory
    provenance = json.loads((directory/'build_provenance.json').read_text())
    assert all(hashlib.sha256((COMMON/name).read_bytes()).hexdigest() == digest for name, digest in provenance.items())
    profiles=directory/'raw/profiles'
    profiles.mkdir(parents=True,exist_ok=True)
    cases=[(128,512,0),(128,512,131)]
    entries=[]
    traces=[]
    for count,threads,memory in cases:
        name=f'timeline_c{count}_t{threads}_m{memory}'
        prefix=profiles/name
        config=CONFIGS/f'{name}.cfg'
        values=dict(source_gpu=6,destination_gpu=7,ctas=count,threads=threads,kind=3 if memory else 0,co_ctas=memory,vector_reads=1,pacing=0,detailed=1,communication=1,global_stride=0,warmup=2,iterations=1,payload_mib=2048,copy_batch=8)
        config.write_text(''.join(f'{key}={value}\n' for key,value in values.items()))
        if not prefix.with_suffix('.nsys-rep').exists():
            with ExitStack() as locks:
                for device in (6,7):
                    handle = locks.enter_context(open(f'/tmp/nccl_thread_store_gpu_{device}.lock', 'a'))
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                ensure_idle('6,7')
                with prefix.with_suffix('.log').open('w') as output:
                    subprocess.run(['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--force-overwrite=false',f'--output={prefix}',str(COMMON/'bin/push_harness'),str(config),str(prefix.with_suffix('.json')),'--profile-range'],stdout=output,stderr=subprocess.STDOUT,timeout=60,check=True)
        if not prefix.with_suffix('.sqlite').exists():
            subprocess.run(['nsys','export','--type=sqlite',f'--output={prefix.with_suffix(".sqlite")}',str(prefix.with_suffix('.nsys-rep'))],timeout=60,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        with sqlite3.connect(prefix.with_suffix('.sqlite')) as connection:
            ranges=connection.execute('select n.start,n.end,coalesce(n.text,s.value) from NVTX_EVENTS n left join StringIds s on n.textId=s.id where coalesce(n.text,s.value)=?',('measurement',)).fetchall()
            if len(ranges)!=1:
                raise RuntimeError(f'Unexpected ranges: {name}')
            started,ended,_=ranges[0]
            kernels=connection.execute('select k.deviceId,k.start,k.end,k.gridX,k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where s.value like ? and k.start>=? and k.end<=? order by k.deviceId',('%fusedPush%',started,ended)).fetchall()
        if len(kernels)!=2 or {kernel[0] for kernel in kernels}!={6,7}:
            raise RuntimeError(f'Unexpected endpoint kernels: {kernels}')
        if kernels[0][3:]!=(count,threads) or kernels[1][3:]!=(1+memory,threads):
            raise RuntimeError(f'Unexpected geometry: {kernels}')
        data = json.loads(prefix.with_suffix('.json').read_text())
        assert data['verified'] and data['bytes'] == 2048*1024*1024 and data['copy_batch'] == 8
        observations=summarize(prefix.with_suffix('.json'))
        if len(observations)!=1 or (memory and observations[0]['minimum_overlap_fraction']<1):
            raise RuntimeError(f'Invalid device overlap: {name}')
        if min(kernel[2] for kernel in kernels)<=max(kernel[1] for kernel in kernels):
            raise RuntimeError(f'No cross-GPU kernel overlap: {name}')
        entries.append(dict(ctas=count,threads=threads,read_ctas=memory,co_rate=observations[0]['co_rate'],throughput_gbps=observations[0]['throughput_gbps'],minimum_overlap_fraction=observations[0]['minimum_overlap_fraction'],source_kernel_ms=(kernels[0][2]-kernels[0][1])/1e6,destination_kernel_ms=(kernels[1][2]-kernels[1][1])/1e6))
        traces.append((started,kernels,count,threads,memory))
        print(name,'validated',flush=True)
    figure,axes=plt.subplots((len(traces)+1)//2,2,figsize=(12,3*((len(traces)+1)//2)),layout='constrained')
    for axis,(start,kernels,count,threads,memory) in zip(axes.flat,traces):
        for row,(_,started,ended,_,_) in enumerate(kernels):
            axis.barh(row,(ended-started)/1e6,left=(started-start)/1e6,height=.35,color='#e1812c' if memory else '#2878b5')
        axis.set(yticks=[0,1],yticklabels=['Source GPU 6','Destination GPU 7'],xlabel='Time since measurement range start (ms)',title=f'Copy {count}×{threads}; reads {memory}×{threads}',xlim=(0,None))
        axis.spines[['top','right']].set_visible(False)
    upper=zero_based_upper([(kernel[2]-start)/1e6 for start,kernels,_,_,_ in traces for kernel in kernels])
    for axis in axes.flat:
        axis.set_xlim(0,upper)
    figure.suptitle('Separate timeline passes: CTAs × threads/CTA; 2048 MiB; batch 8; CTA-local stride\nDestination control: 1 × 512; device probes verify read-role coverage')
    figure.savefig(directory/'figures/timelines.png',dpi=150)
    (directory/'timeline_summary.json').write_text(json.dumps(entries,indent=2)+'\n')


if __name__=='__main__':
    main()
