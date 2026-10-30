#!/usr/bin/env python3
"""Finish communication-only calibration before authorizing a new contention grid."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from collect import summarize

COMMON=Path(__file__).resolve().parents[1]
EXPERIMENT=COMMON.parent/'observation/shared_resources'
RESULTS=EXPERIMENT/'results/2026-10-27-saturation-gpu6-7'
CONFIGS=EXPERIMENT/'configs/saturation_gpu6_7'


def run(label,count,threads,memory=0,binary=None,traversal=0):
    values=dict(source_gpu=6,destination_gpu=7,ctas=count,threads=threads,co_ctas=memory,
                kind=3 if memory else 0,vector_reads=1,detailed=1,communication=1,
                global_stride=traversal,pacing=0,warmup=3,iterations=12)
    config=CONFIGS/f'copy_c{count}_t{threads}_m{memory}_g{traversal}.cfg'
    content=''.join(f'{key}={value}\n' for key,value in values.items())
    if config.exists() and config.read_text()!=content:
        raise RuntimeError('Configuration changed')
    config.write_text(content)
    output=RESULTS/'raw'/f'{label}.json'
    if not output.exists():
        command=[sys.executable,str(COMMON/'scripts/run_one.py'),str(config),str(output)]
        if binary:
            command+=['--binary',str(binary)]
        subprocess.run(command,check=True)
    data=json.loads(output.read_text())
    assert data['verified'] and data['source_gpu']==6 and data['destination_gpu']==7
    observations=summarize(output)
    assert len(observations)==12
    throughput=statistics.median(row['throughput_gbps'] for row in observations)
    print(label,f'{throughput:.3f} GB/s',flush=True)
    return throughput


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['screen','traversal','confirm'])
    arguments=parser.parse_args()
    for directory in (RESULTS/'raw',CONFIGS):
        directory.mkdir(parents=True,exist_ok=True)
    provenance={name:hashlib.sha256((COMMON/name).read_bytes()).hexdigest() for name in ['src/push_harness.cu','bin/push_harness']}
    path=RESULTS/'build_provenance.json'
    if path.exists() and json.loads(path.read_text())!=provenance:
        raise RuntimeError('Binary/source changed during calibration')
    path.write_text(json.dumps(provenance,indent=2)+'\n')
    if arguments.stage=='screen':
        # A single old/new reference and pressure pair checks supported behavior after the bounds change.
        run('check_old_H',32,256,binary=COMMON/'bin/push_harness_before_saturation')
        run('check_old_P',32,256,131,binary=COMMON/'bin/push_harness_before_saturation')
        run('check_new_H',32,256)
        run('check_new_P',32,256,131)
        settings=[(64,512),(128,512),(64,1024),(96,1024),(128,768),(128,1024),(132,1024)]
        for count,threads in settings:
            run(f'screen_c{count}_t{threads}',count,threads)
    elif arguments.stage=='traversal':
        settings=[(64,1024),(96,1024),(128,512),(128,768),(128,1024),(132,1024)]
        for count,threads in settings:
            run(f'traversal_c{count}_t{threads}_g1',count,threads,traversal=1)
    else:
        settings=[(128,512,0),(128,768,0),(128,896,0),(128,1024,0),(64,1024,0),(96,1024,1),(132,1024,1)]
        for repetition in range(1,6):
            rotated=settings[repetition-1:]+settings[:repetition-1]
            for count,threads,traversal in rotated:
                run(f'confirm_c{count}_t{threads}_g{traversal}_R{repetition}',count,threads,traversal=traversal)
    print(arguments.stage.upper(),'COMPLETE',flush=True)


if __name__=='__main__':
    main()
