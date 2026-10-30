#!/usr/bin/env python3
"""Increase independent scalar requests per thread before accepting a throughput ceiling."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from collect import summarize
from run_saturation_calibration import COMMON, EXPERIMENT

ROOT=EXPERIMENT/'results/2026-10-27-saturation-gpu6-7/batch'
CONFIGS=EXPERIMENT/'configs/saturation_gpu6_7'


def run(label,count,threads,payload,batch,memory=0,detailed=1):
    values=dict(source_gpu=6,destination_gpu=7,ctas=count,threads=threads,co_ctas=memory,kind=3 if memory else 0,
                vector_reads=1,detailed=detailed,communication=1,global_stride=0,pacing=0,
                warmup=3,iterations=12,payload_mib=payload,copy_batch=batch)
    config=CONFIGS/f'batch_c{count}_t{threads}_p{payload}_b{batch}_m{memory}_d{detailed}.cfg'
    content=''.join(f'{key}={value}\n' for key,value in values.items())
    if config.exists() and config.read_text()!=content:
        raise RuntimeError('Configuration changed')
    config.write_text(content)
    output=ROOT/'raw'/f'{label}.json'
    if not output.exists():
        subprocess.run([sys.executable,str(COMMON/'scripts/run_one.py'),str(config),str(output)],check=True)
    data=json.loads(output.read_text())
    assert data['verified'] and data['bytes']==payload*1024*1024 and data['copy_batch']==batch
    rows=summarize(output)
    assert len(rows)==12
    print(label,f"{statistics.median(row['throughput_gbps'] for row in rows):.3f} GB/s",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['screen','confirm','probes'])
    arguments=parser.parse_args()
    (ROOT/'raw').mkdir(parents=True,exist_ok=True)
    CONFIGS.mkdir(parents=True,exist_ok=True)
    provenance={name:hashlib.sha256((COMMON/name).read_bytes()).hexdigest() for name in ['src/push_harness.cu','bin/push_harness']}
    path=ROOT/'build_provenance.json'
    if path.exists() and json.loads(path.read_text())!=provenance:
        raise RuntimeError('Program changed')
    path.write_text(json.dumps(provenance,indent=2)+'\n')
    if arguments.stage=='screen':
        run('check_new_H',32,256,256,1)
        run('check_new_P',32,256,256,1,131)
        settings=[(128,1024,2048,batch) for batch in (1,2,4,8)]
        settings += [(128,512,2048,4),(128,512,2048,8),(64,512,2048,8),(64,1024,2048,8),(128,1024,4096,8)]
        for count,threads,payload,batch in settings:
            run(f'screen_c{count}_t{threads}_p{payload}_b{batch}',count,threads,payload,batch)
    elif arguments.stage=='probes':
        for repetition in range(1,6):
            for detailed in (0,1):
                run(f'probe_c128_t512_p2048_b8_d{detailed}_R{repetition}',128,512,2048,8,detailed=detailed)
    else:
        settings=[(128,512,2048,8),(128,768,2048,8),(128,1024,2048,8),(128,1024,2048,1),(128,1024,2048,2),(128,1024,2048,4),(64,1024,2048,8),(128,1024,4096,8)]
        for repetition in range(1,6):
            rotated=settings[repetition-1:]+settings[:repetition-1]
            for count,threads,payload,batch in rotated:
                run(f'confirm_c{count}_t{threads}_p{payload}_b{batch}_R{repetition}',count,threads,payload,batch)
    print(arguments.stage.upper(),'COMPLETE',flush=True)


if __name__=='__main__':
    main()
