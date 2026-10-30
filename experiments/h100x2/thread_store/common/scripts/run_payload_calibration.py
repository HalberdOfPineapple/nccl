#!/usr/bin/env python3
"""Check transfer-size overhead with fixed communication launch settings."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from collect import summarize
from run_saturation_calibration import COMMON, EXPERIMENT

ROOT=EXPERIMENT/'results/2026-10-27-saturation-gpu6-7/payload'
CONFIGS=EXPERIMENT/'configs/saturation_gpu6_7'


def run(label,count,threads,payload,traversal=0):
    values=dict(source_gpu=6,destination_gpu=7,ctas=count,threads=threads,co_ctas=0,kind=0,
                vector_reads=1,detailed=1,communication=1,global_stride=traversal,pacing=0,
                warmup=3,iterations=12,payload_mib=payload)
    config=CONFIGS/f'payload_c{count}_t{threads}_p{payload}_g{traversal}.cfg'
    content=''.join(f'{key}={value}\n' for key,value in values.items())
    if config.exists() and config.read_text()!=content:
        raise RuntimeError('Configuration changed')
    config.write_text(content)
    output=ROOT/'raw'/f'{label}.json'
    if not output.exists():
        subprocess.run([sys.executable,str(COMMON/'scripts/run_one.py'),str(config),str(output)],check=True)
    data=json.loads(output.read_text())
    assert data['verified'] and data['bytes']==payload*1024*1024
    rows=summarize(output)
    assert len(rows)==12
    throughput=statistics.median(row['throughput_gbps'] for row in rows)
    print(label,f'{throughput:.3f} GB/s',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['screen','confirm'])
    arguments=parser.parse_args()
    (ROOT/'raw').mkdir(parents=True,exist_ok=True)
    CONFIGS.mkdir(parents=True,exist_ok=True)
    provenance={name:hashlib.sha256((COMMON/name).read_bytes()).hexdigest() for name in ['src/push_harness.cu','bin/push_harness']}
    path=ROOT/'build_provenance.json'
    if path.exists() and json.loads(path.read_text())!=provenance:
        raise RuntimeError('Program changed')
    path.write_text(json.dumps(provenance,indent=2)+'\n')
    if arguments.stage=='screen':
        settings=[(128,1024,payload,0) for payload in (256,512,1024,2048)]
        settings += [(128,512,2048,0),(128,768,2048,0),(64,1024,2048,0),(132,1024,2048,1)]
        for count,threads,payload,traversal in settings:
            run(f'screen_c{count}_t{threads}_p{payload}_g{traversal}',count,threads,payload,traversal)
    else:
        settings=[(128,512,2048,0),(128,768,2048,0),(128,1024,2048,0),(128,1024,1024,0),(128,1024,4096,0),(132,1024,2048,1)]
        for repetition in range(1,6):
            rotated=settings[repetition-1:]+settings[:repetition-1]
            for count,threads,payload,traversal in rotated:
                run(f'confirm_c{count}_t{threads}_p{payload}_g{traversal}_R{repetition}',count,threads,payload,traversal)
    print(arguments.stage.upper(),'COMPLETE',flush=True)


if __name__=='__main__':
    main()
