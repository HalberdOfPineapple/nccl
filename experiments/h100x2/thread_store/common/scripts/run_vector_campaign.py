#!/usr/bin/env python3
"""Fresh matched comparisons after calibrating a read co-runner with higher request concurrency."""
from pathlib import Path
import hashlib
import json
import subprocess
from run_campaign import COMMON,CONFIGS,RESULTS,run_one


def main():
    provenance={}
    for relative in ['src/push_harness.cu','bin/push_harness']:
        path=COMMON/relative
        provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (RESULTS/'vector_provenance.json').write_text(json.dumps(provenance,indent=2))
    run_one('healthy','vector_revision_healthy_check')
    for name in ['source_memory','destination_memory']:
        levels=[8,32,100] if name=='source_memory' else [8,32,100,131]
        # Calibrate the maximum before interpreting M as achieved pressure.
        run_one(f'vector_{name}_{levels[-1]}',f'calibration_vector_{name}_{levels[-1]}')
        for count in levels:
            subprocess.run(['bash',str(COMMON/'scripts/run_pair.sh'),str(RESULTS),'5',str(CONFIGS/'healthy.cfg'),str(CONFIGS/f'vector_{name}_{count}.cfg'),f'vector_{name}_{count}'],check=True)
            run_one(f'vector_{name}_{count}_solo',f'vector_{name}_{count}_solo')
        for pacing in [64,256,1024]:
            run_one(f'vector_{name}_100_pacing_{pacing}',f'vector_pacing_{name}_{pacing}')
    subprocess.run(['bash',str(COMMON/'scripts/run_pair.sh'),str(RESULTS),'5',str(CONFIGS/'healthy_minimal.cfg'),str(CONFIGS/'healthy.cfg'),'vector_instrumentation_overhead'],check=True)
    run_one('healthy','vector_return_to_baseline')
    subprocess.run(['python3',str(COMMON/'scripts/collect.py'),str(RESULTS)],check=True)
    subprocess.run(['python3',str(COMMON/'scripts/summarize.py'),str(RESULTS)],check=True)

if __name__=='__main__':
    main()
