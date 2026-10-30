#!/usr/bin/env python3
"""E7.0: record cluster placement on each GPU of the pair, then build the SM-to-GPC maps.

Usage: run_gpc_map.py --gpus 4,5 [--repeats 20]
Raw probe records go to results/<date>/gpc_map/raw/gpu<N>.json (with device snapshots);
build_gpc_map.py turns them into configs/gpc_map/gpu<N>.json, which store_congest reads.
"""
import argparse
import fcntl
import json
import subprocess
from contextlib import ExitStack
from pathlib import Path

from run_one import ensure_idle, snapshot

COMMON = Path(__file__).resolve().parents[1]
EXPERIMENT = COMMON.parent / 'observation/data_store_congest'
RESULTS = EXPERIMENT / 'results/2026-10-29'


def probe(gpu, repeats):
    output = RESULTS / 'gpc_map/raw' / f'gpu{gpu}.json'
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    # Same per-device lock as run_one.py, held while checking occupancy and running.
    with ExitStack() as locks:
        handle = locks.enter_context(open(f'/tmp/nccl_thread_store_gpu_{gpu}.lock', 'a'))
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_idle(str(gpu))
        output.with_suffix('.before.json').write_text(json.dumps(snapshot(str(gpu))))
        with output.with_suffix('.log').open('w') as log:
            subprocess.run([str(COMMON / 'bin/gpc_probe'), str(gpu), str(repeats), str(output)], stdout=log, stderr=log, check=True, timeout=120)
        output.with_suffix('.after.json').write_text(json.dumps(snapshot(str(gpu))))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--repeats', type=int, default=20)
    arguments = parser.parse_args()
    for gpu in (int(device) for device in arguments.gpus.split(',')):
        print(probe(gpu, arguments.repeats), flush=True)


if __name__ == '__main__':
    main()
