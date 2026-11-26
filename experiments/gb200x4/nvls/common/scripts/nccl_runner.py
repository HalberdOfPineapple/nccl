#!/usr/bin/env python3
"""Run a plan of nccl-tests invocations on one GB200 node, one run at a time, without intervention.

Runs INSIDE the container (nvcr.io/nvidia/pytorch:25.06-py3). Launch it through
run_in_container.sh, which checks the container, cleans PATH and builds NCCL / nccl-tests.

What one run is: one nccl-tests process on all 4 GPUs (-t 1 -g 4) for one arm, one CTA setting,
one payload size. Around it the runner saves:
  - the NCCL log (NCCL_DEBUG=INFO, subsystems INIT,ENV,TUNING) in its own file,
  - nvidia-smi NVLink data and raw counters of every GPU before and after (metric B3, R8-R10),
  - GPU clocks, temperature, power and throttle reasons before and after (metric G1),
  - the parsed nccl-tests table and whether the log names the intended kernel (C8).

Plans are JSON files (see ../plans/). A plan lists blocks; each block expands into runs over
arms x CTA settings x sizes x rounds. Runs are ordered round by round; inside a round the order is
shuffled with a fixed seed so that arms and settings rotate. A counter pair (long then short run of
the same configuration) always stays back to back.

Resuming: every finished run writes result.json. Rerunning the same plan into the same session
directory skips runs whose result.json exists, so a requeued batch job continues where it stopped.

Output keys in result.json and runs.jsonl are read by summarize_nccl.py: never rename them.
"""

import argparse
import datetime
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
NCCL_LIBRARY = REPOSITORY / 'build-gb200' / 'lib'
NCCL_TESTS = Path('/mnt/home/wli10/nccl-tests/build-gb200')
GPU_COUNT = 4
RUN_TIMEOUT_SECONDS = 900

# Arm name -> how to select it. 'knob' says how a CTA count is applied:
#   symmetric: NCCL_SYM_CTAS (symmetric kernels, needs -R 2 registration)
#   channels:  NCCL_MIN_NCHANNELS = NCCL_MAX_NCHANNELS (Ring)
#   nvls:      NCCL_NVLS_NCHANNELS (non-symmetric NVLS)
# 'expect' is the text the NCCL log must contain for the run to count (C8).
ARMS = {
    # Above NCCL_SYM_CE_THRESHOLD (8 MiB default) a registered AllGather runs on the copy engine,
    # and NCCL_SYM_KERNEL does not override that (found 2026-11-26). 1 TiB keeps the kernel path.
    'AG-MC':   {'program': 'all_gather_perf', 'knob': 'symmetric', 'collective': 'AllGather',
                'environment': {'NCCL_SYM_KERNEL': 'allgather:AllGather_STMC',
                                'NCCL_SYM_CE_THRESHOLD': str(1 << 40)}, 'register': True,
                'expect': r'AllGather \[Symmetric\].*-> Kernel AllGather_STMC'},
    # Copy-engine AllGather (cudaMemcpy to every peer, NVLS synchronization): NCCL's own choice for
    # registered AllGather above 8 MiB. No CTAs. Unicast copies to the 3 peers: 0.75 data bytes per
    # payload byte on each direction (not a multicast arm).
    'AG-CE':   {'program': 'all_gather_perf', 'knob': None, 'collective': 'AllGather',
                'environment': {}, 'register': True,
                'expect': r'AllGather \[Copy Engine\].*-> cudaMemcpy'},
    'RS-MC':   {'program': 'reduce_scatter_perf', 'knob': 'symmetric', 'collective': 'ReduceScatter',
                'environment': {'NCCL_SYM_KERNEL': 'reducescatter:ReduceScatter_LDMC'}, 'register': True,
                'expect': r'ReduceScatter \[Symmetric\].*-> Kernel ReduceScatter_LDMC'},
    'AR-MC':   {'program': 'all_reduce_perf', 'knob': 'symmetric', 'collective': 'AllReduce',
                'environment': {'NCCL_SYM_KERNEL': 'allreduce:AllReduce_RSxLDMC_AGxSTMC'}, 'register': True,
                'expect': r'AllReduce \[Symmetric\].*-> Kernel AllReduce_RSxLDMC_AGxSTMC'},
    'AR-TMA':  {'program': 'all_reduce_perf', 'knob': 'symmetric', 'collective': 'AllReduce',
                'environment': {'NCCL_SYM_KERNEL': 'allreduce:AllReduce_RSxTmaLD_AGxTmaST'}, 'register': True,
                'expect': r'AllReduce \[Symmetric\].*-> Kernel AllReduce_RSxTmaLD_AGxTmaST'},
    'AR-LDST': {'program': 'all_reduce_perf', 'knob': 'symmetric', 'collective': 'AllReduce',
                'environment': {'NCCL_SYM_KERNEL': 'allreduce:AllReduce_RSxLD_AGxST'}, 'register': True,
                'expect': r'AllReduce \[Symmetric\].*-> Kernel AllReduce_RSxLD_AGxST'},
    'AR-NVLS': {'program': 'all_reduce_perf', 'knob': 'nvls', 'collective': 'AllReduce',
                'environment': {'NCCL_ALGO': 'allreduce:nvls'}, 'register': False,
                'expect': r'AllReduce: .*-> Algo NVLS proto SIMPLE'},
    'AR-Ring': {'program': 'all_reduce_perf', 'knob': 'channels', 'collective': 'AllReduce',
                'environment': {'NCCL_ALGO': 'allreduce:ring', 'NCCL_PROTO': 'Simple'}, 'register': False,
                'expect': r'AllReduce: .*-> Algo RING proto SIMPLE'},
    'AG-Ring': {'program': 'all_gather_perf', 'knob': 'channels', 'collective': 'AllGather',
                'environment': {'NCCL_ALGO': 'allgather:ring', 'NCCL_PROTO': 'Simple'}, 'register': False,
                'expect': r'AllGather: .*-> Algo RING proto SIMPLE'},
    'RS-Ring': {'program': 'reduce_scatter_perf', 'knob': 'channels', 'collective': 'ReduceScatter',
                'environment': {'NCCL_ALGO': 'reducescatter:ring', 'NCCL_PROTO': 'Simple'}, 'register': False,
                'expect': r'ReduceScatter: .*-> Algo RING proto SIMPLE'},
    # NCCL's own choice, nothing forced: records what the tuner picks (plan 01 E1.3 step 4).
    'AR-Tuner': {'program': 'all_reduce_perf', 'knob': None, 'collective': 'AllReduce',
                 'environment': {}, 'register': False, 'expect': r'AllReduce'},
}

SIZE_UNITS = {'K': 1 << 10, 'M': 1 << 20, 'G': 1 << 30}


# ---------------------------------------------------------------- plan expansion

def sizeInBytes(sizeText):
    """'2G' -> 2147483648. nccl-tests takes the same text."""
    return int(sizeText[:-1]) * SIZE_UNITS[sizeText[-1]]


def loadCalibration(path):
    """Calibrated CTA count per arm, written by hand after the calibration analysis (plans say where)."""
    with open(path) as handle:
        return json.load(handle)['ctas']


def expandBlock(block, blockIndex, calibration):
    """One block of a plan -> list of run groups per round. A group is one run or a long/short pair."""
    roundsOfGroups = []
    for roundIndex in range(block['rounds']):
        groups = []
        for arm in block['arms']:
            ctaSettings = block.get('ctas', {}).get(arm, block.get('ctas_default', [None]))
            if ctaSettings == 'calibrated':
                ctaSettings = [calibration[arm]]
            for ctas in ctaSettings:
                for size in block['sizes']:
                    base = {
                        'block': block['name'], 'block_index': blockIndex, 'round': roundIndex,
                        'arm': arm, 'ctas': ctas, 'size': size, 'check': block.get('check', 0),
                        'extra_environment': block.get('environment', {}),
                    }
                    if block.get('counter_pair'):
                        long, short = block['counter_pair']['long'], block['counter_pair']['short']
                        groups.append([
                            {**base, 'length': 'long', 'iterations': long['n'], 'warmup': long['w']},
                            {**base, 'length': 'short', 'iterations': short['n'], 'warmup': short['w']},
                        ])
                    else:
                        groups.append([{**base, 'length': 'single',
                                        'iterations': block['iterations'], 'warmup': block['warmup']}])
        roundsOfGroups.append(groups)
    return roundsOfGroups


def expandPlan(plan, calibration, seed):
    """Whole plan -> ordered list of runs with stable ids. Blocks run in plan order."""
    generator = random.Random(seed)
    runs = []
    for blockIndex, block in enumerate(plan['blocks']):
        for groups in expandBlock(block, blockIndex, calibration):
            generator.shuffle(groups)
            for group in groups:
                runs.extend(group)
    for runIndex, run in enumerate(runs):
        ctasText = 'default' if run['ctas'] is None else f"c{run['ctas']}"
        run['run_id'] = (f"{runIndex:04d}_{run['block']}_{run['arm']}_{ctasText}_{run['size']}"
                         f"_r{run['round']}_{run['length']}")
    return runs


# ---------------------------------------------------------------- snapshots

def runCommand(command, timeoutSeconds=60):
    """Run a read-only tool and return its stdout (empty string on failure, error kept by caller)."""
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeoutSeconds).stdout
    except (subprocess.SubprocessError, OSError) as error:
        return f'ERROR {error}'


LINK_LINE = re.compile(r'Link (\d+): (Data|Raw) (Tx|Rx): (\d+) KiB')


def readLinkCounters():
    """NVLink byte counters of every GPU: {'data'|'raw': {gpu: {'tx': [KiB per link], 'rx': [...]}}}."""
    counters = {}
    for counterType, flag in (('data', 'd'), ('raw', 'r')):
        text = runCommand(['nvidia-smi', 'nvlink', '-gt', flag])
        perGpu = {}
        gpu = None
        for line in text.splitlines():
            header = re.match(r'GPU (\d+):', line)
            if header:
                gpu = header.group(1)
                perGpu[gpu] = {'tx': {}, 'rx': {}}
                continue
            match = LINK_LINE.search(line)
            if match and gpu is not None:
                perGpu[gpu][match.group(3).lower()][int(match.group(1))] = int(match.group(4))
        counters[counterType] = {
            gpu: {direction: [links[index] for index in sorted(links)] for direction, links in directions.items()}
            for gpu, directions in perGpu.items()
        }
    return counters


GPU_QUERY = ('index,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,'
             'clocks_event_reasons.active,utilization.gpu,memory.used')


def readGpuState():
    """Clocks, temperature, power, throttle reasons per GPU (G1)."""
    text = runCommand(['nvidia-smi', f'--query-gpu={GPU_QUERY}', '--format=csv,noheader,nounits'])
    keys = GPU_QUERY.split(',')
    return [dict(zip(keys, [field.strip() for field in line.split(',')])) for line in text.strip().splitlines()]


def readComputeProcesses():
    """Processes on the GPUs; must be empty before a run (no other tenant, no leftover run)."""
    text = runCommand(['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid', '--format=csv,noheader'])
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


def gitState(path):
    """Commit and whether the work tree differs, for the session record."""
    commit = runCommand(['git', '-C', str(path), 'rev-parse', 'HEAD']).strip()
    status = runCommand(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no']).strip()
    return {'commit': commit, 'modified_files': status.splitlines()}


def writeSessionContext(sessionDirectory):
    """Session context (G1): node, rack, GPUs, fabric, driver, software versions."""
    hostname = socket.gethostname()
    rackMatch = re.match(r'b(\d+)-', hostname)
    context = {
        'started_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'hostname': hostname,
        'rack': int(rackMatch.group(1)) if rackMatch else None,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'slurm_partition': os.environ.get('SLURM_JOB_PARTITION'),
        'gpus': readGpuState(),
        'compute_processes': readComputeProcesses(),
        'nvidia_smi_q': runCommand(['nvidia-smi', '-q']),
        'nvlink_status': runCommand(['nvidia-smi', 'nvlink', '-s']),
        'topology': runCommand(['nvidia-smi', 'topo', '-m']),
        'nvcc': runCommand(['nvcc', '--version']),
        'nccl_repository': gitState(REPOSITORY),
        'nccl_tests_repository': gitState(NCCL_TESTS.parent),
        'nccl_library': str(NCCL_LIBRARY),
    }
    (sessionDirectory / 'raw').mkdir(parents=True, exist_ok=True)
    path = sessionDirectory / 'raw' / f"session_context_{context['started_utc'].replace(':', '')}.json"
    path.write_text(json.dumps(context, indent=1))
    return context


# ---------------------------------------------------------------- one run

TABLE_ROW = re.compile(r'^\s+(\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(-?\d+)\s+'
                       r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s*$')
SYMMETRIC_LAUNCH = re.compile(r'-> Kernel (\w+) nchannels (\d+) nthreads (\d+)')
CHANNEL_LAUNCH = re.compile(r'-> Algo (\w+) proto (\w+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}')
COPY_ENGINE_LAUNCH = re.compile(r'\[Copy Engine\]: \d+ Bytes -> cudaMemcpy')


def buildEnvironment(run, logPattern):
    """Environment of one nccl-tests process: library path, logging, arm selection, CTA knob."""
    arm = ARMS[run['arm']]
    environment = dict(os.environ)
    environment.update({
        'LD_LIBRARY_PATH': f"{NCCL_LIBRARY}:{environment.get('LD_LIBRARY_PATH', '')}",
        'NCCL_DEBUG': 'INFO',
        'NCCL_DEBUG_SUBSYS': 'INIT,ENV,TUNING',
        'NCCL_DEBUG_FILE': logPattern,
    })
    environment.update(arm['environment'])
    if run['ctas'] is not None:
        if arm['knob'] == 'symmetric':
            environment['NCCL_SYM_CTAS'] = str(run['ctas'])
        elif arm['knob'] == 'channels':
            environment['NCCL_MIN_NCHANNELS'] = str(run['ctas'])
            environment['NCCL_MAX_NCHANNELS'] = str(run['ctas'])
        elif arm['knob'] == 'nvls':
            environment['NCCL_NVLS_NCHANNELS'] = str(run['ctas'])
    environment.update({key: str(value) for key, value in run['extra_environment'].items()})
    return environment


def buildCommand(run):
    """nccl-tests command line. One size per process, so sizes rotate between processes."""
    arm = ARMS[run['arm']]
    command = [str(NCCL_TESTS / arm['program']), '-t', '1', '-g', str(GPU_COUNT),
               '-b', run['size'], '-e', run['size'], '-f', '2',
               '-n', str(run['iterations']), '-w', str(run['warmup']),
               '-c', str(run['check']), '-d', 'float']
    if arm['collective'] != 'AllGather':
        command += ['-o', 'sum']
    if arm['register']:
        command += ['-R', '2']
    return command


def parseTable(stdout):
    """nccl-tests result rows (out-of-place and in-place time, algBW, busBW, wrong count)."""
    rows = []
    for line in stdout.splitlines():
        match = TABLE_ROW.match(line)
        if match:
            fields = match.groups()
            rows.append({
                'size_bytes': int(fields[0]), 'count': int(fields[1]), 'type': fields[2], 'redop': fields[3],
                'out_of_place_time_us': float(fields[5]), 'out_of_place_algbw_gbps': float(fields[6]),
                'out_of_place_busbw_nccltests_gbps': float(fields[7]), 'out_of_place_wrong': fields[8],
                'in_place_time_us': float(fields[9]), 'in_place_algbw_gbps': float(fields[10]),
                'in_place_busbw_nccltests_gbps': float(fields[11]), 'in_place_wrong': fields[12],
            })
    return rows


def inspectLogs(runDirectory, run):
    """Which kernel / algorithm and how many CTAs NCCL actually launched, from all rank logs (C8)."""
    arm = ARMS[run['arm']]
    expected = re.compile(arm['expect'])
    launches = {}
    expectedFound = False
    for logPath in sorted(runDirectory.glob('nccl.*.log')):
        for line in logPath.open(errors='replace'):
            if '->' not in line:
                continue
            if expected.search(line):
                expectedFound = True
            symmetric = SYMMETRIC_LAUNCH.search(line)
            channel = CHANNEL_LAUNCH.search(line)
            if COPY_ENGINE_LAUNCH.search(line):
                key = 'copy_engine'
            elif symmetric:
                key = f'{symmetric.group(1)} ctas={symmetric.group(2)} threads={symmetric.group(3)}'
            elif channel:
                ctaCount = int(channel.group(4)) - int(channel.group(3)) + 1
                key = f'{channel.group(1)}/{channel.group(2)} ctas={ctaCount}'
            else:
                continue
            launches[key] = launches.get(key, 0) + 1
    return expectedFound, launches


def executeRun(run, sessionDirectory):
    """One nccl-tests process with snapshots around it; writes result.json last (marks completion)."""
    runDirectory = sessionDirectory / 'raw' / run['run_id']
    if runDirectory.exists():
        shutil.rmtree(runDirectory)  # an interrupted earlier attempt; result.json was never written
    runDirectory.mkdir(parents=True)
    environment = buildEnvironment(run, str(runDirectory / 'nccl.%p.log'))
    command = buildCommand(run)

    record = {'run': run, 'command': command,
              'environment_overrides': {key: environment[key] for key in sorted(environment)
                                        if key.startswith('NCCL_')},
              'compute_processes_before': readComputeProcesses(),
              'gpu_state_before': readGpuState(), 'links_before': readLinkCounters()}
    startedSeconds = time.time()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, env=environment,
                                   timeout=RUN_TIMEOUT_SECONDS)
        stdout, stderr, exitCode = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or '')
        stderr, exitCode = 'TIMEOUT', None
    record['wall_seconds'] = time.time() - startedSeconds
    record['links_after'] = readLinkCounters()
    record['gpu_state_after'] = readGpuState()
    (runDirectory / 'stdout.txt').write_text(stdout)
    (runDirectory / 'stderr.txt').write_text(stderr)

    record['exit_code'] = exitCode
    record['rows'] = parseTable(stdout)
    record['expected_kernel_found'], record['launches'] = inspectLogs(runDirectory, run)
    if exitCode != 0 or not record['rows']:
        record['status'] = 'failed'
    elif not record['expected_kernel_found']:
        record['status'] = 'kernel_mismatch'
    elif any(row[key] not in ('0', 'N/A') for row in record['rows'] for key in ('out_of_place_wrong', 'in_place_wrong')):
        # nccl-tests prints the count of wrong values with -c 1 and 'N/A' with -c 0 (A11).
        record['status'] = 'wrong_results'
    elif record['compute_processes_before']:
        record['status'] = 'foreign_process'
    else:
        record['status'] = 'ok'
    (runDirectory / 'result.json').write_text(json.dumps(record, indent=1))
    return record


# ---------------------------------------------------------------- session

def parseArguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('plan', type=Path, help='plan JSON file')
    parser.add_argument('--session', type=Path, required=True, help='session directory (created; reused to resume)')
    parser.add_argument('--calibration', type=Path, help='calibrated CTA counts (needed by plans using "calibrated")')
    parser.add_argument('--stop-after-minutes', type=float, default=None,
                        help='start no new run after this many minutes (end cleanly before the job limit)')
    parser.add_argument('--dry-run', action='store_true', help='print the expanded runs only')
    return parser.parse_args()


def main():
    arguments = parseArguments()
    plan = json.loads(arguments.plan.read_text())
    calibration = loadCalibration(arguments.calibration) if arguments.calibration else {}
    runs = expandPlan(plan, calibration, seed=plan.get('seed', 1))
    if arguments.dry_run:
        for run in runs:
            print(run['run_id'], ' '.join(buildCommand(run)))
        print(f'{len(runs)} runs')
        return

    sessionDirectory = arguments.session
    sessionDirectory.mkdir(parents=True, exist_ok=True)
    shutil.copy(arguments.plan, sessionDirectory / f'plan_{plan["name"]}.json')
    writeSessionContext(sessionDirectory)
    deadline = (time.time() + 60 * arguments.stop_after_minutes) if arguments.stop_after_minutes else None
    index = sessionDirectory / 'runs.jsonl'

    done = 0
    for position, run in enumerate(runs):
        resultPath = sessionDirectory / 'raw' / run['run_id'] / 'result.json'
        if resultPath.exists():
            continue
        # A long/short pair must not be split by the deadline: check before the long run for both.
        if deadline and time.time() > deadline and run['length'] != 'short':
            print(f'deadline reached; stopped before {run["run_id"]}', flush=True)
            break
        record = executeRun(run, sessionDirectory)
        done += 1
        summary = {'run_id': run['run_id'], 'status': record['status'], 'wall_seconds': round(record['wall_seconds'], 1),
                   'out_of_place_algbw_gbps': record['rows'][0]['out_of_place_algbw_gbps'] if record['rows'] else None,
                   'launches': record['launches']}
        with index.open('a') as handle:
            handle.write(json.dumps(summary) + '\n')
        print(f"[{position + 1}/{len(runs)}] {run['run_id']} {record['status']} "
              f"algbw={summary['out_of_place_algbw_gbps']} {round(record['wall_seconds'], 1)}s "
              f"{list(record['launches'])[:2]}", flush=True)
    print(f'finished: {done} runs this invocation', flush=True)


if __name__ == '__main__':
    main()
