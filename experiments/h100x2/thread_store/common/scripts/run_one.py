#!/usr/bin/env python3
"""Guard a single config invocation; retain snapshots, telemetry, logs, and raw JSON."""
import argparse
import csv
import datetime
import fcntl
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time

COMMON = Path(__file__).resolve().parents[1]

def capture(command):
    result = subprocess.run(command, text=True, capture_output=True)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

def snapshot(devices="4,5"):
    return {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu": capture(["nvidia-smi", "-i", devices, "-q", "-x"]),
        "link": capture(["nvidia-smi", "nvlink", "-i", devices, "-gt", "d"]),
    }

def ensure_idle(devices="4,5"):
    fields = "index,uuid,memory.used,utilization.gpu"
    # Driver context teardown and utilization samples can briefly outlive our prior process.
    # Wait for genuinely idle devices; persistent occupancy aborts without launching work.
    for attempt in range(20):
        result = subprocess.check_output(["nvidia-smi", "-i", devices, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True)
        occupied = [row for row in csv.reader(io.StringIO(result)) if int(row[2]) > 100 or int(row[3]) > 0]
        if not occupied:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError(f"Selected GPUs remain occupied: {occupied}")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Unset CUDA_VISIBLE_DEVICES to preserve the configured physical indices")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary", type=Path, default=COMMON / "bin/push_harness")
    # Replaces the config's devices, so one config can run on any allocated pair.
    parser.add_argument("--gpus", help="SOURCE,DESTINATION physical GPU indices")
    # store_congest only: the source GPU's SM-to-GPC map (plan 07).
    parser.add_argument("--gpc-map", type=Path)
    parser.add_argument("--destination-gpc-map", type=Path)
    arguments = parser.parse_args()
    config = {}
    for line in arguments.config.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            config[key.strip()] = int(value)
    source_gpu = config.get("source_gpu", 4)
    destination_gpu = config.get("destination_gpu", 5)
    if arguments.gpus:
        source_gpu, destination_gpu = (int(device) for device in arguments.gpus.split(","))
    # User allocation: GPUs 4-7 until 2026-10-27; GPUs 0, 1, 6, 7 from 2026-10-28; GPUs 4, 5 for plan 07 from 2026-10-29; GPUs 0, 1 for plan 07 from 2026-10-30.
    if source_gpu == destination_gpu or any(device not in (0, 1, 4, 5, 6, 7) for device in (source_gpu, destination_gpu)):
        raise ValueError("Expected two distinct physical GPU indices from the allocations 0, 1, 6, 7 or 4, 5")
    devices = f"{source_gpu},{destination_gpu}"
    # Serialize our guarded invocations before checking occupancy: idle checks alone race.
    with ExitStack() as locks:
        for device in sorted((source_gpu, destination_gpu)):
            handle = locks.enter_context(open(f"/tmp/nccl_thread_store_gpu_{device}.lock", "a"))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another guarded experiment holds GPU {device}") from error
        ensure_idle(devices)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        if arguments.output.exists():
            raise RuntimeError(f"Refusing to overwrite {arguments.output}")
        arguments.output.with_suffix(".before.json").write_text(json.dumps(snapshot(devices)))
        telemetry = []
        stop_event = threading.Event()
        def sample_telemetry():
            while not stop_event.is_set():
                started = time.time_ns()
                result = capture(["nvidia-smi", "-i", devices, "--query-gpu=index,timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,utilization.memory,clocks_event_reasons.active", "--format=csv,noheader,nounits"])
                telemetry.append({"host_start_ns": started, "host_end_ns": time.time_ns(), **result})
                stop_event.wait(0.1)
        sampler = threading.Thread(target=sample_telemetry)
        sampler.start()
        started = time.time_ns()
        try:
            with arguments.output.with_suffix(".log").open("w") as log:
                command = [str(arguments.binary.resolve()), str(arguments.config.resolve()), str(arguments.output.resolve())]
                if arguments.gpus:
                    command += ["--gpus", f"{source_gpu},{destination_gpu}"]
                if arguments.gpc_map:
                    command += ["--gpc-map", str(arguments.gpc_map.resolve())]
                if arguments.destination_gpc_map:
                    command += ["--destination-gpc-map", str(arguments.destination_gpc_map.resolve())]
                result = subprocess.run(command, stdout=log, stderr=log, timeout=45, env={**os.environ, "NCCL_DEBUG": "WARN"})
            if result.returncode:
                raise RuntimeError(f"Harness exited {result.returncode}; see {arguments.output.with_suffix('.log')}")
        finally:
            stop_event.set()
            sampler.join()
            arguments.output.with_suffix(".telemetry.json").write_text(json.dumps({"host_start_ns": started, "host_end_ns": time.time_ns(), "samples": telemetry}))
            arguments.output.with_suffix(".after.json").write_text(json.dumps(snapshot(devices)))
        print(arguments.output.name, flush=True)

if __name__ == "__main__":
    main()
