#!/usr/bin/env python3
"""Run one config on one GPU under a guard; keep snapshots, telemetry, the log, and raw JSON.

Same protections as the thread-store wrapper: a per-device lock taken before the idle check,
refusal to overwrite a result, and before/after device snapshots.
"""
import argparse
import csv
import datetime
import fcntl
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


def snapshot(device):
    return {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "gpu": capture(["nvidia-smi", "-i", device, "-q", "-x"])}


def ensure_idle(device):
    # Context teardown from our previous run can briefly look like occupancy; wait, then abort.
    fields = "index,uuid,memory.used,utilization.gpu"
    for attempt in range(20):
        result = subprocess.check_output(["nvidia-smi", "-i", device, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True)
        occupied = [row for row in csv.reader(io.StringIO(result)) if int(row[2]) > 100 or int(row[3]) > 0]
        if not occupied:
            return
        time.sleep(0.5)
    raise RuntimeError(f"Selected GPU remains occupied: {occupied}")


def sample_telemetry(device, samples, stop_event):
    while not stop_event.is_set():
        started = time.time_ns()
        result = capture(["nvidia-smi", "-i", device, "--query-gpu=index,timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,utilization.memory,clocks_event_reasons.active", "--format=csv,noheader,nounits"])
        samples.append({"host_start_ns": started, "host_end_ns": time.time_ns(), **result})
        stop_event.wait(0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary", type=Path, default=COMMON / "bin/hbm_read_calibration")
    arguments = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Unset CUDA_VISIBLE_DEVICES; configs use physical GPU indices")
    config = dict(line.split("=", 1) for line in arguments.config.read_text().splitlines() if line.strip() and not line.startswith("#"))
    gpu = int(config.get("gpu", 6))
    if gpu not in range(4, 8):
        raise ValueError("Expected a physical GPU index in 4-7")
    device = str(gpu)
    with open(f"/tmp/nccl_thread_store_gpu_{gpu}.lock", "a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another guarded experiment holds GPU {gpu}") from error
        ensure_idle(device)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        if arguments.output.exists():
            raise RuntimeError(f"Refusing to overwrite {arguments.output}")
        arguments.output.with_suffix(".before.json").write_text(json.dumps(snapshot(device)))
        samples = []
        stop_event = threading.Event()
        sampler = threading.Thread(target=sample_telemetry, args=(device, samples, stop_event))
        sampler.start()
        started = time.time_ns()
        try:
            with arguments.output.with_suffix(".log").open("w") as log:
                command = [str(arguments.binary.resolve()), str(arguments.config.resolve()), str(arguments.output.resolve())]
                result = subprocess.run(command, stdout=log, stderr=log, timeout=60)
            if result.returncode:
                raise RuntimeError(f"Harness exited {result.returncode}; see {arguments.output.with_suffix('.log')}")
        finally:
            stop_event.set()
            sampler.join()
            arguments.output.with_suffix(".telemetry.json").write_text(json.dumps({"host_start_ns": started, "host_end_ns": time.time_ns(), "samples": samples}))
            arguments.output.with_suffix(".after.json").write_text(json.dumps(snapshot(device)))
        print(arguments.output.name, flush=True)


if __name__ == "__main__":
    main()
