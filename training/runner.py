#!/usr/bin/env python3
"""
GPU job scheduler for the ISLES'26 experiment sweep.

Runs nnU-Net trainings across the free capacity of the 8 shared GPUs. One
worker thread per GPU slot pulls the next pending job from a queue directory,
so new experiments can be dropped in while the sweep is already running
(write a .json into $ISLES_SWEEP_DIR/queue/ and a free worker picks it up).

Job file format ($ISLES_SWEEP_DIR/queue/<id>.json):
    {
      "id": "w1_tversky",
      "trainer": "nnUNetTrainerTverskyCE_250epochs",
      "plans": "nnUNetResEncUNetMPlans",
      "config": "3d_fullres",
      "dataset": 1,
      "fold": 0,
      "extra_args": ["--npz"],
      "min_free_gb": 25,       # don't start unless the GPU has this much free
      "priority": 1            # lower runs first
    }

Usage:
    source training/env.sh
    python3 training/runner.py --gpus 0,1,2,3,4,5,6,7 --slots-per-gpu 1

Everything it needs about the filesystem comes from the environment (see
training/env.sh): ISLES_ROOT, ISLES_SWEEP_DIR, nnUNet_raw/_preprocessed/
_results, nnUNet_extTrainer. Values already exported are left alone; the
defaults below only fill in the gaps, so a job can still override any of them
through its own "env" block.
"""
import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime

ROOT = os.environ.get("ISLES_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.environ.get("ISLES_SWEEP_DIR", os.path.join(ROOT, "sweep"))
# Path-separator-joined list handed to nnUNet_extTrainer and PYTHONPATH below;
# append another directory here if you keep extra trainer classes elsewhere.
TRAINER_DIRS = os.path.join(ROOT, "training", "trainers")

QUEUE_DIR = os.path.join(WORK, "queue")
RUNNING_DIR = os.path.join(WORK, "queue_running")
DONE_DIR = os.path.join(WORK, "queue_done")
FAILED_DIR = os.path.join(WORK, "queue_failed")
LOG_DIR = os.path.join(WORK, "logs")
STATUS_FILE = os.path.join(LOG_DIR, "status.json")

for d in (QUEUE_DIR, RUNNING_DIR, DONE_DIR, FAILED_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

_claim_lock = threading.Lock()
_status_lock = threading.Lock()
STATUS = {}


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, "runner.log"), "a") as f:
        f.write(line + "\n")


def gpu_free_gb(gpu):
    """Free VRAM in GB, as reported by nvidia-smi (the cards are shared with other users)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits", "-i", str(gpu)],
            text=True, timeout=30).strip()
        used, total = [float(x) for x in out.split(",")]
        return (total - used) / 1024.0
    except Exception as e:
        log(f"nvidia-smi failed for GPU {gpu}: {e}")
        return 0.0


def write_status():
    with _status_lock:
        tmp = STATUS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(STATUS, f, indent=2, sort_keys=True)
        os.replace(tmp, STATUS_FILE)


def claim_next_job():
    """Atomically move the highest-priority pending job into queue_running/."""
    with _claim_lock:
        pending = [f for f in os.listdir(QUEUE_DIR) if f.endswith(".json")]
        if not pending:
            return None
        jobs = []
        for f in pending:
            try:
                with open(os.path.join(QUEUE_DIR, f)) as fh:
                    j = json.load(fh)
                j["_file"] = f
                jobs.append(j)
            except Exception as e:
                log(f"unreadable job {f}: {e}")
        if not jobs:
            return None
        jobs.sort(key=lambda j: (j.get("priority", 5), j.get("id", "")))
        job = jobs[0]
        src = os.path.join(QUEUE_DIR, job["_file"])
        dst = os.path.join(RUNNING_DIR, job["_file"])
        try:
            shutil.move(src, dst)
        except Exception:
            return None  # someone else took it
        return job


def build_cmd(job):
    cmd = ["nnUNetv2_train", str(job.get("dataset", 1)), job.get("config", "3d_fullres"),
           str(job.get("fold", 0)),
           "-p", job.get("plans", "nnUNetResEncUNetMPlans"),
           "-tr", job["trainer"]]
    cmd += job.get("extra_args", [])
    return cmd


def run_job(job, gpu):
    jid = job["id"]
    logfile = os.path.join(LOG_DIR, f"{jid}.log")
    env = os.environ.copy()
    # The GPU pin is the one thing the scheduler must impose.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("nnUNet_raw", os.path.join(ROOT, "nnUNet_raw"))
    env.setdefault("nnUNet_preprocessed", os.path.join(ROOT, "nnUNet_preprocessed"))
    env.setdefault("nnUNet_results", os.path.join(ROOT, "nnUNet_results"))
    # where nnU-Net looks for the sweep's own -tr classes (see training/env.sh)
    env.setdefault("nnUNet_extTrainer", TRAINER_DIRS)
    env.setdefault("nnUNet_compile", "f")
    env["nnUNet_n_proc_DA"] = str(job.get("n_proc_da", env.get("nnUNet_n_proc_DA", 12)))
    env["PYTHONPATH"] = TRAINER_DIRS + os.pathsep + env.get("PYTHONPATH", "")
    env.update(job.get("env", {}))
    cmd = build_cmd(job)
    log(f"GPU{gpu} START {jid}: {' '.join(cmd)}")
    STATUS[jid] = {"state": "running", "gpu": gpu, "started": datetime.now().isoformat(),
                   "cmd": " ".join(cmd), "log": logfile}
    write_status()
    t0 = time.time()
    with open(logfile, "w") as lf:
        lf.write(f"# {' '.join(cmd)}\n# GPU {gpu}\n# started {datetime.now()}\n\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        rc = proc.wait()
    dur = (time.time() - t0) / 3600.0
    ok = rc == 0
    log(f"GPU{gpu} {'DONE' if ok else 'FAIL'} {jid} rc={rc} after {dur:.2f}h")
    STATUS[jid] = {"state": "done" if ok else "failed", "gpu": gpu, "rc": rc,
                   "hours": round(dur, 3), "finished": datetime.now().isoformat(),
                   "cmd": " ".join(cmd), "log": logfile}
    write_status()
    fname = job.get("_file", jid + ".json")
    src = os.path.join(RUNNING_DIR, fname)
    dst = os.path.join(DONE_DIR if ok else FAILED_DIR, fname)
    if os.path.exists(src):
        shutil.move(src, dst)
    return ok


def worker(gpu, slot, stop_event, idle_exit_s):
    idle_since = time.time()
    while not stop_event.is_set():
        job = claim_next_job()
        if job is None:
            if idle_exit_s and (time.time() - idle_since) > idle_exit_s:
                log(f"GPU{gpu}/slot{slot} idle for {idle_exit_s}s, exiting")
                return
            time.sleep(20)
            continue
        idle_since = time.time()
        need = job.get("min_free_gb", 25)
        waited = 0
        while gpu_free_gb(gpu) < need and not stop_event.is_set():
            if waited % 300 == 0:
                log(f"GPU{gpu} waiting for {need}GB free (have {gpu_free_gb(gpu):.1f}GB) for {job['id']}")
            time.sleep(60)
            waited += 60
            if waited > 3600 * 6:
                log(f"GPU{gpu} gave up waiting for memory, running {job['id']} anyway")
                break
        if stop_event.is_set():
            return
        try:
            run_job(job, gpu)
        except Exception as e:
            log(f"GPU{gpu} EXCEPTION on {job.get('id')}: {e}")
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--slots-per-gpu", type=int, default=1)
    ap.add_argument("--idle-exit-s", type=int, default=0,
                    help="exit a worker after this many idle seconds (0 = never)")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    stop_event = threading.Event()
    threads = []
    for g in gpus:
        for s in range(args.slots_per_gpu):
            t = threading.Thread(target=worker, args=(g, s, stop_event, args.idle_exit_s), daemon=False)
            t.start()
            threads.append(t)
            time.sleep(3)  # stagger starts so jobs don't all hit the dataloader at once
    log(f"runner up: gpus={gpus} slots_per_gpu={args.slots_per_gpu} work_dir={WORK}")
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        stop_event.set()
        log("interrupted, waiting for workers")
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()
