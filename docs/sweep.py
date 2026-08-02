"""Parallel reward-weight sweep: train a set of configs, evaluate each, log JSONL.

Each job is one (reward weights, seed) pair. A job trains a teacher from scratch
with those weights exported as CAT_W_* and then evaluates it with docs/evaluate.py,
which reports success rate *and* the unweighted penalty magnitudes (m_*). Those
magnitudes are the thing to read when tuning: a weight is doing work when the
magnitude it prices moves, and the weight is too high when success starts to fall.

Usage:
    python docs/sweep.py jobs.json --out results.jsonl --parallel 4
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_job(job, args, log_dir):
    """Train one config, evaluate it, return the merged record."""
    name = job["name"]
    env = dict(os.environ)
    env.update({f"CAT_W_{k.upper()}": str(v) for k, v in job.get("weights", {}).items()})
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    model = os.path.join(args.model_dir, f"{name}.zip")
    log = open(os.path.join(log_dir, f"{name}.log"), "w")
    t0 = time.time()

    if not (args.skip_existing and os.path.exists(model)):
        train_cmd = [
            sys.executable, "train.py",
            "--variant", job.get("variant", args.variant),
            "--steps", str(job.get("steps", args.steps)),
            "--envs", str(job.get("envs", args.envs)),
            "--gradient-steps", str(job.get("gradient_steps", args.gradient_steps)),
            "--seed", str(job.get("seed", 0)),
            "--out", model,
            "--run-name", name,
        ]
        # Control arm: train on the pre-DR 25-dim observation. evaluate.py reads
        # the width back off the checkpoint, so the eval below needs no flag.
        if not job.get("privileged", True):
            train_cmd.append("--no-privileged")
        log.write(f"$ {' '.join(train_cmd)}\n")
        log.flush()
        p = subprocess.run(train_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            log.close()
            return {"name": name, "error": f"train exited {p.returncode}", **job}
    train_s = time.time() - t0

    eval_cmd = [
        sys.executable, "docs/evaluate.py",
        "--variant", job.get("variant", args.variant),
        "--agent", "teacher",
        "--policy", model,
        "--episodes", str(args.episodes),
        "--seed", str(args.eval_seed),
        "--json",
    ]
    log.write(f"$ {' '.join(eval_cmd)}\n")
    log.flush()
    p = subprocess.run(eval_cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    log.write(p.stdout + p.stderr)
    log.close()
    if p.returncode != 0:
        return {"name": name, "error": f"eval exited {p.returncode}", **job}

    res = json.loads(p.stdout.strip().splitlines()[-1])
    return {"name": name, "train_s": round(train_s), "model": model, **job, **res}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", help="JSON file: list of {name, weights, seed, ...}")
    ap.add_argument("--out", required=True, help="JSONL results file (appended)")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--gradient-steps", type=int, default=6)
    ap.add_argument("--variant", default="tail")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--model-dir", default="sweep_models")
    ap.add_argument("--log-dir", default="sweep_logs")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse an already-trained model with the same name")
    args = ap.parse_args()

    jobs = json.load(open(args.jobs))
    os.makedirs(os.path.join(ROOT, args.model_dir), exist_ok=True)
    log_dir = os.path.join(ROOT, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    out = open(args.out, "a")

    print(f"{len(jobs)} jobs, {args.parallel} at a time, {args.steps} steps each")
    done = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for res in pool.map(lambda j: run_job(j, args, log_dir), jobs):
            done += 1
            out.write(json.dumps(res) + "\n")
            out.flush()
            if "error" in res:
                print(f"[{done}/{len(jobs)}] {res['name']}: ERROR {res['error']}", flush=True)
            else:
                print(f"[{done}/{len(jobs)}] {res['name']}: {res['success_pct']:.1f}%  "
                      f"tilt {res['mean_tilt']:.0f}  "
                      + " ".join(f"{k[2:]}={res[k]:.3g}"
                                 for k in ("m_sm", "m_en", "m_av", "m_jv", "m_time"))
                      + f"  ({res['train_s']}s)", flush=True)
    out.close()

if __name__ == "__main__":
    main()
