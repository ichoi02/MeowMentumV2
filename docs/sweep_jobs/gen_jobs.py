"""Build sweep job lists from measured baselines.

The grid is sized in REWARD UNITS, not raw weights: each point sets
w = frac * R_task / m_baseline, so a given grid position costs the same fraction
of the baseline task reward for every term. See docs/REWARD_TUNING.md.
"""
import argparse
import json
import os
from collections import defaultdict

TERMS = ("sm", "en", "av", "jv", "time")
FRACS = (0.02, 0.06, 0.18, 0.50)
ZERO = {t: 0.0 for t in TERMS}


def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in recs if "error" not in r]


def baselines(recs):
    """Mean baseline magnitudes + task reward per variant, over all baseline seeds."""
    acc = defaultdict(list)
    for r in recs:
        if r["name"].startswith("base_"):
            acc[r["variant"]].append(r)
    out = {}
    for v, rs in acc.items():
        n = len(rs)
        out[v] = {
            "n_seeds": n,
            "task": sum(r["r_pos"] + r["r_bonus"] for r in rs) / n,
            "success": sum(r["success_pct"] for r in rs) / n,
            "success_spread": max(r["success_pct"] for r in rs) - min(r["success_pct"] for r in rs),
            **{t: sum(r[f"m_{t}"] for r in rs) / n for t in TERMS},
        }
    return out


def sig(x, n=3):
    """Round to n significant figures so job names and weights stay readable."""
    if x == 0:
        return 0.0
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))


def per_term(base):
    jobs = []
    for v, b in base.items():
        for t in TERMS:
            for frac in FRACS:
                w = sig(frac * b["task"] / b[t])
                jobs.append({
                    "name": f"{t}_{v}_{int(frac*100):02d}",
                    "variant": v,
                    "weights": {**ZERO, t: w, "notail_scale": 1.0},
                    "seed": 0,
                    "frac": frac,
                    "term": t,
                })
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="per_term")
    args = ap.parse_args()

    recs = load(args.results)
    base = baselines(recs)
    for v, b in base.items():
        print(f"{v}: {b['n_seeds']} seeds  success {b['success']:.1f}% "
              f"(spread {b['success_spread']:.1f} pp)  task {b['task']:.3f}  "
              + "  ".join(f"m_{t}={b[t]:.4g}" for t in TERMS))

    jobs = per_term(base)
    json.dump(jobs, open(args.out, "w"), indent=1)
    print(f"\n{len(jobs)} jobs -> {args.out}")
    for j in jobs:
        t = j["term"]
        print(f"  {j['name']:>16}  w_{t}={j['weights'][t]:g}")
