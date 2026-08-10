#!/usr/bin/env python3
"""Analyze motor commands over a telemetry drop.

CSV columns match hardware/controller.py:
  Cmd_F1 = rot1 (roll),  Cmd_F2 = pitch,
  Cmd_B1 = tail,         Cmd_B2 = rot2 (= -roll on the policy)

Also compares commands to encoder feedback (F_M1/F_M2/B_M2/B_M1) for tracking.

By default only the first CONTROL_DURATION (0.74 s) is used.

Usage:
  python hardware/analyze_commands.py telemetry/45deg_fail1.csv
  python hardware/analyze_commands.py telemetry/45deg_fail1.csv --save cmds.png
  python hardware/analyze_commands.py telemetry/45deg_fail1.csv --no-plot
"""

from __future__ import annotations

import argparse
import csv
import io
import os

import numpy as np

# Match hardware/controller.py / model/cat.xml jnt_range.
CONTROL_DURATION_S = 0.74
ROLL_RANGE = 7.28
PITCH_RANGE = 1.57
TAIL_RANGE = 1.9199

# (csv cmd, csv enc, label, +/- limit rad)
CHANNELS = (
    ("Cmd_F1", "F_M1", "rot1 (roll)", ROLL_RANGE),
    ("Cmd_F2", "F_M2", "pitch", PITCH_RANGE),
    ("Cmd_B1", "B_M1", "tail", TAIL_RANGE),
    ("Cmd_B2", "B_M2", "rot2 (-roll)", ROLL_RANGE),
)


def load_telemetry(path: str):
    with open(path, newline="") as f:
        raw_lines = f.readlines()

    data_lines = []
    for line in raw_lines:
        if line.lstrip().startswith("#"):
            continue
        data_lines.append(line)

    rows = list(csv.DictReader(io.StringIO("".join(data_lines))))
    if not rows:
        raise SystemExit(f"empty telemetry: {path}")

    def col(name):
        return np.array([float(r[name]) for r in rows], dtype=np.float64)

    t = col("Time")
    cmds = {name: col(name) for name, _, _, _ in CHANNELS}
    encs = {enc: col(enc) for _, enc, _, _ in CHANNELS}
    return t, cmds, encs


def summarize(name, cmd, enc, limit, dt):
    err = cmd - enc
    dcmd = np.diff(cmd) / dt if len(cmd) > 1 else np.array([0.0])
    sat = np.mean(np.abs(cmd) >= 0.98 * limit) * 100.0
    return {
        "name": name,
        "cmd_min": float(np.min(cmd)),
        "cmd_max": float(np.max(cmd)),
        "cmd_mean": float(np.mean(cmd)),
        "cmd_rms": float(np.sqrt(np.mean(cmd ** 2))),
        "cmd_peak_deg": float(np.degrees(np.max(np.abs(cmd)))),
        "rate_rms": float(np.sqrt(np.mean(dcmd ** 2))),
        "rate_peak": float(np.max(np.abs(dcmd))),
        "err_rms": float(np.sqrt(np.mean(err ** 2))),
        "err_peak": float(np.max(np.abs(err))),
        "sat_pct": float(sat),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze motor commands in telemetry")
    parser.add_argument("csv_file", help="Telemetry CSV from hardware/controller.py")
    parser.add_argument("--duration", type=float, default=CONTROL_DURATION_S,
                        help=f"Only use first N seconds (default {CONTROL_DURATION_S})")
    parser.add_argument("--save", metavar="PATH", help="Save figure instead of showing")
    parser.add_argument("--no-plot", action="store_true", help="Print stats only")
    args = parser.parse_args()

    t, cmds_raw, encs_raw = load_telemetry(args.csv_file)
    t_rel = t - t[0]
    keep = t_rel <= args.duration
    if not np.any(keep):
        raise SystemExit(f"no samples within first {args.duration:.3f}s")
    t_rel = t_rel[keep]
    cmds = {k: v[keep] for k, v in cmds_raw.items()}
    encs = {k: v[keep] for k, v in encs_raw.items()}

    dt = float(np.mean(np.diff(t_rel))) if len(t_rel) > 1 else args.duration

    print(f"file: {args.csv_file}")
    print(f"samples: {len(t_rel)}  duration: {t_rel[-1]:.3f}s  mean dt: {dt*1e3:.1f} ms"
          f"  (first {args.duration:.2f}s)")
    print()
    print(f"{'joint':<14} {'min':>8} {'max':>8} {'mean':>8} {'rms':>8} "
          f"{'|peak|°':>8} {'d/dt rms':>9} {'err rms':>8} {'sat%':>6}")
    print("-" * 90)

    for cmd_k, enc_k, label, lim in CHANNELS:
        s = summarize(label, cmds[cmd_k], encs[enc_k], lim, dt)
        print(f"{label:<14} {s['cmd_min']:+8.3f} {s['cmd_max']:+8.3f} {s['cmd_mean']:+8.3f} "
              f"{s['cmd_rms']:8.3f} {s['cmd_peak_deg']:8.1f} {s['rate_rms']:9.2f} "
              f"{s['err_rms']:8.3f} {s['sat_pct']:5.1f}%")

    # Policy sends rot2 = -rot1; check command mirror.
    mirror_err = cmds["Cmd_B2"] + cmds["Cmd_F1"]
    print()
    print(f"rot2 ≈ -rot1 check: rms(|Cmd_B2+Cmd_F1|)={np.sqrt(np.mean(mirror_err**2)):.4f} rad "
          f"(should be ~0 if policy mirror intact)")

    if args.no_plot:
        return

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(f"Motor commands: {os.path.basename(args.csv_file)}", fontsize=13)

    colors = ("C0", "C1", "C2", "C3")
    for ax, (cmd_k, enc_k, label, lim), c in zip(axs, CHANNELS, colors):
        ax.plot(t_rel, np.degrees(cmds[cmd_k]), color=c, label=f"cmd {label}", linewidth=1.5)
        ax.plot(t_rel, np.degrees(encs[enc_k]), color=c, linestyle="--", alpha=0.75,
                label=f"enc {label}")
        ax.axhline(0, color="k", linewidth=0.6, alpha=0.4)
        ax.axhline(np.degrees(lim), color="k", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axhline(-np.degrees(lim), color="k", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("deg")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_title(label, loc="left", fontsize=10)

    axs[-1].set_xlabel("Time since start (s)")
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"saved {args.save}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
