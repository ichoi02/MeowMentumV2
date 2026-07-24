#!/usr/bin/env python3
"""
Plot motor encoder angles vs time from hardware/controller.py telemetry CSV.

Example:
  python hardware/plot_telemetry_motor_angles.py telemetry_log_1774812183.csv
  python hardware/plot_telemetry_motor_angles.py log.csv -o angles.png --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIMESTAMP_COL = "timestamp_s"

DEG_COLS = ("front_m1_deg", "front_m2_deg", "back_m1_deg", "back_m2_deg")
RAD_COLS = ("front_m1_rad", "front_m2_rad", "back_m1_rad", "back_m2_rad")
LABELS = ("Front M1", "Front M2", "Back M1", "Back M2")

# Distinct, colorblind-friendly hues
COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7")


def _angles_deg(df: pd.DataFrame) -> pd.DataFrame:
    if all(c in df.columns for c in DEG_COLS):
        return df[list(DEG_COLS)].apply(pd.to_numeric, errors="coerce")
    if all(c in df.columns for c in RAD_COLS):
        rads = df[list(RAD_COLS)].apply(pd.to_numeric, errors="coerce")
        return pd.DataFrame(
            {d: np.degrees(rads[r].to_numpy()) for d, r in zip(DEG_COLS, RAD_COLS)}
        )
    raise SystemExit(
        f"CSV must include either {list(DEG_COLS)} or {list(RAD_COLS)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "csv",
        type=Path,
        help="Telemetry CSV (e.g. telemetry_log_*.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save figure to this path (PNG or other matplotlib-supported format)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (use with -o on headless systems)",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Use four stacked subplots instead of one combined chart",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Not a file: {args.csv}")

    df = pd.read_csv(args.csv)
    if TIMESTAMP_COL not in df.columns:
        raise SystemExit(f"CSV missing {TIMESTAMP_COL!r}")

    t = pd.to_numeric(df[TIMESTAMP_COL], errors="coerce")
    angles = _angles_deg(df)

    if args.grid:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8), constrained_layout=True)
        for ax, col, label, color in zip(axes, DEG_COLS, LABELS, COLORS):
            ax.plot(t, angles[col], color=color, linewidth=1.5, label=label)
            ax.set_ylabel("deg")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Motor encoder angle vs time")
    else:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for col, label, color in zip(DEG_COLS, LABELS, COLORS):
            ax.plot(t, angles[col], color=color, linewidth=1.5, label=label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg)")
        ax.set_title("Motor encoder angles vs time")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    if args.output is not None:
        fig.savefig(args.output, dpi=150)
        print(f"Wrote {args.output}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
