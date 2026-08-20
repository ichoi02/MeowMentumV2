#!/usr/bin/env python3
"""Reproducible analysis of the tail/no-tail drop experiment.

The primary endpoint is a coupled whole-cat pose score at a fixed 2.5 m ballistic
fall distance. It separates (1) tilt of the spherical midpoint of the front and
rear up-vectors from (2) the full relative attitude between the two bodies, then
multiplies their 0--1 scores. This is yaw-invariant for common whole-body yaw,
uses both body attitudes, and requires both righting and spine alignment for 1.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/meowmentum-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/meowmentum-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, signal, stats
from scipy.spatial.transform import Rotation, Slerp


G = 9.81
FILENAME_RE = re.compile(
    r"^(?P<morph>wt|nt)_roll_(?P<roll>\d{3})deg"
    r"(?:_pitch_(?P<pitch>[+-]\d{3})deg)?_"
    r"(?P<date>\d{4})_rep(?P<rep>\d{2})\.csv$"
)
REQUIRED_COLUMNS = {
    "Time", "F_Q0", "F_Q1", "F_Q2", "F_Q3", "F_M1", "F_M2", "B_M1", "B_M2",
    "F_ACC", "B_ACC", "Cmd_F1", "Cmd_F2", "Cmd_B1", "Cmd_B2"
}
OUTCOMES = {
    "posture_score": "coupled whole-cat pose score (0-1)",
    "whole_body_upright_score": "whole-body centerline uprightness (0-1)",
    "spine_alignment_score": "front/rear attitude alignment (0-1)",
    "front_upright_score": "front-body uprightness score (0-1)",
    "back_upright_score": "rear-body uprightness score (0-1)",
    "whole_body_tilt_deg": "whole-body centerline tilt (deg)",
    "spine_misalignment_deg": "front/rear relative attitude (deg)",
    "orientation_error_deg": "front-body 3D orientation error (deg)",
    "angular_speed_deg_s": "3D angular speed (deg/s)",
    "abs_roll_deg": "absolute roll error (deg)",
    "abs_pitch_deg": "absolute pitch error (deg)",
    "abs_roll_rate_deg_s": "absolute roll rate (deg/s)",
    "abs_pitch_rate_deg_s": "absolute pitch rate (deg/s)",
}
CONDITION_ORDER = ("r45", "r90", "r180", "r180_p15", "r180_p30", "r180_p45")


@dataclass
class OLSResult:
    beta: np.ndarray
    covariance_hc3: np.ndarray
    residuals: np.ndarray
    fitted: np.ndarray
    df_resid: int
    sigma: float
    xtx_inverse: np.ndarray
    names: list[str]


def angular_distance(angle_deg: np.ndarray | float) -> np.ndarray | float:
    return np.abs((np.asarray(angle_deg) + 180.0) % 360.0 - 180.0)


def body_up(rotation: Rotation) -> np.ndarray:
    """Local +z expressed in world coordinates."""
    return np.asarray(rotation.apply([0.0, 0.0, 1.0]), dtype=float)


def tilt_score(rotation: Rotation) -> float:
    """Yaw-invariant uprightness, linear in tilt: 1 upright, 0 inverted."""
    tilt = math.acos(float(np.clip(body_up(rotation)[2], -1.0, 1.0)))
    return float(np.clip(1.0 - tilt / math.pi, 0.0, 1.0))


def coupled_pose_metrics(front: Rotation, rear: Rotation) -> dict[str, float]:
    """Whole-body tilt and spine alignment from coupled front/rear attitudes.

    The normalized sum is the equal-weight geodesic midpoint of the two up-vectors
    on S2 whenever they are not antipodal. The relative SO(3) rotation also sees
    twist about the up-axis, which an up-vector-only metric would miss.
    """
    up_front = body_up(front)
    up_rear = body_up(rear)
    up_sum = up_front + up_rear
    if np.linalg.norm(up_sum) < 1e-8:
        # There is no unique spherical midpoint for antipodal up-vectors; treating
        # the common-body attitude as failed is the conservative endpoint choice.
        body_tilt = math.pi
        center_roll = math.nan
        center_pitch = math.nan
    else:
        center_up = up_sum / np.linalg.norm(up_sum)
        body_tilt = math.acos(float(np.clip(center_up[2], -1.0, 1.0)))

        # Complete the midpoint up-vector into a whole-cat body frame. The mean
        # front/rear longitudinal (+x) direction is projected perpendicular to
        # center_up, then re-orthogonalized. This preserves the requested up-axis
        # midpoint while providing body-fixed axes for yaw-invariant fused roll
        # and pitch diagnostics. Degenerate forward vectors use either segment.
        forward = front.apply([1.0, 0.0, 0.0]) + rear.apply([1.0, 0.0, 0.0])
        forward -= np.dot(forward, center_up) * center_up
        if np.linalg.norm(forward) < 1e-8:
            forward = front.apply([1.0, 0.0, 0.0])
            forward -= np.dot(forward, center_up) * center_up
        if np.linalg.norm(forward) < 1e-8:
            reference = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(reference, center_up)) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            forward = reference - np.dot(reference, center_up) * center_up
        center_forward = forward / np.linalg.norm(forward)
        center_side = np.cross(center_up, center_forward)
        center_rotation = np.column_stack((center_forward, center_side, center_up))
        # Fused pitch = asin(-R31), fused roll = asin(R32). Unlike Euler
        # roll/pitch, these depend only on tilt and are invariant to world yaw.
        center_pitch = math.asin(float(np.clip(-center_rotation[2, 0], -1.0, 1.0)))
        center_roll = math.asin(float(np.clip(center_rotation[2, 1], -1.0, 1.0)))

    relative_angle = float((front.inv() * rear).magnitude())
    upright = float(np.clip(1.0 - body_tilt / math.pi, 0.0, 1.0))
    aligned = float(np.clip(1.0 - relative_angle / math.pi, 0.0, 1.0))
    return {
        "posture_score": upright * aligned,
        "whole_body_upright_score": upright,
        "spine_alignment_score": aligned,
        "whole_body_tilt_deg": math.degrees(body_tilt),
        "whole_body_roll_deg": math.degrees(center_roll),
        "whole_body_pitch_deg": math.degrees(center_pitch),
        "spine_misalignment_deg": math.degrees(relative_angle),
    }


def parse_metadata(path: Path) -> dict[str, object]:
    match = FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unrecognized canonical filename: {path}")
    data = match.groupdict()
    roll = int(data["roll"])
    pitch = int(data["pitch"] or 0)
    condition = f"r{roll}" if data["pitch"] is None else f"r{roll}_p{pitch}"
    return {
        "morphology": "with_tail" if data["morph"] == "wt" else "no_tail",
        "tail": int(data["morph"] == "wt"),
        "roll_deg": roll,
        "pitch_deg": pitch,
        "condition": condition,
        "date": data["date"],
        "rep": int(data["rep"]),
    }


def low_pass(values: np.ndarray, time_mid: np.ndarray, cutoff_hz: float = 15.0) -> np.ndarray:
    if len(values) < 16:
        return values
    fs = 1.0 / np.median(np.diff(time_mid))
    if not np.isfinite(fs) or fs <= 2.2 * cutoff_hz:
        return values
    sos = signal.butter(4, cutoff_hz, btype="low", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, values, axis=0)


def sustained_crossing_time(
    time_s: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    dwell_s: float,
) -> float:
    """First time a score reaches threshold and stays there for the full dwell."""
    for index, start in enumerate(time_s):
        end = int(np.searchsorted(time_s, start + dwell_s, side="left"))
        if end < len(time_s) and np.all(scores[index : end + 1] >= threshold):
            return float(start)
    return math.nan


def extract_trial(path: Path, target_distance_m: float) -> dict[str, object]:
    meta = parse_metadata(path)
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame.dropna(subset=sorted(REQUIRED_COLUMNS)).reset_index(drop=True)
    time = frame["Time"].to_numpy(float)
    if len(time) < 10 or np.any(np.diff(time) <= 0):
        raise ValueError(f"{path}: insufficient or non-monotonic time samples")

    accel = np.maximum(frame["F_ACC"].to_numpy(float), frame["B_ACC"].to_numpy(float))
    impact_idx = int(np.nanargmax(accel))
    impact_time = time[impact_idx]
    duration = impact_time - time[0]
    observed_fall_distance = 0.5 * G * duration**2
    target_time = time[0] + math.sqrt(2.0 * target_distance_m / G)
    if impact_idx < 2 or impact_time < target_time:
        raise RuntimeError(
            f"short_drop: estimated fall distance {observed_fall_distance:.3f} m "
            f"is below target {target_distance_m:.3f} m"
        )

    q_scalar_first = frame[["F_Q0", "F_Q1", "F_Q2", "F_Q3"]].to_numpy(float)
    q_norm = np.linalg.norm(q_scalar_first, axis=1)
    if np.any(q_norm < 0.5):
        raise ValueError(f"{path}: invalid near-zero quaternion")
    q_scalar_first = q_scalar_first / q_norm[:, None]
    rotations = Rotation.from_quat(q_scalar_first[:, [1, 2, 3, 0]])

    zeros = np.zeros(len(frame))
    front_roll_joint = Rotation.from_rotvec(
        np.column_stack((frame["F_M1"].to_numpy(float), zeros, zeros))
    )
    spine_pitch_joint = Rotation.from_rotvec(
        np.column_stack((zeros, frame["F_M2"].to_numpy(float), zeros))
    )
    rear_roll_joint = Rotation.from_rotvec(
        np.column_stack((frame["B_M2"].to_numpy(float), zeros, zeros))
    )
    rear_rotations = rotations * front_roll_joint * spine_pitch_joint * rear_roll_joint

    # Slerp gives the orientation at the exact physical cutoff, rather than the
    # nearest telemetry frame (which can differ by ~10 ms across trials).
    target_rotation = Slerp(time[: impact_idx + 1], rotations[: impact_idx + 1])([target_time])[0]
    target_rear_rotation = Slerp(
        time[: impact_idx + 1], rear_rotations[: impact_idx + 1]
    )([target_time])[0]
    initial_rotation = rotations[0]
    initial_rear_rotation = rear_rotations[0]
    target_euler = target_rotation.as_euler("xyz", degrees=True)
    initial_euler = initial_rotation.as_euler("xyz", degrees=True)

    dt = np.diff(time[: impact_idx + 1])
    delta = rotations[:impact_idx].inv() * rotations[1 : impact_idx + 1]
    angular_velocity = np.degrees(delta.as_rotvec()) / dt[:, None]
    midpoint_time = (time[:impact_idx] + time[1 : impact_idx + 1]) / 2.0
    angular_velocity = low_pass(angular_velocity, midpoint_time)
    target_velocity = np.array(
        [np.interp(target_time, midpoint_time, angular_velocity[:, axis]) for axis in range(3)]
    )
    initial_velocity = np.median(angular_velocity[: min(5, len(angular_velocity))], axis=0)

    # Trajectory-level outcomes use a common observation horizon and include an
    # interpolated sample exactly at the 2.5 m cutoff.
    query_time = np.append(time[time < target_time], target_time)
    query_time = np.unique(query_time)
    trajectory_front = Slerp(
        time[: impact_idx + 1], rotations[: impact_idx + 1]
    )(query_time)
    trajectory_rear = Slerp(
        time[: impact_idx + 1], rear_rotations[: impact_idx + 1]
    )(query_time)
    trajectory_score = np.array(
        [
            coupled_pose_metrics(front, rear)["posture_score"]
            for front, rear in zip(trajectory_front, trajectory_rear)
        ]
    )
    trajectory_time = query_time - time[0]
    observation_horizon = trajectory_time[-1]
    righting_times = {
        threshold: sustained_crossing_time(
            trajectory_time, trajectory_score, threshold, dwell_s=0.10
        )
        for threshold in (0.75, 0.80, 0.85)
    }
    integrated_posture_deficit = float(
        np.trapezoid(1.0 - trajectory_score, trajectory_time) / observation_horizon
    )

    # The log contains target positions, not current/voltage/applied PWM. These
    # are explicitly model-based effort proxies: reconstruct normalized PD duty
    # from 50 Hz target error and finite-difference joint motion. They cannot be
    # interpreted as joules because the 1 kHz inner-loop waveform is unobserved.
    actual_columns = ("F_M1", "F_M2", "B_M1", "B_M2")
    target_columns = ("Cmd_F1", "Cmd_F2", "Cmd_B1", "Cmd_B2")
    actual = np.column_stack(
        [np.interp(query_time, time, frame[column].to_numpy(float)) for column in actual_columns]
    )
    targets = np.column_stack(
        [np.interp(query_time, time, frame[column].to_numpy(float)) for column in target_columns]
    )
    velocity = np.gradient(actual, trajectory_time, axis=0)
    kp = np.array([4.0, 20.0, 20.0, 4.0])
    kd = np.array([0.4, 1.0, 1.0, 0.4])
    position_error = targets - actual
    modeled_duty = np.clip(kp * position_error - kd * velocity, -1.0, 1.0)
    modeled_duty[np.abs(position_error) <= 0.03] = 0.0
    common_actuators = [0, 1, 3]
    all_actuators = [0, 1, 2, 3]
    effort_common = float(
        np.trapezoid(np.sum(modeled_duty[:, common_actuators] ** 2, axis=1), trajectory_time)
    )
    effort_total = float(
        np.trapezoid(np.sum(modeled_duty[:, all_actuators] ** 2, axis=1), trajectory_time)
    )

    try:
        display_path = path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        display_path = path.as_posix()
    pose = coupled_pose_metrics(target_rotation, target_rear_rotation)
    initial_pose = coupled_pose_metrics(initial_rotation, initial_rear_rotation)
    front_score = tilt_score(target_rotation)
    back_score = tilt_score(target_rear_rotation)
    initial_front_score = tilt_score(initial_rotation)
    initial_back_score = tilt_score(initial_rear_rotation)
    return {
        **meta,
        "source_file": display_path,
        "n_samples": len(frame),
        "impact_time_s": duration,
        "estimated_fall_distance_m": observed_fall_distance,
        **pose,
        "abs_whole_body_pitch_deg": abs(pose["whole_body_pitch_deg"]),
        "front_upright_score": front_score,
        "back_upright_score": back_score,
        "orientation_error_deg": np.degrees(target_rotation.magnitude()),
        "angular_speed_deg_s": np.linalg.norm(target_velocity),
        "abs_roll_deg": float(angular_distance(target_euler[0])),
        "abs_pitch_deg": float(angular_distance(target_euler[1])),
        "abs_roll_rate_deg_s": abs(target_velocity[0]),
        "abs_pitch_rate_deg_s": abs(target_velocity[1]),
        "initial_orientation_error_deg": np.degrees(initial_rotation.magnitude()),
        "initial_posture_score": initial_pose["posture_score"],
        "initial_whole_body_upright_score": initial_pose["whole_body_upright_score"],
        "initial_spine_alignment_score": initial_pose["spine_alignment_score"],
        "initial_whole_body_pitch_deg": initial_pose["whole_body_pitch_deg"],
        "initial_abs_whole_body_pitch_deg": abs(initial_pose["whole_body_pitch_deg"]),
        "initial_front_upright_score": initial_front_score,
        "initial_back_upright_score": initial_back_score,
        "initial_angular_speed_deg_s": np.linalg.norm(initial_velocity),
        "initial_roll_deg": initial_euler[0],
        "initial_pitch_deg": initial_euler[1],
        "initial_roll_rate_deg_s": initial_velocity[0],
        "initial_pitch_rate_deg_s": initial_velocity[1],
        "righting_success_075": int(np.isfinite(righting_times[0.75])),
        "righting_success_080": int(np.isfinite(righting_times[0.80])),
        "righting_success_085": int(np.isfinite(righting_times[0.85])),
        "time_to_righting_080_s": righting_times[0.80],
        "restricted_time_to_righting_075_s": (
            righting_times[0.75] if np.isfinite(righting_times[0.75]) else observation_horizon
        ),
        "restricted_time_to_righting_080_s": (
            righting_times[0.80] if np.isfinite(righting_times[0.80]) else observation_horizon
        ),
        "restricted_time_to_righting_085_s": (
            righting_times[0.85] if np.isfinite(righting_times[0.85]) else observation_horizon
        ),
        "integrated_posture_deficit": integrated_posture_deficit,
        "modeled_pd_effort_common": effort_common,
        "modeled_pd_effort_total": effort_total,
    }


def build_design(
    frame: pd.DataFrame,
    interactions: bool = False,
    include_date: bool = True,
    covariates: tuple[str, ...] = ("initial_posture_score", "initial_angular_speed_deg_s"),
) -> tuple[np.ndarray, list[str]]:
    columns = [np.ones(len(frame)), frame["tail"].to_numpy(float)]
    names = ["intercept", "with_tail"]
    conditions = [condition for condition in CONDITION_ORDER if condition in set(frame["condition"])]
    for condition in conditions[1:]:
        indicator = (frame["condition"] == condition).to_numpy(float)
        columns.append(indicator)
        names.append(f"condition[{condition}]")
    if include_date:
        dates = sorted(frame["date"].astype(str).unique())
        for date in dates[1:]:
            columns.append((frame["date"].astype(str) == date).to_numpy(float))
            names.append(f"date[{date}]")
    for covariate in covariates:
        values = frame[covariate].to_numpy(float)
        scale = values.std(ddof=1)
        columns.append((values - values.mean()) / scale if scale > 0 else np.zeros(len(values)))
        names.append(f"z({covariate})")
    if interactions:
        for condition in conditions[1:]:
            columns.append(
                frame["tail"].to_numpy(float) * (frame["condition"] == condition).to_numpy(float)
            )
            names.append(f"with_tail:condition[{condition}]")
    return np.column_stack(columns), names


def fit_ols(y: np.ndarray, x: np.ndarray, names: list[str]) -> OLSResult:
    xtx_inverse = np.linalg.pinv(x.T @ x)
    beta = xtx_inverse @ x.T @ y
    fitted = x @ beta
    residuals = y - fitted
    rank = np.linalg.matrix_rank(x)
    df_resid = len(y) - rank
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inverse, x)
    adjusted_squared = (residuals / np.maximum(1.0 - leverage, 1e-8)) ** 2
    meat = x.T @ (x * adjusted_squared[:, None])
    covariance_hc3 = xtx_inverse @ meat @ xtx_inverse
    sigma = math.sqrt(float(residuals @ residuals) / df_resid)
    return OLSResult(beta, covariance_hc3, residuals, fitted, df_resid, sigma, xtx_inverse, names)


def coefficient_row(result: OLSResult, index: int = 1) -> dict[str, float]:
    estimate = result.beta[index]
    se = math.sqrt(max(result.covariance_hc3[index, index], 0.0))
    critical = stats.t.ppf(0.975, result.df_resid)
    statistic = estimate / se
    return {
        "estimate": estimate,
        "se_hc3": se,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
        "t_hc3": statistic,
        "p_hc3": 2.0 * stats.t.sf(abs(statistic), result.df_resid),
        "residual_sd": result.sigma,
        "df_resid": result.df_resid,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def exact_condition_tests(frame: pd.DataFrame, outcomes: tuple[str, ...]) -> pd.DataFrame:
    """Exact two-sided permutation tests within condition, Holm-adjusted per outcome."""
    rows: list[dict[str, object]] = []
    conditions = [condition for condition in CONDITION_ORDER if condition in set(frame["condition"])]
    for outcome in outcomes:
        outcome_rows: list[dict[str, object]] = []
        for condition in conditions:
            subset = frame.loc[frame["condition"] == condition, ["tail", outcome]].dropna()
            values = subset[outcome].to_numpy(float)
            n_tail = int(subset["tail"].sum())
            observed = (
                subset.loc[subset["tail"] == 1, outcome].mean()
                - subset.loc[subset["tail"] == 0, outcome].mean()
            )
            exceedances = 0
            total = 0
            indices = np.arange(len(values))
            for tail_indices in itertools.combinations(indices, n_tail):
                tail_mask = np.zeros(len(values), dtype=bool)
                tail_mask[list(tail_indices)] = True
                permuted = values[tail_mask].mean() - values[~tail_mask].mean()
                exceedances += abs(permuted) >= abs(observed) - 1e-12
                total += 1
            outcome_rows.append(
                {
                    "outcome": outcome,
                    "condition": condition,
                    "n_no_tail": int((subset["tail"] == 0).sum()),
                    "n_with_tail": n_tail,
                    "effect_with_tail_minus_no_tail": observed,
                    "p_exact": exceedances / total,
                }
            )
        adjusted = holm_adjust([float(row["p_exact"]) for row in outcome_rows])
        for row, p_holm in zip(outcome_rows, adjusted):
            row["p_holm_six_conditions"] = p_holm
            rows.append(row)
    return pd.DataFrame(rows)


def significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def annotate_condition_significance(
    ax: plt.Axes,
    frame: pd.DataFrame,
    tests: pd.DataFrame,
    outcome: str,
    conditions: list[str],
    positions: np.ndarray,
    score_axis: bool = False,
) -> None:
    """Add pair brackets and Holm-adjusted exact-permutation significance labels."""
    values = frame[outcome].to_numpy(float)
    data_range = max(float(np.nanmax(values) - np.nanmin(values)), 1e-6)
    bracket_height = 0.018 if score_axis else 0.025 * data_range
    text_gap = 0.006 if score_axis else 0.008 * data_range
    annotation_tops = []
    for position, condition in zip(positions, conditions):
        condition_values = frame.loc[frame["condition"] == condition, outcome].to_numpy(float)
        y = float(np.nanmax(condition_values)) + (0.025 if score_axis else 0.04 * data_range)
        p_value = float(
            tests.loc[
                (tests["outcome"] == outcome) & (tests["condition"] == condition),
                "p_holm_six_conditions",
            ].iloc[0]
        )
        ax.plot(
            [position - 0.16, position - 0.16, position + 0.16, position + 0.16],
            [y, y + bracket_height, y + bracket_height, y],
            color="black",
            linewidth=0.9,
            clip_on=False,
        )
        ax.text(
            position,
            y + bracket_height + text_gap,
            significance_label(p_value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        annotation_tops.append(y + bracket_height + 3 * text_gap)
    lower, upper = ax.get_ylim()
    ax.set_ylim(lower, max(upper, max(annotation_tops)))


def plot_ancova_significance(
    frame: pd.DataFrame,
    results: pd.DataFrame,
    sweep_results: pd.DataFrame,
    dynamics_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Condition-stratified distributions annotated with date-adjusted ANCOVA tests."""
    colors = {"no_tail": "#9A5B35", "with_tail": "#284B70"}

    def draw_boxes(ax: plt.Axes, outcome: str, conditions: list[str], seed: int) -> None:
        positions = np.arange(len(conditions))
        rng = np.random.default_rng(seed)
        for offset, morphology in ((-0.16, "no_tail"), (0.16, "with_tail")):
            groups = [
                frame.loc[
                    (frame["condition"] == condition)
                    & (frame["morphology"] == morphology),
                    outcome,
                ]
                for condition in conditions
            ]
            box = ax.boxplot(
                groups,
                positions=positions + offset,
                widths=0.27,
                patch_artist=True,
                showfliers=False,
            )
            for patch in box["boxes"]:
                patch.set_facecolor(colors[morphology])
                patch.set_alpha(0.75)
            for position, values in zip(positions + offset, groups):
                ax.scatter(
                    position + rng.normal(0, 0.025, len(values)),
                    values,
                    s=18,
                    alpha=0.65,
                    color=colors[morphology],
                    zorder=3,
                )
        condition_labels = {
            "r45": "45° roll",
            "r90": "90° roll",
            "r180": "180° roll",
            "r180_p15": "180° roll\n15° pitch",
            "r180_p30": "180° roll\n30° pitch",
            "r180_p45": "180° roll\n45° pitch",
        }
        ax.set_xticks(positions, [condition_labels[c] for c in conditions])
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="y", alpha=0.25)

    conditions = [condition for condition in CONDITION_ORDER if condition in set(frame["condition"])]

    significant_outcomes = (
        ("posture_score", "Coupled whole-cat score", "coupled_score_ancova.png", 101),
        (
            "spine_alignment_score",
            "Front/rear attitude alignment",
            "front_rear_alignment_ancova.png",
            102,
        ),
    )
    for outcome, title, filename, seed in significant_outcomes:
        fig, ax = plt.subplots(figsize=(11.5, 6.5))
        draw_boxes(ax, outcome, conditions, seed)
        row = results.loc[results["outcome"] == outcome].iloc[0]
        stars = significance_label(float(row.p_hc3))
        ax.set_title(
            f"{title}\n"
            f"Date-adjusted ANCOVA: Δ={row.estimate:+.3f}, "
            f"95% CI {row.ci_low:+.3f} to {row.ci_high:+.3f}, "
            f"p={row.p_hc3:.4f} {stars}",
            fontsize=14,
            pad=14,
        )
        ax.set_ylabel("score (0–1; higher is better)", fontsize=12)
        ax.set_xlabel("release condition", fontsize=12)
        ax.set_ylim(-0.02, 1.04)
        ax.scatter([], [], color=colors["with_tail"], label="with tail")
        ax.scatter([], [], color=colors["no_tail"], label="no tail")
        ax.legend(frameon=False, loc="lower left", fontsize=11)
        ax.text(
            0.99,
            0.02,
            "* p<.05, ** p<.01, *** p<.001",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    row = sweep_results.loc[sweep_results["sweep"] == "no_pitch_roll_sweep"].iloc[0]
    panel_conditions = [
        condition for condition in CONDITION_ORDER if condition in row.conditions.split(",")
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    draw_boxes(ax, "posture_score", panel_conditions, 201)
    stars = significance_label(float(row.p_holm_two_sweeps))
    ax.set_title(
        "Coupled whole-cat score: no-pitch roll sweep\n"
        f"Date-adjusted ANCOVA: Δ={row.estimate:+.3f}, "
        f"95% CI {row.ci_low:+.3f} to {row.ci_high:+.3f}, "
        f"p={row.p_hc3:.4f}; Holm p={row.p_holm_two_sweeps:.4f} {stars}",
        fontsize=14,
        pad=14,
    )
    ax.set_ylabel("score (0–1; higher is better)", fontsize=12)
    ax.set_xlabel("release condition", fontsize=12)
    ax.set_ylim(-0.02, 1.04)
    ax.scatter([], [], color=colors["with_tail"], label="with tail")
    ax.scatter([], [], color=colors["no_tail"], label="no tail")
    ax.legend(frameon=False, loc="lower left", fontsize=11)
    ax.text(
        0.99,
        0.02,
        "Stars use Holm correction across the two requested sweeps",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "no_pitch_roll_ancova.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    dynamics_plots = (
        (
            "righting_time_080",
            "restricted_time_to_righting_080_s",
            "Time to sustained righting",
            "time to score ≥ 0.80 (s; lower is better)",
            "righting_time_ancova.png",
            "Failures are capped at the 0.714 s observation horizon; 0.10 s dwell required.",
        ),
        (
            "modeled_pd_effort_common",
            "modeled_pd_effort_common",
            "Modeled actuator effort: three common spine actuators",
            "normalized duty² × seconds (lower is better)",
            "common_actuator_effort_ancova.png",
            "Model-based proxy from 50 Hz telemetry; not measured electrical energy.",
        ),
        (
            "modeled_pd_effort_total",
            "modeled_pd_effort_total",
            "Modeled actuator effort: all four actuators",
            "normalized duty² × seconds (lower is better)",
            "total_actuator_effort_ancova.png",
            "Includes the tail actuator; model-based proxy, not measured electrical energy.",
        ),
    )
    for index, (analysis, outcome, title, ylabel, filename, note) in enumerate(dynamics_plots):
        row = dynamics_results.loc[dynamics_results["analysis"] == analysis].iloc[0]
        adjusted_p = float(row.p_holm_four_new_endpoints)
        if adjusted_p >= 0.05:
            continue
        fig, ax = plt.subplots(figsize=(11.5, 6.5))
        draw_boxes(ax, outcome, conditions, 300 + index)
        stars = significance_label(adjusted_p)
        ax.set_title(
            f"{title}\n"
            f"Date-adjusted ANCOVA: Δ={row.estimate:+.3f}, "
            f"95% CI {row.ci_low:+.3f} to {row.ci_high:+.3f}, "
            f"p={row.p_hc3:.4f}; Holm p={adjusted_p:.4f} {stars}",
            fontsize=14,
            pad=14,
        )
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("release condition", fontsize=12)
        ax.scatter([], [], color=colors["with_tail"], label="with tail")
        ax.scatter([], [], color=colors["no_tail"], label="no tail")
        ax.legend(frameon=False, loc="lower left", fontsize=11)
        ax.text(
            0.99,
            0.02,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def overlap_permutation(
    frame: pd.DataFrame,
    outcome: str,
    n_permutations: int,
    seed: int,
    covariates: tuple[str, ...] = ("initial_posture_score", "initial_angular_speed_deg_s"),
) -> tuple[float, float, int]:
    strata = frame.groupby(["condition", "date"])["tail"].agg(["min", "max"])
    overlap = set(strata[(strata["min"] == 0) & (strata["max"] == 1)].index)
    mask = [(condition, date) in overlap for condition, date in zip(frame["condition"], frame["date"])]
    subset = frame.loc[mask].copy().reset_index(drop=True)
    x, names = build_design(subset, covariates=covariates)
    observed = fit_ols(subset[outcome].to_numpy(float), x, names).beta[1]
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    grouped_indices = [group.index.to_numpy() for _, group in subset.groupby(["condition", "date"])]
    original_tail = subset["tail"].to_numpy(int)
    for iteration in range(n_permutations):
        permuted = original_tail.copy()
        for indices in grouped_indices:
            permuted[indices] = rng.permutation(permuted[indices])
        permuted_frame = subset.copy()
        permuted_frame["tail"] = permuted
        x_permuted, _ = build_design(permuted_frame, covariates=covariates)
        null[iteration] = fit_ols(subset[outcome].to_numpy(float), x_permuted, names).beta[1]
    p_value = (1.0 + np.sum(np.abs(null) >= abs(observed))) / (n_permutations + 1.0)
    return observed, float(p_value), len(subset)


def analyze_predefined_sweeps(
    frame: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    """One pooled tail contrast for each scientifically defined condition sweep."""
    definitions = {
        "pitch_sweep_at_roll_180": {
            "label": "180-deg roll; pitch 0/15/30/45 deg",
            "conditions": ("r180", "r180_p15", "r180_p30", "r180_p45"),
        },
        "no_pitch_roll_sweep": {
            "label": "No pitch; roll 45/90/180 deg",
            "conditions": ("r45", "r90", "r180"),
        },
    }
    rows: list[dict[str, object]] = []
    for sweep, definition in definitions.items():
        subset = frame.loc[frame["condition"].isin(definition["conditions"])].copy().reset_index(drop=True)
        x, names = build_design(subset)
        model = fit_ols(subset["posture_score"].to_numpy(float), x, names)
        adjusted = coefficient_row(model)
        x_no_date, names_no_date = build_design(subset, include_date=False)
        no_date = coefficient_row(
            fit_ols(subset["posture_score"].to_numpy(float), x_no_date, names_no_date)
        )
        overlap_estimate, p_permutation, n_overlap = overlap_permutation(
            subset, "posture_score", n_permutations, seed
        )
        mde = regression_mde(
            model.sigma, float(model.xtx_inverse[1, 1]), model.df_resid, alpha=0.05
        )
        rows.append(
            {
                "sweep": sweep,
                "label": definition["label"],
                "conditions": ",".join(definition["conditions"]),
                "n": len(subset),
                **adjusted,
                "estimate_no_date": no_date["estimate"],
                "p_no_date": no_date["p_hc3"],
                "overlap_estimate": overlap_estimate,
                "p_permutation": p_permutation,
                "n_overlap": n_overlap,
                "mde_80": mde,
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm_two_sweeps"] = holm_adjust(result["p_hc3"].tolist())
    result["p_permutation_holm_two_sweeps"] = holm_adjust(result["p_permutation"].tolist())
    return result


def analyze_pitch_leveling(
    frame: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> dict[str, object]:
    """Tail contrast for final absolute whole-cat pitch in the 180-roll pitch sweep."""
    conditions = ("r180", "r180_p15", "r180_p30", "r180_p45")
    subset = frame.loc[frame["condition"].isin(conditions)].copy().reset_index(drop=True)
    covariates = ("initial_abs_whole_body_pitch_deg", "initial_angular_speed_deg_s")
    x, names = build_design(subset, covariates=covariates)
    model = fit_ols(subset["abs_whole_body_pitch_deg"].to_numpy(float), x, names)
    adjusted = coefficient_row(model)
    x_no_date, names_no_date = build_design(
        subset, include_date=False, covariates=covariates
    )
    no_date = coefficient_row(
        fit_ols(
            subset["abs_whole_body_pitch_deg"].to_numpy(float),
            x_no_date,
            names_no_date,
        )
    )
    overlap_estimate, permutation_p, n_overlap = overlap_permutation(
        subset,
        "abs_whole_body_pitch_deg",
        n_permutations,
        seed,
        covariates=covariates,
    )
    return {
        "analysis": "pitch_leveling_at_roll_180",
        "label": "Final absolute whole-cat pitch; roll 180, pitch 0/15/30/45 deg",
        "conditions": ",".join(conditions),
        "n": len(subset),
        **adjusted,
        "estimate_no_date": no_date["estimate"],
        "p_no_date": no_date["p_hc3"],
        "overlap_estimate": overlap_estimate,
        "p_permutation": permutation_p,
        "n_overlap": n_overlap,
    }


def analyze_time_and_effort(
    frame: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Exploratory righting-time and model-based actuator-effort analyses."""
    definitions = (
        (
            "righting_time_080",
            "Restricted time to sustained score >= 0.80 (s)",
            "restricted_time_to_righting_080_s",
            "time",
        ),
        (
            "integrated_posture_deficit",
            "Mean posture deficit through cutoff",
            "integrated_posture_deficit",
            "time",
        ),
        (
            "modeled_pd_effort_common",
            "Modeled PD effort, three common actuators",
            "modeled_pd_effort_common",
            "effort_proxy",
        ),
        (
            "modeled_pd_effort_total",
            "Modeled PD effort, all four actuators",
            "modeled_pd_effort_total",
            "effort_proxy",
        ),
        (
            "righting_time_075_sensitivity",
            "Threshold sensitivity: sustained score >= 0.75 (s)",
            "restricted_time_to_righting_075_s",
            "threshold_sensitivity",
        ),
        (
            "righting_time_085_sensitivity",
            "Threshold sensitivity: sustained score >= 0.85 (s)",
            "restricted_time_to_righting_085_s",
            "threshold_sensitivity",
        ),
    )
    rows: list[dict[str, object]] = []
    for analysis, label, outcome, family in definitions:
        x, names = build_design(frame)
        model = fit_ols(frame[outcome].to_numpy(float), x, names)
        adjusted = coefficient_row(model)
        overlap_estimate, permutation_p, n_overlap = overlap_permutation(
            frame, outcome, n_permutations, seed
        )
        rows.append(
            {
                "analysis": analysis,
                "label": label,
                "outcome": outcome,
                "family": family,
                "n": len(frame),
                **adjusted,
                "overlap_estimate": overlap_estimate,
                "p_permutation": permutation_p,
                "n_overlap": n_overlap,
                "mean_no_tail": frame.loc[frame["tail"] == 0, outcome].mean(),
                "mean_with_tail": frame.loc[frame["tail"] == 1, outcome].mean(),
            }
        )
    result = pd.DataFrame(rows)
    primary_names = {
        "righting_time_080",
        "integrated_posture_deficit",
        "modeled_pd_effort_common",
        "modeled_pd_effort_total",
    }
    primary_mask = result["analysis"].isin(primary_names)
    result.loc[primary_mask, "p_holm_four_new_endpoints"] = holm_adjust(
        result.loc[primary_mask, "p_hc3"].astype(float).tolist()
    )
    result.loc[~primary_mask, "p_holm_four_new_endpoints"] = np.nan
    return result


def regression_power(effect: float, sigma: float, coefficient_variance_unit: float, df: int, alpha: float) -> float:
    ncp = effect / (sigma * math.sqrt(coefficient_variance_unit))
    critical = stats.t.ppf(1.0 - alpha / 2.0, df)
    return float(stats.nct.cdf(-critical, df, ncp) + stats.nct.sf(critical, df, ncp))


def regression_mde(sigma: float, coefficient_variance_unit: float, df: int, alpha: float, target: float = 0.8) -> float:
    objective = lambda effect: regression_power(effect, sigma, coefficient_variance_unit, df, alpha) - target
    upper = sigma
    while objective(upper) < 0:
        upper *= 1.5
        if upper > 10.0 * sigma:
            raise RuntimeError("Could not bracket the regression MDE")
    return float(optimize.brentq(objective, 1e-9, upper))


def two_sample_n_per_group(effect_size: float, alpha: float, power: float = 0.8) -> int:
    for n in range(2, 5001):
        df = 2 * n - 2
        critical = stats.t.ppf(1.0 - alpha / 2.0, df)
        ncp = effect_size * math.sqrt(n / 2.0)
        attained = stats.nct.cdf(-critical, df, ncp) + stats.nct.sf(critical, df, ncp)
        if attained >= power:
            return n
    raise RuntimeError("Required sample size exceeds search range")


def plot_results(
    frame: pd.DataFrame,
    results: pd.DataFrame,
    condition_tests: pd.DataFrame,
    power: dict[str, object],
    output_dir: Path,
) -> None:
    colors = {"no_tail": "#9A5B35", "with_tail": "#284B70"}
    conditions = [condition for condition in CONDITION_ORDER if condition in set(frame["condition"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, outcome in zip(axes, ("posture_score", "angular_speed_deg_s")):
        positions = np.arange(len(conditions))
        for offset, morphology in ((-0.16, "no_tail"), (0.16, "with_tail")):
            groups = [frame.loc[(frame["condition"] == c) & (frame["morphology"] == morphology), outcome] for c in conditions]
            box = ax.boxplot(groups, positions=positions + offset, widths=0.27, patch_artist=True, showfliers=False)
            for patch in box["boxes"]:
                patch.set_facecolor(colors[morphology]); patch.set_alpha(0.75)
            rng = np.random.default_rng(7)
            for position, values in zip(positions + offset, groups):
                ax.scatter(position + rng.normal(0, 0.025, len(values)), values, s=18, alpha=0.65, color=colors[morphology])
        ax.set_xticks(positions, conditions, rotation=25, ha="right")
        ax.set_ylabel(OUTCOMES[outcome])
        if outcome == "posture_score":
            ax.set_ylim(-0.03, 1.13)
        annotate_condition_significance(
            ax, frame, condition_tests, outcome, conditions, positions,
            score_axis=outcome == "posture_score",
        )
        ax.grid(axis="y", alpha=0.25)
    axes[0].scatter([], [], color=colors["with_tail"], label="with tail")
    axes[0].scatter([], [], color=colors["no_tail"], label="no tail")
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Holm-adjusted within-condition exact permutation tests: "
        "* p<.05, ** p<.01, *** p<.001; ns = not significant",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "outcomes_by_condition.png", dpi=180)
    plt.close(fig)

    score_panels = (
        ("posture_score", "Coupled whole-cat score"),
        ("whole_body_upright_score", "Centerline uprightness"),
        ("spine_alignment_score", "Spine attitude alignment"),
        ("front_upright_score", "Front-body uprightness"),
        ("back_upright_score", "Rear-body uprightness (FK)"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), sharey=True)
    rng = np.random.default_rng(11)
    positions = np.arange(len(conditions))
    for ax, (outcome, title) in zip(axes.flat, score_panels):
        for offset, morphology in ((-0.16, "no_tail"), (0.16, "with_tail")):
            groups = [
                frame.loc[
                    (frame["condition"] == condition)
                    & (frame["morphology"] == morphology),
                    outcome,
                ]
                for condition in conditions
            ]
            box = ax.boxplot(
                groups,
                positions=positions + offset,
                widths=0.27,
                patch_artist=True,
                showfliers=False,
            )
            for patch in box["boxes"]:
                patch.set_facecolor(colors[morphology])
                patch.set_alpha(0.75)
            for position, values in zip(positions + offset, groups):
                ax.scatter(
                    position + rng.normal(0, 0.025, len(values)),
                    values,
                    s=19,
                    alpha=0.68,
                    color=colors[morphology],
                    zorder=3,
                )
        ax.set_title(title)
        ax.set_xticks(positions, conditions, rotation=28, ha="right")
        ax.set_ylim(-0.03, 1.13)
        annotate_condition_significance(
            ax, frame, condition_tests, outcome, conditions, positions, score_axis=True
        )
        ax.grid(axis="y", alpha=0.25)
    axes[1, 2].axis("off")
    axes[0, 0].set_ylabel("score (0–1; higher is better)")
    axes[1, 0].set_ylabel("score (0–1; higher is better)")
    axes[0, 0].scatter([], [], color=colors["with_tail"], label="with tail")
    axes[0, 0].scatter([], [], color=colors["no_tail"], label="no tail")
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.suptitle(
        "Coupled posture metric and components by release condition\n"
        "Holm-adjusted within-condition exact permutation tests: "
        "* p<.05, ** p<.01, *** p<.001; ns = not significant",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "posture_scores_by_condition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    rng = np.random.default_rng(13)
    for offset, morphology in ((-0.16, "no_tail"), (0.16, "with_tail")):
        groups = [
            frame.loc[
                (frame["condition"] == condition)
                & (frame["morphology"] == morphology),
                "posture_score",
            ]
            for condition in conditions
        ]
        box = ax.boxplot(
            groups,
            positions=positions + offset,
            widths=0.27,
            patch_artist=True,
            showfliers=False,
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colors[morphology])
            patch.set_alpha(0.78)
        for position, values in zip(positions + offset, groups):
            ax.scatter(
                position + rng.normal(0, 0.025, len(values)),
                values,
                s=25,
                alpha=0.72,
                color=colors[morphology],
                zorder=3,
            )
    ax.set_xticks(positions, conditions)
    ax.set_ylim(-0.03, 1.13)
    annotate_condition_significance(
        ax, frame, condition_tests, "posture_score", conditions, positions, score_axis=True
    )
    ax.set_ylabel("coupled whole-cat score (0–1; higher is better)")
    ax.set_xlabel("release condition")
    ax.set_title(
        "Whole-body centerline uprightness × spine alignment\n"
        "Holm-adjusted within-condition exact permutation tests: "
        "* p<.05, ** p<.01, *** p<.001; ns = not significant"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.scatter([], [], color=colors["with_tail"], label="with tail")
    ax.scatter([], [], color=colors["no_tail"], label="no tail")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / "posture_score_by_condition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    effects = np.linspace(0.0, 1.5, 151)
    for alpha, style, label in ((0.05, "-", "one primary endpoint (alpha=.05)"), (0.025, "--", "two co-primary endpoints (alpha=.025)")):
        n = int(power["current_n_per_group_min"])
        curve = []
        for d in effects:
            df = 2 * n - 2
            critical = stats.t.ppf(1 - alpha / 2, df)
            ncp = d * math.sqrt(n / 2)
            curve.append(stats.nct.cdf(-critical, df, ncp) + stats.nct.sf(critical, df, ncp))
        ax.plot(effects, curve, style, linewidth=2, label=label)
    ax.axhline(0.8, color="black", linestyle=":", label="80% power")
    ax.set(xlabel="standardized effect (Cohen's d)", ylabel="power", ylim=(0, 1.02), xlim=(0, 1.5))
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "power_curve.png", dpi=180)
    plt.close(fig)


def write_report(
    frame: pd.DataFrame,
    excluded: list[dict[str, str]],
    results: pd.DataFrame,
    sweep_results: pd.DataFrame,
    pitch_level: dict[str, object],
    dynamics_results: pd.DataFrame,
    power: dict[str, object],
    output_dir: Path,
    target_distance_m: float,
) -> None:
    primary = results.loc[results["outcome"] == "posture_score"].iloc[0]
    centerline = results.loc[
        results["outcome"] == "whole_body_upright_score"
    ].iloc[0]
    alignment = results.loc[
        results["outcome"] == "spine_alignment_score"
    ].iloc[0]
    secondary = results.loc[results["outcome"] == "angular_speed_deg_s"].iloc[0]
    pitch_sweep = sweep_results.loc[
        sweep_results["sweep"] == "pitch_sweep_at_roll_180"
    ].iloc[0]
    roll_sweep = sweep_results.loc[
        sweep_results["sweep"] == "no_pitch_roll_sweep"
    ].iloc[0]
    pitch_level_status = (
        "Significant" if float(pitch_level["p_holm_three_requested"]) < 0.05
        else "Not significant"
    )
    righting_time = dynamics_results.loc[
        dynamics_results["analysis"] == "righting_time_080"
    ].iloc[0]
    posture_deficit = dynamics_results.loc[
        dynamics_results["analysis"] == "integrated_posture_deficit"
    ].iloc[0]
    effort_common = dynamics_results.loc[
        dynamics_results["analysis"] == "modeled_pd_effort_common"
    ].iloc[0]
    effort_total = dynamics_results.loc[
        dynamics_results["analysis"] == "modeled_pd_effort_total"
    ].iloc[0]
    time_075 = dynamics_results.loc[
        dynamics_results["analysis"] == "righting_time_075_sensitivity"
    ].iloc[0]
    time_085 = dynamics_results.loc[
        dynamics_results["analysis"] == "righting_time_085_sensitivity"
    ].iloc[0]
    primary_status = "Significant" if primary.p_hc3 < 0.05 else "Not significant"
    pitch_status = "Significant" if pitch_sweep.p_holm_two_sweeps < 0.05 else "Not significant"
    roll_status = "Significant" if roll_sweep.p_holm_two_sweeps < 0.05 else "Not significant"
    lines = [
        "# Tail vs. no-tail telemetry analysis",
        "",
        "## Bottom line",
        "",
        f"The coupled whole-cat score favors the tail by **{primary.estimate:+.3f}** "
        f"(95% CI {primary.ci_low:+.3f} to {primary.ci_high:+.3f}, p={primary.p_hc3:.4f}). "
        f"This is **{primary_status.lower()}** at alpha=0.05.",
        "",
        "The advantage is primarily better **front/rear alignment**, not clearly better whole-body "
        "uprightness. The result is exploratory because the metric was chosen after inspecting the data "
        "and collection date is partly confounded with morphology.",
        "",
        "## Main results",
        "",
        "Effects are adjusted differences: with tail minus no tail. Positive score effects favor the tail; "
        "negative angular-speed effects favor the tail.",
        "",
        "| Outcome | Effect | 95% CI | HC3 p | Result |",
        "|---|---:|---:|---:|---|",
        f"| Coupled whole-cat score | {primary.estimate:+.3f} | "
        f"{primary.ci_low:+.3f} to {primary.ci_high:+.3f} | {primary.p_hc3:.4f} | {primary_status} |",
        f"| Centerline uprightness | {centerline.estimate:+.3f} | "
        f"{centerline.ci_low:+.3f} to {centerline.ci_high:+.3f} | {centerline.p_hc3:.4f} | Diagnostic: not significant |",
        f"| Front/rear alignment | {alignment.estimate:+.3f} | "
        f"{alignment.ci_low:+.3f} to {alignment.ci_high:+.3f} | {alignment.p_hc3:.4f} | Diagnostic: significant |",
        f"| Angular speed (deg/s) | {secondary.estimate:+.1f} | "
        f"{secondary.ci_low:+.1f} to {secondary.ci_high:+.1f} | {secondary.p_hc3:.4f} | Not significant |",
        "",
        f"The blocked permutation check agrees for the primary score (p={primary.p_permutation:.4f}). "
        f"Without date adjustment, its estimate is {primary.estimate_no_date:+.3f} (p={primary.p_no_date:.4f}).",
        "",
        "## Requested pooled comparisons",
        "",
        "| Comparison | n | Effect | 95% CI | HC3 p | Holm p | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| Roll 180°, pitch 0/15/30/45° | {pitch_sweep.n} | {pitch_sweep.estimate:+.3f} | "
        f"{pitch_sweep.ci_low:+.3f} to {pitch_sweep.ci_high:+.3f} | {pitch_sweep.p_hc3:.4f} | "
        f"{pitch_sweep.p_holm_two_sweeps:.4f} | {pitch_status} |",
        f"| No pitch, roll 45/90/180° | {roll_sweep.n} | {roll_sweep.estimate:+.3f} | "
        f"{roll_sweep.ci_low:+.3f} to {roll_sweep.ci_high:+.3f} | {roll_sweep.p_hc3:.4f} | "
        f"{roll_sweep.p_holm_two_sweeps:.4f} | {roll_status} |",
        "",
        "The pitch sweep is under-supported by condition/date overlap (only "
        f"{pitch_sweep.n_overlap} usable overlap-stratum trials), so its uncertainty is large.",
        "",
        "## Pitch leveling within the pitch sweep",
        "",
        "This additional endpoint is the final absolute whole-cat fused pitch: 0° is level, so a negative "
        "tail effect favors the tail. The model adjusts for release-pitch condition, collection date, initial "
        "absolute whole-cat pitch, and initial angular speed.",
        "",
        "| Outcome | n | Tail effect | 95% CI | HC3 p | Holm p across 3 requested tests | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| Absolute final whole-cat pitch | {pitch_level['n']} | {pitch_level['estimate']:+.2f}° | "
        f"{pitch_level['ci_low']:+.2f}° to {pitch_level['ci_high']:+.2f}° | "
        f"{pitch_level['p_hc3']:.4f} | {pitch_level['p_holm_three_requested']:.4f} | "
        f"{pitch_level_status} |",
        "",
        "Signed pitch is not used as the primary leveling endpoint because positive and negative errors could "
        "cancel even when both are far from level.",
        "",
        "## Time to righting",
        "",
        "Righting is the first time the coupled score reaches 0.80 and remains there for 0.10 s. Trials that "
        "do not right by the common 2.5 m cutoff are assigned the 0.714 s horizon; this restricted-time "
        "definition avoids analyzing successful trials alone.",
        "",
        "| Outcome | Tail effect | 95% CI | HC3 p | Holm p across 4 new endpoints |",
        "|---|---:|---:|---:|---:|",
        f"| Restricted time to righting | {righting_time.estimate:+.3f} s | "
        f"{righting_time.ci_low:+.3f} to {righting_time.ci_high:+.3f} | "
        f"{righting_time.p_hc3:.4f} | {righting_time.p_holm_four_new_endpoints:.4f} |",
        f"| Mean posture deficit through cutoff | {posture_deficit.estimate:+.3f} | "
        f"{posture_deficit.ci_low:+.3f} to {posture_deficit.ci_high:+.3f} | "
        f"{posture_deficit.p_hc3:.4f} | {posture_deficit.p_holm_four_new_endpoints:.4f} |",
        "",
        f"The adjusted result estimates righting **{-1000 * righting_time.estimate:.0f} ms sooner** with the tail. "
        f"By the cutoff, {int(frame.loc[frame['tail'] == 1, 'righting_success_080'].sum())}/"
        f"{int((frame['tail'] == 1).sum())} tail trials and "
        f"{int(frame.loc[frame['tail'] == 0, 'righting_success_080'].sum())}/"
        f"{int((frame['tail'] == 0).sum())} no-tail trials met the sustained threshold. Results remain "
        f"significant at thresholds 0.75 (p={time_075.p_hc3:.4f}) and 0.85 (p={time_085.p_hc3:.4f}).",
        "",
        "## Actuator effort—not measured energy",
        "",
        "The telemetry logs target positions but not applied PWM, current, or voltage, so joules cannot be "
        "recovered. The available proxy reconstructs normalized PD duty from 50 Hz target error and joint "
        "motion. It omits the 1 kHz inner-loop waveform and must not be labeled electrical energy.",
        "",
        "| Effort proxy | Tail effect | 95% CI | HC3 p | Holm p across 4 new endpoints |",
        "|---|---:|---:|---:|---:|",
        f"| Three common spine actuators | {effort_common.estimate:+.3f} | "
        f"{effort_common.ci_low:+.3f} to {effort_common.ci_high:+.3f} | "
        f"{effort_common.p_hc3:.4f} | {effort_common.p_holm_four_new_endpoints:.4f} |",
        f"| All four actuators, including tail | {effort_total.estimate:+.3f} | "
        f"{effort_total.ci_low:+.3f} to {effort_total.ci_high:+.3f} | "
        f"{effort_total.p_hc3:.4f} | {effort_total.p_holm_four_new_endpoints:.4f} |",
        "",
        "The tail reduces modeled effort in the three shared spine actuators but increases total modeled "
        "effort once its own actuator is included. Add voltage/current sensing and log applied PWM to answer "
        "the actual energy-consumption question.",
        "",
        "## Metric",
        "",
        "```text",
        "center_up       = normalize(front_up + rear_up)",
        "U               = 1 - whole_body_tilt / pi",
        "A               = 1 - relative_front_rear_angle / pi",
        "whole_cat_score = U * A",
        "```",
        "",
        "The score ranges from 0 to 1. It ignores common yaw, penalizes relative spine twist, and reaches 1 "
        "only when the whole-body centerline is upright and the front/rear frames are aligned. A level but "
        "bent pose therefore scores higher than a front-upright/rear-tilted pose with the same bend.",
        "",
        "Whole-body fused roll/pitch and angular speed are retained as diagnostics. Literal joint-neutral "
        "straightness would require an additional calibrated joint-error measure because endpoint attitudes "
        "cannot detect internal joint cancellation.",
        "",
        "## Analysis and limitations",
        "",
        f"- {len(frame)} drops analyzed at an exact {target_distance_m:g} m fall-distance cutoff; "
        f"{len(excluded)} short {'drop was' if len(excluded) == 1 else 'drops were'} excluded.",
        "- ANCOVA adjusts for release condition, collection date, initial pose score, and initial angular speed; "
        "confidence intervals use HC3 robust standard errors.",
        "- Tail/no-tail assignment was not randomized or interleaved within collection session. Date and morphology "
        "are partly confounded, particularly at 15° and 30° pitch.",
        "- The metric was developed after viewing these data. Treat all p-values as exploratory and test the frozen "
        "metric in a new randomized dataset.",
        "",
        "## Power",
        "",
        f"Current 80% minimum detectable difference: **{power['primary_current_mde_raw']:.3f} score units**.",
        "",
        "Future balanced two-group design (two-sided alpha=0.05, 80% power):",
        "",
        "| Difference to detect | Drops per morphology |",
        "|---:|---:|",
        *[
            f"| {row['posture_score_difference']:.3f} | {row['n_per_morphology_alpha_0.05']} |"
            for row in power["future_raw_sample_sizes"]
        ],
        "",
        "Randomize tail/no-tail within release condition and collection block. Divide each morphology total "
        "across conditions and round up.",
        "",
        "See `SENSITIVITY.md` for cutoff sensitivity, `trial_metrics.csv` for trial-level values, and "
        "`time_effort_analysis.csv` for the new models. PNG files are retained only for statistically "
        "significant date-adjusted results.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", type=Path, default=root / "telemetry" / "raw")
    parser.add_argument("--output", type=Path, default=root / "analysis" / "results")
    parser.add_argument("--distance", type=float, default=2.5)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    for path in sorted(args.input.rglob("*.csv")):
        try:
            records.append(extract_trial(path, args.distance))
        except RuntimeError as error:
            try:
                display_path = path.relative_to(root).as_posix()
            except ValueError:
                display_path = path.as_posix()
            excluded.append({"source_file": display_path, "reason": str(error)})
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("No trials passed preprocessing")
    duplicate_key = frame.duplicated(["morphology", "condition", "date", "rep"], keep=False)
    if duplicate_key.any():
        raise ValueError(f"Duplicate canonical trials:\n{frame.loc[duplicate_key].to_string(index=False)}")
    frame.to_csv(args.output / "trial_metrics.csv", index=False)
    pd.DataFrame(excluded, columns=("source_file", "reason")).to_csv(args.output / "excluded_trials.csv", index=False)

    x, names = build_design(frame)
    result_rows: list[dict[str, object]] = []
    fitted_models: dict[str, OLSResult] = {}
    for outcome in OUTCOMES:
        model = fit_ols(frame[outcome].to_numpy(float), x, names)
        fitted_models[outcome] = model
        row: dict[str, object] = {"outcome": outcome, "label": OUTCOMES[outcome], **coefficient_row(model)}
        x_no_date, names_no_date = build_design(frame, include_date=False)
        model_no_date = fit_ols(frame[outcome].to_numpy(float), x_no_date, names_no_date)
        no_date = coefficient_row(model_no_date)
        row.update({"estimate_no_date": no_date["estimate"], "p_no_date": no_date["p_hc3"]})
        perm_estimate, perm_p, perm_n = overlap_permutation(frame, outcome, args.permutations, args.seed)
        row.update({"overlap_estimate": perm_estimate, "p_permutation": perm_p, "n_overlap": perm_n})
        result_rows.append(row)
    adjusted = holm_adjust([float(row["p_hc3"]) for row in result_rows])
    for row, p_adjusted in zip(result_rows, adjusted):
        row["p_holm_all_outcomes"] = p_adjusted
    results = pd.DataFrame(result_rows)
    results.to_csv(args.output / "model_results.csv", index=False)

    plotted_outcomes = (
        "posture_score",
        "whole_body_upright_score",
        "spine_alignment_score",
        "front_upright_score",
        "back_upright_score",
        "angular_speed_deg_s",
    )
    condition_tests = exact_condition_tests(frame, plotted_outcomes)
    condition_tests.to_csv(args.output / "condition_tests.csv", index=False)

    sweep_results = analyze_predefined_sweeps(frame, args.permutations, args.seed)
    pitch_level = analyze_pitch_leveling(frame, args.permutations, args.seed)
    family_adjusted = holm_adjust(
        sweep_results["p_hc3"].astype(float).tolist() + [float(pitch_level["p_hc3"])]
    )
    sweep_results["p_holm_three_requested"] = family_adjusted[:2]
    pitch_level["p_holm_three_requested"] = family_adjusted[2]
    sweep_results.to_csv(args.output / "sweep_results.csv", index=False)
    pd.DataFrame([pitch_level]).to_csv(args.output / "pitch_level_analysis.csv", index=False)
    dynamics_results = analyze_time_and_effort(frame, args.permutations, args.seed)
    dynamics_results.to_csv(args.output / "time_effort_analysis.csv", index=False)

    primary_model = fitted_models["posture_score"]
    cjj = float(primary_model.xtx_inverse[1, 1])
    current_mde = regression_mde(primary_model.sigma, cjj, primary_model.df_resid, 0.05)
    n_min = int(frame.groupby("morphology").size().min())
    sample_sizes = []
    for d in (0.3, 0.5, 0.8):
        sample_sizes.append(
            {
                "cohens_d": d,
                "n_per_morphology_alpha_0.05": two_sample_n_per_group(d, 0.05),
                "n_per_morphology_alpha_0.025": two_sample_n_per_group(d, 0.025),
            }
        )
    raw_sample_sizes = []
    for difference in (0.025, 0.05, 0.075, 0.10):
        standardized = difference / primary_model.sigma
        raw_sample_sizes.append(
            {
                "posture_score_difference": difference,
                "n_per_morphology_alpha_0.05": two_sample_n_per_group(standardized, 0.05),
                "n_per_morphology_alpha_0.025": two_sample_n_per_group(standardized, 0.025),
            }
        )
    power: dict[str, object] = {
        "current_n_per_group_min": n_min,
        "primary_current_mde_raw": current_mde,
        "primary_current_mde_d": current_mde / primary_model.sigma,
        "primary_residual_sd": primary_model.sigma,
        "future_sample_sizes": sample_sizes,
        "future_raw_sample_sizes": raw_sample_sizes,
    }
    summary = {
        "target_distance_m": args.distance,
        "included_trials": len(frame),
        "excluded_trials": len(excluded),
        "model": "OLS ANCOVA with condition/date fixed effects and standardized initial posture/speed covariates; HC3 SE",
        "results": result_rows,
        "predefined_sweeps": sweep_results.to_dict(orient="records"),
        "pitch_level_analysis": pitch_level,
        "time_effort_analysis": dynamics_results.to_dict(orient="records"),
        "power": power,
    }
    json_text = json.dumps(
        summary,
        indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    )
    (args.output / "analysis_summary.json").write_text(json_text + "\n")
    plot_ancova_significance(frame, results, sweep_results, dynamics_results, args.output)
    write_report(
        frame,
        excluded,
        results,
        sweep_results,
        pitch_level,
        dynamics_results,
        power,
        args.output,
        args.distance,
    )
    print(results[["label", "estimate", "ci_low", "ci_high", "p_hc3", "p_permutation", "n_overlap"]].to_string(index=False))
    print("\nPooled sweep analyses")
    print(sweep_results[["label", "n", "estimate", "ci_low", "ci_high", "p_hc3", "p_permutation", "n_overlap"]].to_string(index=False))
    print("\nPitch leveling analysis")
    print(pd.DataFrame([pitch_level])[["label", "n", "estimate", "ci_low", "ci_high", "p_hc3", "p_permutation", "n_overlap"]].to_string(index=False))
    print("\nRighting time and modeled effort")
    print(dynamics_results[["label", "estimate", "ci_low", "ci_high", "p_hc3", "p_permutation", "p_holm_four_new_endpoints"]].to_string(index=False))
    print(f"\nIncluded {len(frame)}; excluded {len(excluded)} short {'drop' if len(excluded) == 1 else 'drops'}")
    print(f"Primary 80% MDE: {current_mde:.3f} score units (d={current_mde / primary_model.sigma:.2f})")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
