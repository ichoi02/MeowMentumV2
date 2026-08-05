"""
End-to-end hardware-pipeline test (no physical hardware required).

Drives the MuJoCo Cat-v0 sim as the "physical robot" and runs the ACTUAL
hardware inference pipeline from controller.py (projected-gravity obs, 4-frame
stacking, ONNX inference, action low-pass filter, joint-range mapping). Verifies
each stage matches the simulation/training pipeline, then confirms closed-loop
righting.

The sim's front_body world-frame quaternion stands in for the front IMU (i.e. we
assume the on-robot IMU alignment yields a z-up world frame); joint encoders are
read from qpos. Everything downstream of that is the real controller code.

Run from the repo root:  python hardware/e2e_test.py
"""
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "hardware"))

import gymnasium as gym
import cat_env  # noqa: F401  (registers Cat-v0 / CatNoTail-v0)
import distillation as D
import controller as C  # the real hardware controller (constants + stack_frames + util)
from variants import VARIANTS, student_path, onnx_path

# Floor for the closed-loop check. This is a pipeline-breakage tripwire, not a
# performance target. N=100 with random attitudes is noisy -- observed no-tail runs
# have ranged 12-25/100 -- so the floors sit well under the low end. They catch a
# collapse to passivity (which reads ~0-2%, see docs/EVALUATION.md), not a few
# points of drift.
RIGHTING_FLOOR = {"tail": 10, "notail": 5}

results = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    results.append(ok)


# ---- shared helpers: emulate the on-robot sensors + controller frame ----
def emulate_sensors(u):
    """Front IMU quaternion (world frame) + 4 joint encoder angles, from sim state."""
    front_quat = u.data.xquat[u._body_idx["front_body"]].copy()
    qi = u._joint_qpos_idx
    joints = np.array([u.data.qpos[qi["rot1"]], u.data.qpos[qi["pitch"]],
                       u.data.qpos[qi["rot2"]], u.data.qpos[qi["tail"]]])
    return front_quat, joints

def controller_frame(front_quat, joints):
    """One student frame exactly as controller.py builds it."""
    grav = C.util.to_projected_gravity(front_quat)
    return np.concatenate([grav, joints])

def sim_student_frame(obs):
    """Ground-truth student frame from the sim obs (front_proj_grav[0:3] + joints[12:16])."""
    return np.concatenate([obs[0:3], obs[12:16]])

def tilt_deg(u, body):
    up = R.from_quat(u.data.xquat[u._body_idx[body]], scalar_first=True).apply([0, 0, 1])
    return np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))


def main(variant="tail"):
    cfg = VARIANTS[variant]
    ONNX_FILE = onnx_path(variant)
    PTH_FILE = student_path(variant)

    print(f"variant={variant}  env={cfg['env_id']}  onnx={os.path.basename(ONNX_FILE)}")
    print(f"N_FRAMES={C.N_FRAMES}  FILTER_ALPHA={C.FILTER_ALPHA}  "
          f"ranges(roll/pitch/tail)={C.ROLL_RANGE}/{C.PITCH_RANGE}/{C.TAIL_RANGE}")

    sess = ort.InferenceSession(ONNX_FILE)
    inp = sess.get_inputs()[0].name
    onnx_dim = sess.get_inputs()[0].shape[-1]

    student = D.StudentPolicy(D.STUDENT_OBS_DIM, 3)
    student.load_state_dict(torch.load(PTH_FILE, map_location="cpu"))
    student.eval()

    # ---- TEST 0: dimensions line up across the whole chain ----
    check(f"dims consistent (controller/distill/onnx = {D.STUDENT_OBS_DIM})",
          C.N_FRAMES * D.FRAME_DIM == D.STUDENT_OBS_DIM == onnx_dim,
          f"ctrl={C.N_FRAMES*D.FRAME_DIM} distill={D.STUDENT_OBS_DIM} onnx={onnx_dim}")

    # ---- TEST 1: ONNX == PyTorch student ----
    X = np.random.randn(1000, D.STUDENT_OBS_DIM).astype(np.float32)
    with torch.no_grad():
        yt = student(torch.from_numpy(X)).numpy()
    yo = sess.run(None, {inp: X})[0]
    d1 = float(np.max(np.abs(yt - yo)))
    check("ONNX == PyTorch student", d1 < 1e-4, f"max|diff|={d1:.2e}")

    env = gym.make(cfg["env_id"])
    u = env.unwrapped

    # ---- TEST 2: controller obs-build == sim student obs (over a rollout) ----
    obs, _ = env.reset()
    ctrl_hist = deque(maxlen=C.N_FRAMES)
    sim_hist = deque(maxlen=D.N_FRAMES)
    maxdiff2 = 0.0
    for _ in range(300):
        fq, jt = emulate_sensors(u)
        cf = controller_frame(fq, jt)
        sf = sim_student_frame(obs)
        for hist, frame in ((ctrl_hist, cf), (sim_hist, sf)):
            if len(hist) == 0:
                for _ in range(hist.maxlen):
                    hist.append(frame)
            else:
                hist.append(frame)
        ctrl_obs = C.stack_frames(list(ctrl_hist))
        sim_obs = D.stack_frames(list(sim_hist))
        maxdiff2 = max(maxdiff2, float(np.max(np.abs(ctrl_obs - sim_obs))))
        raw = sess.run(None, {inp: ctrl_obs.astype(np.float32).reshape(1, -1)})[0][0]
        obs, _, term, trunc, _ = env.step(raw)
        if term or trunc:
            obs, _ = env.reset()
            ctrl_hist.clear()
            sim_hist.clear()
    check("controller obs == sim student obs", maxdiff2 < 1e-6, f"max|diff|={maxdiff2:.2e}")

    # ---- TEST 3: controller filter+map == sim internal executed target ----
    # Disable the sim's domain-randomization action delay so the comparison is 1:1.
    obs, _ = env.reset()
    u.action_delay = 0
    u.action_buffer = []
    ctrl_filt = np.zeros(3, dtype=np.float32)
    rng = np.random.default_rng(0)
    maxdiff3 = 0.0
    n_cmp = 0
    for _ in range(300):
        raw = rng.uniform(-1, 1, 3).astype(np.float32)
        ctrl_filt = C.FILTER_ALPHA * raw + (1.0 - C.FILTER_ALPHA) * ctrl_filt
        c_targets = np.array([
            C.util.map_value(float(ctrl_filt[0]), -1, 1, -C.ROLL_RANGE, C.ROLL_RANGE),
            C.util.map_value(float(ctrl_filt[1]), -1, 1, -C.PITCH_RANGE, C.PITCH_RANGE),
            C.util.map_value(float(ctrl_filt[2]), -1, 1, -C.TAIL_RANGE, C.TAIL_RANGE),
        ])
        obs, _, term, trunc, _ = env.step(raw)
        if not (term or trunc):
            sim_exec = np.asarray(u.ctrls[-1][:3])  # sim's mapped [rot1,pitch,tail] target
            maxdiff3 = max(maxdiff3, float(np.max(np.abs(c_targets - sim_exec))))
            n_cmp += 1
        else:
            obs, _ = env.reset()
            u.action_delay = 0
            u.action_buffer = []
            ctrl_filt = np.zeros(3, dtype=np.float32)
    check("controller filter+map == sim executed target", maxdiff3 < 1e-6,
          f"max|diff|={maxdiff3:.2e} over {n_cmp} steps")

    # ---- TEST 4: closed-loop righting through the full hardware pipeline ----
    # Build obs from emulated sensors -> ONNX -> feed to env (sim filter+map+PD == controller+Teensy).
    N = 100
    ok = 0
    ftilt, rtilt = [], []
    infer_times = []
    for _ in range(N):
        obs, _ = env.reset()
        ctrl_hist = deque(maxlen=C.N_FRAMES)
        done = False
        while not done:
            fq, jt = emulate_sensors(u)
            cf = controller_frame(fq, jt)
            if len(ctrl_hist) == 0:
                for _ in range(C.N_FRAMES):
                    ctrl_hist.append(cf)
            else:
                ctrl_hist.append(cf)
            x = C.stack_frames(list(ctrl_hist)).astype(np.float32).reshape(1, -1)
            t0 = time.perf_counter()
            raw = sess.run(None, {inp: x})[0][0]
            infer_times.append((time.perf_counter() - t0) * 1e3)
            obs, _, term, trunc, _ = env.step(raw)
            done = term or trunc
        fa, ra = tilt_deg(u, "front_body"), tilt_deg(u, "rear_body")
        ftilt.append(fa); rtilt.append(ra)
        ok += (fa < 30 and ra < 30)
    env.close()
    pct = ok
    floor = RIGHTING_FLOOR[variant]
    check(f"closed-loop righting >= {floor}% ({variant} tripwire)", pct >= floor,
          f"{pct}/100 both-upright  |  final tilt med f/r "
          f"{np.median(ftilt):.0f}/{np.median(rtilt):.0f} deg")
    print(f"       ONNX inference: mean {np.mean(infer_times):.3f} ms, "
          f"p99 {np.percentile(infer_times,99):.3f} ms  (budget @50Hz = 20 ms)")

    print("\n=== SUMMARY ===")
    print(f"{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="tail",
                        help="ablation condition to test (default: tail)")
    args = parser.parse_args()
    main(args.variant)
