"""
Reconstruct robot kinematics from Teensy telemetry and SAVE to a file.  [Raspberry Pi]

Pipeline (mirrors hardware/controller.py exactly):
  1. Read Teensy telemetry ("qr,qi,qj,qk,ang1,ang2,acc") from BOTH boards.
  2. Build the single student-observation frame [front_proj_grav(3) + joints(4)]
     (front IMU only, joints [rot1, pitch, rot2, tail]) -- exactly what the policy sees.
  3. Reconstruct the pose from that frame ALONE: recover the front-body orientation
     from projected gravity (yaw is unobservable -> chosen minimally) + joint angles.
  4. Save the reconstructed kinematics (root quat + joints + rear cross-check) to .npz.

Replay the saved file on a desktop with hardware/reconstruct_viz_view.py (MuJoCo).

This half is MuJoCo-free (numpy + scipy only; +pyserial for --source serial): the
rear kinematics are computed analytically from the cat.xml joint chain, which matches
mj_kinematics to ~1e-4 deg. Rendering (the heavy part) happens on the desktop.

Cross-check: rear pose implied by (front IMU + joints) vs the independent rear IMU
(unused by the policy). ~0 => consistent; a few degrees on hardware => model /
calibration / mounting mismatch to fix before deployment.

No hardware needed to test: synthetic sources (sim, sweep) fabricate telemetry.
On the robot: --source serial reads the two Teensys live.

Usage:
  python hardware/reconstruct_viz.py --source serial --out recon.npz --duration 10
  python hardware/reconstruct_viz.py --source sweep  --out recon.npz --frames 400   # offline test
  python hardware/reconstruct_viz.py --selftest --source sweep                      # validate pipeline
"""
import os
import sys
import time
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import cat_env.env_util as util  # noqa: E402  (to_projected_gravity)

WORLD_DOWN = np.array([0.0, 0.0, -1.0])
ROOT_HEIGHT = 1.5  # fixed root height for the reconstructed pose (viewer convenience)

# Placeholder accel magnitudes for the telemetry field (unused by obs/reconstruction).
FRONT_ACC, BACK_ACC = 0.5, 9.8


# ----------------------------------------------------------------------------
# IMU frame alignment.  MUST match hardware/controller.py::align_imu_quaternions.
# (NB: cat_env.env_util.reverse_align_imu_quaternions is NOT the true inverse of
#  this -- it round-trips with ~0.77 error -- so we define the correct inverse here.)
# ----------------------------------------------------------------------------
def _r_align(imu_type):
    if imu_type == "Front":
        return R.from_euler("xyz", [0, 0, 90], degrees=True)
    return R.from_euler("xyz", [180, 0, -90], degrees=True)  # Back

def align_imu(quat_wxyz, imu_type):
    """Raw IMU quaternion -> world frame (as controller.py does on the robot)."""
    a = _r_align(imu_type)
    r = R.from_quat(quat_wxyz, scalar_first=True)
    return (a.inv() * r * a).as_quat(scalar_first=True)

def unalign_imu(quat_wxyz, imu_type):
    """World frame -> raw IMU quaternion (true inverse of align_imu; synthetic gen only)."""
    a = _r_align(imu_type)
    r = R.from_quat(quat_wxyz, scalar_first=True)
    return (a * r * a.inv()).as_quat(scalar_first=True)


# ----------------------------------------------------------------------------
# Teensy telemetry <-> values
# ----------------------------------------------------------------------------
_TELEM_FMT = "{:.6f},{:.6f},{:.6f},{:.6f},{:.4f},{:.4f},{:.4f}"

def format_telem(quat_wxyz, ang1, ang2, acc):
    return _TELEM_FMT.format(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3],
                             ang1, ang2, acc)

def parse_telem(line):
    """Parse one board's telemetry line -> dict, or None if malformed."""
    parts = line.strip().split(",")
    if len(parts) != 7:
        return None
    try:
        v = [float(x) for x in parts]
    except ValueError:
        return None
    return {"quat": np.array(v[:4]), "m1": v[4], "m2": v[5], "acc": v[6]}


# ----------------------------------------------------------------------------
# Telemetry -> student observation frame  (mirrors controller.py)
# ----------------------------------------------------------------------------
def build_student_frame(front, back):
    """[front_proj_grav(3), joints(4)]  with joints = [rot1, pitch, rot2, tail]."""
    front_world = align_imu(front["quat"], "Front")
    grav = util.to_projected_gravity(front_world)
    joints = np.array([front["m1"], front["m2"], back["m2"], back["m1"]])
    return np.concatenate([grav, joints])


# ----------------------------------------------------------------------------
# Student observation -> reconstructed kinematics (MuJoCo-free)
# ----------------------------------------------------------------------------
def grav_to_root_quat(grav):
    """Front-body orientation (body->world) whose projected gravity equals `grav`.

    Solves R @ grav = world_down; yaw about gravity is a free DOF that
    align_vectors resolves to the minimal rotation (heading is unobservable)."""
    rot, _ = R.align_vectors([WORLD_DOWN], [grav])
    return rot.as_quat(scalar_first=True)

def reconstruct(frame):
    """student frame -> (root_quat wxyz, joints[rot1,pitch,rot2,tail])."""
    grav, joints = frame[:3], np.asarray(frame[3:], dtype=float)
    return grav_to_root_quat(grav), joints

def rear_world_quat(root_quat, joints):
    """Rear-body world orientation from the cat.xml chain:
    front -> Rx(rot1) -> Ry(pitch) -> Rx(rot2) -> rear.  (Matches mj_kinematics.)"""
    rot1, pitch, rot2, _tail = joints
    Rf = R.from_quat(root_quat, scalar_first=True)
    Rrear = Rf * R.from_euler("x", rot1) * R.from_euler("y", pitch) * R.from_euler("x", rot2)
    return Rrear.as_quat(scalar_first=True)

def _angle_deg(u, v):
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))

def rear_crosscheck_deg(root_quat, joints, back):
    """Angle between reconstructed rear gravity (front IMU + joints) and the
    measured rear gravity (independent rear IMU). ~0 => consistent."""
    rear_recon = util.to_projected_gravity(rear_world_quat(root_quat, joints))
    rear_meas = util.to_projected_gravity(align_imu(back["quat"], "Back"))
    return _angle_deg(rear_recon, rear_meas)


# ----------------------------------------------------------------------------
# Telemetry sources.  Each is iterable, yielding (front_line, back_line).
# sim/sweep are synthetic (import MuJoCo lazily, dev-machine only); serial is real.
# ----------------------------------------------------------------------------
def _state_to_telem(front_world, rear_world, rot1, pitch, rot2, tail):
    """(world quats + joint angles) -> the two Teensy lines a real robot would send.
    Front board sends (m1=rot1, m2=pitch); back board sends (m1=tail, m2=rot2)."""
    fl = format_telem(unalign_imu(front_world, "Front"), rot1, pitch, FRONT_ACC)
    bl = format_telem(unalign_imu(rear_world, "Back"), tail, rot2, BACK_ACC)
    return fl, bl

def sim_source(dt_holder):
    """Teacher-driven righting episodes in the real sim; emit telemetry per step.
    The rear IMU comes from independent physics, so the cross-check is meaningful."""
    import mujoco
    import gymnasium as gym
    import cat_env  # noqa: F401
    from stable_baselines3 import SAC

    env = gym.make("Cat-v0")
    u = env.unwrapped
    dt_holder[0] = u.dt
    try:
        from variants import teacher_path
        policy = SAC.load(teacher_path("tail"))
        predict = lambda o: policy.predict(o, deterministic=True)[0]
        print("sim source: driving with teacher policy (cat_controller)")
    except Exception as e:
        predict = lambda o: env.action_space.sample()
        print(f"sim source: no teacher ({e.__class__.__name__}); using random actions")

    qi = u._joint_qpos_idx
    fb, rb = u._body_idx["front_body"], u._body_idx["rear_body"]
    while True:
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(predict(obs))
            done = term or trunc
            # After mj_step, xquat lags the integrated qpos by one kinematics eval;
            # resync so the synthetic IMU + encoder snapshot is mutually consistent.
            mujoco.mj_kinematics(u.model, u.data)
            yield _state_to_telem(
                u.data.xquat[fb].copy(), u.data.xquat[rb].copy(),
                u.data.qpos[qi["rot1"]], u.data.qpos[qi["pitch"]],
                u.data.qpos[qi["rot2"]], u.data.qpos[qi["tail"]],
            )

def sweep_source(dt_holder):
    """Prescribed DOF-by-DOF motion: tilt the body through roll then pitch, then
    sweep each joint one at a time. Rear pose from a helper model (kinematically consistent)."""
    import mujoco
    dt_holder[0] = 0.02
    hm = mujoco.MjModel.from_xml_path(os.path.join(REPO, "model", "cat.xml"))
    hd = mujoco.MjData(hm)
    jadr = {n: hm.jnt_qposadr[mujoco.mj_name2id(hm, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ("rot1", "pitch", "rot2", "tail")}
    fb = mujoco.mj_name2id(hm, mujoco.mjtObj.mjOBJ_BODY, "front_body")
    rb = mujoco.mj_name2id(hm, mujoco.mjtObj.mjOBJ_BODY, "rear_body")

    def forward(front_quat, r1, pt, r2, tl):
        hd.qpos[0:3] = [0, 0, ROOT_HEIGHT]
        hd.qpos[3:7] = front_quat
        hd.qpos[jadr["rot1"]], hd.qpos[jadr["pitch"]] = r1, pt
        hd.qpos[jadr["rot2"]], hd.qpos[jadr["tail"]] = r2, tl
        mujoco.mj_kinematics(hm, hd)
        return hd.xquat[fb].copy(), hd.xquat[rb].copy()

    def tri(lo, hi, n):  # lo -> hi -> lo
        return np.concatenate([np.linspace(lo, hi, n), np.linspace(hi, lo, n)])

    while True:
        for axis in ("x", "y"):                      # body roll, then pitch (joints zero)
            for a in tri(-np.pi / 2, np.pi / 2, 60):
                fq = R.from_euler(axis, a).as_quat(scalar_first=True)
                yield _state_to_telem(*forward(fq, 0, 0, 0, 0), 0, 0, 0, 0)
        ident = np.array([1.0, 0, 0, 0])             # sweep each joint, body upright
        ranges = {"rot1": 1.2, "pitch": 1.4, "rot2": 1.2, "tail": 1.7}
        for j, amp in ranges.items():
            for a in tri(-amp, amp, 60):
                r1 = a if j == "rot1" else 0.0
                pt = a if j == "pitch" else 0.0
                r2 = a if j == "rot2" else 0.0
                tl = a if j == "tail" else 0.0
                yield _state_to_telem(*forward(ident, r1, pt, r2, tl), r1, pt, r2, tl)


_MAX_DRAIN_LINES = 32  # cap lines processed per drain (matches controller.py)

class SerialSource:
    """Live telemetry from the two Teensys over USB serial (real hardware).

    Reuses hardware/controller.py board discovery (get_port_by_sn, SN_FRONT/BACK).
    Each Teensy prints the *raw* IMU quaternion + 2 encoder angles + acc exactly
    like controller.py reads (7 fields, see PD_control_*.ino), so this yields the
    raw (front_line, back_line) and the shared parse -> align -> build_student_frame
    path reproduces controller.py's pipeline verbatim.

    Iterator: __next__ blocks until BOTH boards have produced a fresh line.
    `_front_ser`/`_back_ser` inject serial-like objects for offline testing.
    """
    def __init__(self, dt_holder, sn_front=None, sn_back=None, baud=115200,
                 _front_ser=None, _back_ser=None):
        import serial  # local import: synthetic modes need no pyserial
        self._serial = serial
        dt_holder[0] = 0.0  # the blocking read paces to the live ~50 Hz stream

        if _front_ser is not None and _back_ser is not None:  # test injection
            self.front, self.back = _front_ser, _back_ser
        else:
            from controller import get_port_by_sn, SN_FRONT, SN_BACK
            pf = get_port_by_sn(sn_front or SN_FRONT)
            pb = get_port_by_sn(sn_back or SN_BACK)
            if not pf or not pb:
                raise RuntimeError(
                    f"Teensy not found (front={pf}, back={pb}). Check USB / serial numbers.")
            self.front = serial.Serial(pf, baud, timeout=0.005)
            self.back = serial.Serial(pb, baud, timeout=0.005)
            print(f"SerialSource: front={pf}  back={pb}")
            time.sleep(0.2)  # let the streams warm up

        self._rem = {"front": b"", "back": b""}
        self._last = {"front": None, "back": None}
        self._prev = (None, None)

    def _latest_line(self, ser, key):
        """Drain waiting bytes; keep the newest complete 7-field line (persisted)."""
        try:
            waiting = ser.in_waiting
        except (OSError, self._serial.SerialException):
            return self._last[key]
        if waiting > 0:
            try:
                self._rem[key] += ser.read(min(waiting, 4096))
            except (OSError, self._serial.SerialException):
                return self._last[key]
            if b"\n" in self._rem[key]:
                chunks = self._rem[key].split(b"\n")
                self._rem[key] = chunks[-1]
                for c in chunks[:-1][-_MAX_DRAIN_LINES:]:
                    s = c.decode("utf-8", errors="ignore").strip()
                    if s.count(",") == 6:  # 7 fields
                        self._last[key] = s
            elif len(self._rem[key]) > 8192:
                self._rem[key] = self._rem[key][-4096:]
        return self._last[key]

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            fl = self._latest_line(self.front, "front")
            bl = self._latest_line(self.back, "back")
            if fl is not None and bl is not None and (fl, bl) != self._prev:
                self._prev = (fl, bl)
                return fl, bl
            time.sleep(0.001)


def _make_source(name, dt_holder):
    return {"sim": sim_source, "sweep": sweep_source, "serial": SerialSource}[name](dt_holder)


# ----------------------------------------------------------------------------
def run_selftest(source_name, n=400):
    """Headless: run the full pipeline and report the rear cross-check error stats.
    With synthetic data the reconstruction is exact, so error must be ~0."""
    dt_holder = [0.02]
    src = iter(_make_source(source_name, dt_holder))

    errs, grav_norm_err, parsed_ok = [], [], 0
    for _ in range(n):
        fl, bl = next(src)
        front, back = parse_telem(fl), parse_telem(bl)
        assert front is not None and back is not None, "telemetry parse failed"
        parsed_ok += 1
        frame = build_student_frame(front, back)
        grav_norm_err.append(abs(np.linalg.norm(frame[:3]) - 1.0))
        root_quat, joints = reconstruct(frame)
        errs.append(rear_crosscheck_deg(root_quat, joints, back))

    errs = np.array(errs)
    print(f"\n=== SELF-TEST ({source_name}, {n} frames) ===")
    print(f"telemetry parsed          : {parsed_ok}/{n}")
    print(f"front gravity |‖g‖-1| max : {max(grav_norm_err):.2e}")
    print(f"rear cross-check error deg: mean {errs.mean():.4f}, p95 {np.percentile(errs,95):.4f}, max {errs.max():.4f}")
    ok = parsed_ok == n and max(grav_norm_err) < 1e-6 and errs.max() < 0.5
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def record(source_name, out_path, duration=None, frames=None):
    """Stream telemetry, reconstruct each frame, and save the kinematics to .npz.

    Saved arrays (one row per frame):
      t            (T,)   wall-clock seconds since start
      root_quat    (T,4)  reconstructed front-body orientation (wxyz)
      joints       (T,4)  [rot1, pitch, rot2, tail] (rad)
      rear_err_deg (T,)   rear-IMU cross-check (deg)
    Stops on --duration, --frames, source exhaustion, or Ctrl+C (saves either way).
    """
    dt_holder = [0.02]
    src = iter(_make_source(source_name, dt_holder))

    ts, roots, jnts, errs = [], [], [], []
    t0 = time.time()
    print(f"Recording (source={source_name}) -> {out_path}. Ctrl+C to stop.")
    try:
        for fl, bl in src:
            front, back = parse_telem(fl), parse_telem(bl)
            if front is None or back is None:
                continue
            frame = build_student_frame(front, back)
            root_quat, joints = reconstruct(frame)
            t = time.time() - t0
            ts.append(t)
            roots.append(root_quat)
            jnts.append(joints)
            errs.append(rear_crosscheck_deg(root_quat, joints, back))
            if (duration is not None and t >= duration) or \
               (frames is not None and len(ts) >= frames):
                break
    except KeyboardInterrupt:
        print("\nstopped (Ctrl+C).")

    if not ts:
        print("No frames captured; nothing saved.")
        return 1

    errs = np.asarray(errs)
    np.savez(
        out_path,
        t=np.asarray(ts, dtype=np.float32),
        root_quat=np.asarray(roots, dtype=np.float32),
        joints=np.asarray(jnts, dtype=np.float32),
        rear_err_deg=errs.astype(np.float32),
        source=source_name,
        root_height=np.float32(ROOT_HEIGHT),
    )
    print(f"Saved {len(ts)} frames to {out_path}")
    print(f"rear cross-check deg: mean {errs.mean():.3f}, p95 {np.percentile(errs,95):.3f}, "
          f"max {errs.max():.3f}   (near 0 = good; several deg = calibration/model mismatch)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=["sim", "sweep", "serial"], default="serial",
                    help="telemetry source: sim/sweep are synthetic; serial reads the real Teensys")
    ap.add_argument("--out", default="reconstruction.npz", help="output .npz path")
    ap.add_argument("--duration", type=float, default=None, help="record this many seconds, then stop")
    ap.add_argument("--frames", type=int, default=None, help="record this many frames, then stop")
    ap.add_argument("--selftest", action="store_true", help="headless pipeline validation (no recording)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(run_selftest(args.source))
    sys.exit(record(args.source, args.out, args.duration, args.frames))


if __name__ == "__main__":
    main()
