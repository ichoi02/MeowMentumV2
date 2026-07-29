"""
Replay reconstructed kinematics (.npz from reconstruct_viz.py) in a MuJoCo viewer. [desktop]

reconstruct_viz.py runs on the Raspberry Pi (headless, MuJoCo-free) and saves the
reconstructed pose per frame. Copy that .npz to a machine with a display and run:

  python hardware/reconstruct_viz_view.py recon.npz
  python hardware/reconstruct_viz_view.py recon.npz --fps 30 --loop

Each frame sets the root orientation (recovered from the front IMU's projected
gravity) + joint angles and runs forward kinematics. There is no rear IMU; check
visually that the pose matches the physical robot. Heading (yaw) is unobservable,
so the robot may sit at an arbitrary yaw -- expected, and exactly why the policy
is yaw-invariant.
"""
import os
import sys
import time
import argparse

import numpy as np
import mujoco
import mujoco.viewer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO, "model", "cat.xml")
JOINTS = ("rot1", "pitch", "rot2", "tail")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file", help="reconstruction .npz saved by reconstruct_viz.py")
    ap.add_argument("--fps", type=float, default=50.0, help="playback frame rate")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--loop", action="store_true", help="loop playback")
    args = ap.parse_args()

    d = np.load(args.file, allow_pickle=True)
    root_quat = d["root_quat"]
    joints = d["joints"]
    root_height = float(d["root_height"]) if "root_height" in d else 1.5
    n = len(root_quat)
    src = str(d["source"]) if "source" in d else "?"
    print(f"Loaded {n} frames (source={src}) from {args.file}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0  # kinematic replay only; no dynamics
    jadr = {j: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in JOINTS}

    frame_dt = 1.0 / max(args.fps, 1e-3) / max(args.speed, 1e-3)
    print("Launching viewer. Ctrl+C to quit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Robot is pinned at a fixed root height and only rotates, so a free camera
        # aimed there keeps it centered and in view.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, root_height]
        viewer.cam.distance = 2.2
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

        try:
            while viewer.is_running():
                for i in range(n):
                    if not viewer.is_running():
                        break
                    t0 = time.time()
                    data.qpos[0:3] = [0.0, 0.0, root_height]
                    data.qpos[3:7] = root_quat[i]
                    data.qpos[jadr["rot1"]] = joints[i, 0]
                    data.qpos[jadr["pitch"]] = joints[i, 1]
                    data.qpos[jadr["rot2"]] = joints[i, 2]
                    data.qpos[jadr["tail"]] = joints[i, 3]
                    mujoco.mj_kinematics(model, data)
                    print(f"\rframe {i+1}/{n}   ", end="", flush=True)
                    viewer.sync()
                    sleep = frame_dt - (time.time() - t0)
                    if sleep > 0:
                        time.sleep(sleep)
                if not args.loop:
                    break
            print("\nDone.")
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
