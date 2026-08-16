import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import time
import os

def load_telemetry_csv(filepath):
    columns = [
        "Time", "F_Q0", "F_Q1", "F_Q2", "F_Q3", "F_M1", "F_M2", "F_ACC",  "Cmd_F1", "Cmd_F2", 
        "B_Q0", "B_Q1", "B_Q2", "B_Q3", "B_M1", "B_M2", "B_ACC",  "Cmd_B1", "Cmd_B2"
    ]
    
    df = pd.read_csv(filepath, usecols=columns)
    
    front_quats = df[["F_Q0", "F_Q1", "F_Q2", "F_Q3"]].to_numpy()
    rear_quats = df[["B_Q0", "B_Q1", "B_Q2", "B_Q3"]].to_numpy()
    rot1 = df["F_M1"].to_numpy()
    pitch = df["F_M2"].to_numpy()
    acc1 = df["F_ACC"].to_numpy()
    acc2 = df["B_ACC"].to_numpy()
    rot2 = df["B_M2"].to_numpy()
    tail = df["B_M1"].to_numpy()
    time = df["Time"].to_numpy()
    
    return {
        "time": time,
        "front_quat": front_quats,
        "rear_quat": rear_quats,
        "rot1": rot1,
        "pitch": pitch,
        "rot2": rot2,
        "tail": tail,
        "acc1": acc1,
        "acc2": acc2
    }

def visualize():
    model_path = os.path.abspath("model/cat_viz.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Rear body ghost viz
    ghost_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rear_ghost")
    ghost_mocap_id = model.body_mocapid[ghost_body_id]
    real_rear_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rear_body")
    
    # Disable gravity
    model.opt.gravity[:] = [0, 0, 0]

    joint_names = ["rot1", "pitch", "rot2", "tail"]
    joint_qpos_idx = {
        name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] 
        for name in joint_names
    }

    # Load telemetry
    telemetry = load_telemetry_csv("/Users/itak/Documents/CMU/24775_Bioinspired_Robot/MeowMentum/telemetry/telemetry_1776712613.csv")
    num_steps = len(telemetry["front_quat"])
    fps = 100
    
    print("Launching viewer.")
    
    # Launch the passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 3]
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 90
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        
        for step in range(num_steps):
            if not viewer.is_running():
                break
            
            # Root position
            data.qpos[0:3] = [0, 0, 3] 
            
            # Root orientation: quat - (w, x, y, z)
            data.qpos[3:7] = telemetry["front_quat"][step]
            
            # Joint angles
            data.qpos[joint_qpos_idx["rot1"]] = telemetry["rot1"][step]
            data.qpos[joint_qpos_idx["pitch"]] = telemetry["pitch"][step]
            data.qpos[joint_qpos_idx["rot2"]] = telemetry["rot2"][step]
            data.qpos[joint_qpos_idx["tail"]] = telemetry["tail"][step]
            
            # Update kinematics (no physics)
            mujoco.mj_kinematics(model, data)

            # Snap the ghost's position to the simulated rear body's position
            data.mocap_pos[ghost_mocap_id] = data.xpos[real_rear_body_id]
            # Set the ghost's orientation purely from the rear IMU telemetry
            data.mocap_quat[ghost_mocap_id] = telemetry["rear_quat"][step]
            
            viewer.sync()
            time.sleep(1.0 / fps)

if __name__ == "__main__":
    visualize()