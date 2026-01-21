"""
Utility functions for extracting LocoMuJoCo demonstration data.
"""

import numpy as np
from scipy.signal import find_peaks

from loco_mujoco import ImitationFactory
from loco_mujoco.task_factories import DefaultDatasetConf

def extract_locomujoco_data(env_name="SkeletonTorque", task="walk", dataset_type="mocap"):
    mdp = ImitationFactory.make(
        env_name,
        default_dataset_conf=DefaultDatasetConf(task=task, dataset_type=dataset_type),
        timestep=0.001,
        n_substeps=10,
        horizon=1000,
        use_box_feet=True,
        disable_arms=True,
        headless=True)

    traj = mdp.th.traj
    qpos = np.array(traj.data.qpos)
    split_points = np.array(traj.data.split_points)
    joint_names = [name for name in traj.info.joint_names if name != "root"]
    joint_name2ind = traj.info.joint_name2ind_qpos

    cycles = {f"q_{name}": [] for name in joint_names}
    for start, end in zip(split_points[:-1], split_points[1:]):
        segment = qpos[start:end]
        if segment.size == 0:
            continue

        series = {}
        for name in joint_names:
            idx = np.array(joint_name2ind[name]).reshape(-1)[0]
            series[name] = np.rad2deg(segment[:, idx])

        hip_series = series.get("hip_flexion_r")
        if hip_series is None:
            continue

        peaks, _ = find_peaks(hip_series, height=10, distance=80)
        if len(peaks) < 2:
            continue

        for i in range(len(peaks) - 1):
            for name in joint_names:
                key = f"q_{name}"
                cycles[key].append(series[name][peaks[i]:peaks[i + 1]])

    return cycles
