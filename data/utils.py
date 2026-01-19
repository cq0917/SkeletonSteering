"""
Utility functions for extracting LocoMuJoCo demonstration data.
"""

from collections import Counter
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
from loco_mujoco import LocoEnv
import mujoco

def extract_locomujoco_data():
    mdp = LocoEnv.make(
        "HumanoidTorque.walk.real",
        headless=True, random_start=False, init_step_no=0)
    num_steps = 89800-1
    samples = []
    obss = []
    mdp._init_step_no = 0
    mdp.reset()
    sample = mdp.trajectories.get_current_sample()
    obs = mdp._create_observation(np.concatenate(sample))

    samples.append(sample)
    obss.append(obs)

    for _ in tqdm(range(num_steps), desc='Loading LocoMuJoCo Data'):
        mdp.set_sim_state(sample)
        mdp._simulation_pre_step()
        mujoco.mj_forward(mdp._model, mdp._data)
        mdp._simulation_post_step()
        sample = mdp.trajectories.get_next_sample()
        obs = mdp._create_observation(np.concatenate(sample))
        samples.append(sample)
        obss.append(obs)

    mdp.reset()
    mdp.stop()
    joints = [joint[0] for joint in mdp.obs_helper.observation_spec]
    np_obss = np.array(obss)
    np_samples = np.array([np.array(sample).flatten() for sample in samples])
    df = pd.DataFrame()
    df['Timestep'] = np.arange(len(np_samples))
    for joint in joints:
        # skip velocity fields
        if joint[0] == 'd':
            continue
        if joint == 'q_pelvis_tx':
            df[joint] = np_samples[:, 0].flatten()
        elif joint == 'q_pelvis_tz':
            df[joint] = np_samples[:, 1].flatten()
        else:
            joint_idx = mdp.get_obs_idx(joint)
            df[joint] = np_obss[:, joint_idx].flatten()
    not_angle_joints = {'Timestep', 'q_pelvis_tx', 'q_pelvis_tz', 'q_pelvis_ty'}
    for column in df.columns:
        if column not in not_angle_joints:
            df[column] = np.rad2deg(df[column])
    peaks, _ = find_peaks(df['q_hip_flexion_r'], height=10, distance=80)
    heelstrike = [False] * 89800
    for peak in peaks:
        heelstrike[peak] = True

    df['Heelstrike'] = np.array(heelstrike)
    cycle_idx = 0
    cycle_idxs = []
    for heelstrike in df['Heelstrike']:
        if heelstrike:
            cycle_idx += 1
        cycle_idxs.append(cycle_idx)
    df = df.assign(Cycle_Idx = cycle_idxs)
    counter = Counter(cycle_idxs)
    proper_cycles_idxs = [idx for idx in counter if counter[idx] > 1]

    cycles = {}
    for idx in proper_cycles_idxs:
        df_ = df[df['Cycle_Idx'] == idx]
        for key in joints:
            if key[0] != 'd':
                y = df_[key].to_numpy()
                if key not in cycles:
                    cycles[key] = []
                cycles[key].append(y)
    return cycles
