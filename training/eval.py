"""
This script evaluates a trained model against the LocoMuJoCo dataset at 1.25 m/s.
"""

import argparse
import os
import loco_mujoco
from loco_mujoco import LocoEnv
import numpy as np
from pathlib import Path
import importlib.util
from dm_control.mujoco import Physics
from utils import interpolate_data, eval_model, calculate_metrics

def _load_data_utils():
    data_utils_path = Path(__file__).resolve().parents[1] / "data" / "utils.py"
    spec = importlib.util.spec_from_file_location("data_utils", data_utils_path)
    data_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_utils)
    return data_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model by providing model checkpoint')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    args = parser.parse_args()
    assert args.model_path is not None, 'Please provide model checkpoint path'

    library_path = os.path.dirname(loco_mujoco.__file__)
    file_path = os.path.join(
        library_path,
        'environments',
        'data',
        'humanoid',
        'humanoid_torque.xml')
    physics = Physics.from_xml_path(file_path)
    mass = np.sum(physics.named.model.body_mass._field)
    target_speed = 1.25
    speed_range = [target_speed]



    # mdp = LocoEnv.make("HumanoidTorque.walk.real", headless=True)
    mdp = LocoEnv.make("HumanoidTorque.walk.real", headless=False)
    
    
    
    _ = mdp.reset()

    data_utils = _load_data_utils()
    locomujoco_cycles = data_utils.extract_locomujoco_data()
    joint_keys = ['q_hip_flexion_r', 'q_knee_angle_r', 'q_ankle_angle_r']
    single_speed_data = {
        target_speed: {
            joint: locomujoco_cycles[joint] for joint in joint_keys
        }
    }
    mean_length = round(np.mean([len(cycle) for cycle in locomujoco_cycles[joint_keys[0]]]))
    interpolated_data = interpolate_data(single_speed_data, mean_length)

    # Specify model path
    model_path = args.model_path

    
    
    
    # Model inference and calculate RMSE and R2
    # results = eval_model(
    #     mdp,
    #     model_path,
    #     speed_range,
    #     mass,
    #     n_trials=5,
    #     n_episodes=3,
    #     cycle_length_cutoff=60,
    #     record=False)
    results = eval_model(
        mdp,
        model_path,
        speed_range,
        mass,
        n_trials=5,
        n_episodes=3,
        cycle_length_cutoff=60,
        record=True)
    


    processed_results = {}
    metric = calculate_metrics(results, interpolated_data)
    speed_errors, bio_RMSEs, bio_R2s = [], [], []
    for trial in metric:
        trial_data = metric[trial][target_speed]
        speed_errors.append(abs(trial_data['actual_speed'] - target_speed))
        bio_RMSEs.append(trial_data['RMSE'])
        bio_R2s.append(trial_data['R2'])
    processed_results['speed_error'] = speed_errors
    processed_results['bio_RMSE'] = bio_RMSEs
    processed_results['bio_R2'] = bio_R2s

    bio_RMSE_avg = np.mean(processed_results['bio_RMSE'])
    bio_RMSE_std = np.std(processed_results['bio_RMSE'])
    bio_R2_avg = np.mean(processed_results['bio_R2'])
    bio_R2_std = np.std(processed_results['bio_R2'])

    speed_error_avg = np.mean(processed_results['speed_error'])
    speed_error_std = np.std(processed_results['speed_error'])

    print(f'bio RMSE avg {bio_RMSE_avg}')
    print(f'bio RMSE std {bio_RMSE_std}')
    print(f'bio R2 avg {bio_R2_avg}')
    print(f'bio R2 std {bio_R2_std}')

    print(f'speed error avg {speed_error_avg}')
    print(f'speed error std {speed_error_std}')



