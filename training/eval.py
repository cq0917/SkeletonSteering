import argparse
import numpy as np
from pathlib import Path
import importlib.util
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from loco_mujoco import RLFactory
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf
from utils import interpolate_data, eval_model, calculate_metrics, wrap_mdp_for_mushroom

def _load_data_utils():
    data_utils_path = Path(__file__).resolve().parents[1] / "data" / "utils.py"
    spec = importlib.util.spec_from_file_location("data_utils", data_utils_path)
    data_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_utils)
    return data_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model by providing model checkpoint')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--compare_expert', action='store_true', help='Compare against expert trajectories')
    parser.add_argument('--record', action='store_true', help='Record video (requires ffmpeg)')
    parser.add_argument('--deterministic', action='store_true', help='Use mean action for evaluation')
    parser.add_argument('--print_joint_stats', action='store_true', help='Print policy joint angle stats')
    parser.add_argument(
        '--eval_env',
        choices=['imitation', 'rl'],
        default='imitation',
        help='Evaluation environment: imitation (with trajectories) or rl (no trajectories).')
    args = parser.parse_args()
    assert args.model_path is not None, 'Please provide model checkpoint path'

    env_name = "SkeletonTorque"
    task = "walk"
    dataset_type = "mocap"
    target_speed = 1.25
    speed_range = [target_speed]



    if args.eval_env == "imitation":
        mdp = ImitationFactory.make(
            env_name,
            default_dataset_conf=DefaultDatasetConf(task=task, dataset_type=dataset_type),
            reward_type="TargetXVelocityReward",
            reward_params=dict(target_velocity=target_speed),
            timestep=0.001,
            n_substeps=10,
            horizon=1000,
            use_box_feet=True,
            disable_arms=True,
            headless=not args.record)
    else:
        mdp = RLFactory.make(
            env_name,
            goal_type="NoGoal",
            reward_type="TargetXVelocityReward",
            reward_params=dict(target_velocity=target_speed),
            timestep=0.001,
            n_substeps=10,
            horizon=1000,
            use_box_feet=True,
            disable_arms=True,
            headless=not args.record)
    mdp = wrap_mdp_for_mushroom(mdp)
    mass = float(np.sum(mdp._model.body_mass))
    
    
    
    _ = mdp.reset()

    interpolated_data = None
    if args.compare_expert:
        data_utils = _load_data_utils()
        locomujoco_cycles = data_utils.extract_locomujoco_data(
            env_name=env_name,
            task=task,
            dataset_type=dataset_type)
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
        record=args.record,
        deterministic=args.deterministic,
        print_joint_stats=args.print_joint_stats)
    


    speed_errors = []
    for trial in results:
        trial_speed = trial[target_speed]['mean_speed']
        speed_errors.append(abs(trial_speed - target_speed))
    speed_error_avg = np.mean(speed_errors)
    speed_error_std = np.std(speed_errors)

    print(f'speed error avg {speed_error_avg}')
    print(f'speed error std {speed_error_std}')

    if args.compare_expert and interpolated_data is not None:
        metric = calculate_metrics(results, interpolated_data)
        bio_RMSEs, bio_R2s = [], []
        for trial in metric:
            trial_data = metric[trial][target_speed]
            bio_RMSEs.append(trial_data['RMSE'])
            bio_R2s.append(trial_data['R2'])
        print(f'bio RMSE avg {np.mean(bio_RMSEs)}')
        print(f'bio RMSE std {np.std(bio_RMSEs)}')
        print(f'bio R2 avg {np.mean(bio_R2s)}')
        print(f'bio R2 std {np.std(bio_R2s)}')
