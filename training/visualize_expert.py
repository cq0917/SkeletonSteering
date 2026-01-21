#!/usr/bin/env python3
import argparse
import os

import numpy as np


def _parse_env_id(env_id: str):
    parts = env_id.split(".")
    env_name = parts[0] if parts else env_id
    task = parts[1] if len(parts) > 1 else "walk"
    dataset_type = parts[2] if len(parts) > 2 else "mocap"
    return env_name, task, dataset_type


def _get_obs_index(env, obs_name, entry_index=0):
    obs = env.obs_container[obs_name]
    obs_ind = np.array(obs.obs_ind).reshape(-1)
    return int(obs_ind[entry_index])


def _get_expert_observations(env):
    transitions = env.create_dataset()
    if hasattr(transitions, "to_np"):
        transitions = transitions.to_np()

    observations = getattr(transitions, "observations", None)
    if observations is None and isinstance(transitions, dict):
        observations = transitions.get("observations")
    return observations


def _print_speed_stats(env, observations):
    if observations is None:
        print("No observations found in the expert dataset; cannot compute speed stats.")
        return

    try:
        x_vel_idx = _get_obs_index(env, "dq_root", entry_index=0)
    except Exception:
        keys = sorted(getattr(env, "obs_container", {}).keys())
        print("Observation 'dq_root' not found; available keys:")
        print(", ".join(keys))
        return

    x_vel = np.asarray(observations)[:, x_vel_idx]
    if x_vel.size == 0:
        print("No velocity samples found; cannot compute speed stats.")
        return

    p10, p50, p90 = np.percentile(x_vel, [10, 50, 90])
    print("Expert dataset root-x velocity stats (m/s):")
    print(f"  count={x_vel.size}")
    print(f"  mean={x_vel.mean():.4f} std={x_vel.std():.4f}")
    print(f"  min={x_vel.min():.4f} p10={p10:.4f} p50={p50:.4f} p90={p90:.4f} max={x_vel.max():.4f}")


def _print_joint_angle_stats(env, observations, joint_keys):
    if observations is None:
        print("No observations found in the expert dataset; cannot compute joint stats.")
        return

    print("Expert dataset joint angle stats (deg):")
    for key in joint_keys:
        try:
            idx = _get_obs_index(env, key)
        except Exception:
            continue
        angles = np.rad2deg(np.asarray(observations)[:, idx])
        if angles.size == 0:
            continue
        p10, p50, p90 = np.percentile(angles, [10, 50, 90])
        print(f"  {key}: min={angles.min():.2f} p10={p10:.2f} p50={p50:.2f} p90={p90:.2f} max={angles.max():.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize expert trajectories used for training.")
    parser.add_argument(
        "--env_id",
        default="SkeletonTorque.walk.mocap",
        help="Format: EnvName.task.dataset_type (e.g., SkeletonTorque.walk.mocap)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print expert root-x velocity statistics from the default dataset.")
    parser.add_argument(
        "--jax_cpu",
        action="store_true",
        help="Force JAX to use CPU (avoids GPU/cuDNN issues).")
    args = parser.parse_args()

    if args.jax_cpu:
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf

    env_name, task, dataset_type = _parse_env_id(args.env_id)
    env = ImitationFactory.make(
        env_name,
        default_dataset_conf=DefaultDatasetConf(task=task, dataset_type=dataset_type),
        timestep=0.001,
        n_substeps=10,
        horizon=1000,
        use_box_feet=True,
        disable_arms=True,
        headless=args.headless)

    if args.stats:
        observations = _get_expert_observations(env)
        _print_speed_stats(env, observations)
        joint_keys = [
            "q_hip_flexion_l", "q_knee_angle_l", "q_ankle_angle_l",
            "q_hip_flexion_r", "q_knee_angle_r", "q_ankle_angle_r",
        ]
        _print_joint_angle_stats(env, observations, joint_keys)

    env.play_trajectory(
        n_episodes=args.episodes,
        n_steps_per_episode=args.steps,
        render=not args.headless,
        record=args.record)


if __name__ == "__main__":
    main()

# python visualize_expert.py --env_id SkeletonTorque.walk.mocap --stats 
# 专家数据确实是近似 1.25 m/s 的直行走路