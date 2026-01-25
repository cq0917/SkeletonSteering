#!/usr/bin/env python3
import argparse
import datetime
import os
import random

import numpy as np
from omegaconf import OmegaConf, open_dict


def _parse_env_id(env_id: str):
    parts = env_id.split(".")
    env_name = parts[0] if parts else env_id
    task = parts[1] if len(parts) > 1 else "walk"
    dataset_type = parts[2] if len(parts) > 2 else "mocap"
    return env_name, task, dataset_type


def _apply_env_id_overrides(config, env_id: str):
    env_name, task, dataset_type = _parse_env_id(env_id)
    mjx_env_name = env_name if env_name.startswith("Mjx") else f"Mjx{env_name}"
    target_velocity = 2.5 if task == "run" else 1.25

    with open_dict(config):
        config.env.env_params.env_name = mjx_env_name
        config.env.task_factory.params.default_dataset_conf.task = task
        config.env.task_factory.params.default_dataset_conf.dataset_type = dataset_type
        config.env.env_params.reward_params.target_velocity = target_velocity


def main():
    parser = argparse.ArgumentParser(description="Train AMP+PPO (JAX) with AMP-style discriminator observations.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "amp_ppo_jax_conf.yaml"))
    parser.add_argument("--env_id", default=None,
                        help="Format: EnvName.task.dataset_type (e.g., SkeletonTorque.walk.mocap)")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jax_cpu", action="store_true", help="Force JAX to use CPU.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic execution (GPU: XLA deterministic ops, "
                             "CPU: use with --jax_cpu).")
    args = parser.parse_args()

    if args.deterministic:
        xla_flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_gpu_deterministic_ops" not in xla_flags:
            xla_flags = (xla_flags + " --xla_gpu_deterministic_ops").strip()
        os.environ["XLA_FLAGS"] = xla_flags

    if args.jax_cpu:
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    np.random.seed(args.seed)
    random.seed(args.seed)

    from amp_ppo_jax import train

    config = OmegaConf.load(args.config)
    if args.env_id:
        _apply_env_id_overrides(config, args.env_id)

    if args.results_dir:
        results_dir = args.results_dir
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(os.path.dirname(__file__), results_dir)
    else:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = os.path.join(os.path.dirname(__file__), "logs", f"amp_ppo_jax_{stamp}")

    train(config, results_dir, seed=args.seed)


if __name__ == "__main__":
    main()


'''
训练：
python training/train_amp_ppo_jax.py --seed 42 \
    --config training/amp_ppo_jax_conf.yaml --env_id SkeletonTorque.walk.mocap

tensorboard --logdir 'training/logs'



评估 (AMP 保存为 .pkl):
export MUJOCO_GL=glfw
python training/eval_amp_ppo_jax.py \
    --record --n_steps 2000 --deterministic \
    --model_path training/logs/amp_ppo_jax_2026-01-25_17-16-15/amp_ppo_jax_best.pkl
'''