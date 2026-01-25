#!/usr/bin/env python3
import argparse
import datetime
import os

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from amp_ppo_jax import (MLP, load_checkpoint, _init_obs_history, _push_obs_history,
                         _reset_obs_history, _flatten_obs_history)
from loco_mujoco.task_factories import TaskFactory
from loco_mujoco.trajectory import Trajectory
from loco_mujoco.core.wrappers.mjx import VecEnv


def _strip_mjx_prefix(env_name: str):
    return env_name[3:] if env_name.startswith("Mjx") else env_name


def _build_actor(config, obs_dim, action_dim, params):
    actor = MLP(config.model.actor_hidden_layers, action_dim,
                activation=config.model.activation, out_init_scale=config.model.actor_init_scale)
    return actor, params["actor"]


def _apply_obs_norm(obs, mean, std, clip):
    return jnp.clip((obs - mean) / std, -clip, clip)


def _make_env(config, record: bool, mjx: bool):
    env_name = config.env.env_params.env_name
    if mjx:
        if not env_name.startswith("Mjx"):
            env_name = f"Mjx{env_name}"
    else:
        env_name = _strip_mjx_prefix(env_name)

    env_params = dict(config.env.env_params)
    env_params["env_name"] = env_name

    if record:
        record_root = os.path.join(os.path.dirname(__file__), "mushroom_rl_recordings")
        tag = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        env_params["recorder_params"] = dict(path=record_root, tag=tag)

    if mjx:
        env_params["headless"] = True
    else:
        env_params["headless"] = not record

    factory = TaskFactory.get_factory_cls(config.env.task_factory.name)
    params = OmegaConf.to_container(config.env.task_factory.params, resolve=True) or {}
    custom_path = params.pop("custom_traj_path", None) or params.pop("custom_dataset_path", None)
    if custom_path:
        try:
            traj = Trajectory.load(custom_path)
            params["custom_dataset_conf"] = {"traj": traj}
            params["default_dataset_conf"] = None
            params["amass_dataset_conf"] = None
            params["lafan1_dataset_conf"] = None
            print(f"[AMP PPO] Using custom_traj_path for eval: {custom_path}")
        except Exception as exc:
            print(f"[AMP PPO] Failed to load custom_traj_path '{custom_path}': {exc}")
    return factory.make(**env_params, **params)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AMP+PPO JAX policy.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--mjx", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config, payload = load_checkpoint(args.model_path)
    params = payload["params"]

    env = _make_env(config, record=args.record, mjx=args.mjx)

    obs_dim = int(np.prod(env.info.observation_space.shape))
    action_dim = int(np.prod(env.info.action_space.shape))
    obs_hist_len = int(config.disc_obs.obs_history_len)

    obs_norm = payload["obs_norm"]
    obs_mean = jnp.asarray(obs_norm["mean"])
    obs_std = jnp.asarray(obs_norm["std"])
    obs_clip = float(obs_norm["clip"])
    action_mean = jnp.asarray(payload["action_mean"])
    action_std = jnp.asarray(payload["action_std"])
    log_std = jnp.log(jnp.ones((action_dim,), dtype=jnp.float32) * config.model.action_std)

    actor, actor_params = _build_actor(config, obs_dim * obs_hist_len, action_dim, params)

    rng = jax.random.PRNGKey(args.seed)

    if args.mjx:
        env = VecEnv(env)
        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, 1)
        obs, state = env.reset(reset_keys)
        obs_hist = _init_obs_history(obs, obs_hist_len)

        for _ in range(args.n_steps):
            obs_hist = _push_obs_history(obs_hist, obs)
            flat_obs = _flatten_obs_history(obs_hist)
            norm_obs = _apply_obs_norm(flat_obs, obs_mean, obs_std, obs_clip)

            mean = actor.apply({"params": actor_params}, norm_obs)
            if args.deterministic:
                norm_action = mean
            else:
                rng, act_key = jax.random.split(rng)
                norm_action = mean + jax.random.normal(act_key, mean.shape) * jnp.exp(log_std)
            action = norm_action * action_std + action_mean

            next_obs, reward, absorbing, done, info, state = env.step(state, action)
            if args.record:
                env.unwrapped().mjx_render(state, record=True)

            obs = state.observation
            obs_hist = _reset_obs_history(obs_hist, obs, done)
        if args.record:
            env.unwrapped().stop()
            video_path = env.unwrapped().video_file_path
            if video_path:
                print(f"Video saved to: {video_path}")
    else:
        obs = env.reset()
        obs = jnp.asarray(obs)[None, :]
        obs_hist = _init_obs_history(obs, obs_hist_len)

        for _ in range(args.n_steps):
            obs_hist = _push_obs_history(obs_hist, obs)
            flat_obs = _flatten_obs_history(obs_hist)
            norm_obs = _apply_obs_norm(flat_obs, obs_mean, obs_std, obs_clip)
            mean = actor.apply({"params": actor_params}, norm_obs)
            if args.deterministic:
                norm_action = mean
            else:
                rng, act_key = jax.random.split(rng)
                norm_action = mean + jax.random.normal(act_key, mean.shape) * jnp.exp(log_std)
            action = np.asarray(norm_action * action_std + action_mean)[0]

            obs, reward, absorbing, done, info = env.step(action)
            if args.record:
                env.render(record=True)
            obs = jnp.asarray(obs)[None, :]
            obs_hist = _reset_obs_history(obs_hist, obs, jnp.asarray([done]))
            if done:
                obs = jnp.asarray(env.reset())[None, :]
                obs_hist = _init_obs_history(obs, obs_hist_len)
        if args.record and hasattr(env, "stop"):
            env.stop()
            if hasattr(env, "video_file_path") and env.video_file_path:
                print(f"Video saved to: {env.video_file_path}")


if __name__ == "__main__":
    main()
