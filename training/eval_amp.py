#!/usr/bin/env python3
import argparse
import datetime
import os
import numpy as np
import jax.numpy as jnp
from flax import struct

from omegaconf import OmegaConf

from loco_mujoco import TaskFactory
from amp_override import AMPJax
from loco_mujoco.core.utils import Box, MDPInfo


def _strip_mjx_prefix(env_name: str) -> str:
    return env_name[3:] if env_name.startswith("Mjx") else env_name


class MjxObsHistoryWrapper:
    def __init__(self, env, history_len: int):
        self.env = env
        self.history_len = history_len

        info = env.info
        low = np.tile(info.observation_space.low, history_len)
        high = np.tile(info.observation_space.high, history_len)
        self._mdp_info = MDPInfo(Box(low, high), info.action_space, info.gamma, info.horizon, info.dt)

    @property
    def info(self):
        return self._mdp_info

    @property
    def mdp_info(self):
        return self._mdp_info

    def reset(self, rng_key):
        if hasattr(self.env, "mjx_reset"):
            env_state = self.env.mjx_reset(rng_key)
            obs = env_state.observation
        else:
            obs, env_state = self.env.reset(rng_key)
        buffer = jnp.tile(jnp.zeros_like(obs), (self.history_len, 1))
        buffer = buffer.at[-1].set(obs)
        state = ObsHistoryState(env_state=env_state, observation_buffer=buffer)
        return jnp.reshape(buffer, (-1,)), state

    def step(self, state, action):
        env_state, buffer = state.env_state, state.observation_buffer
        if hasattr(self.env, "mjx_step"):
            next_state = self.env.mjx_step(env_state, action)
            obs = jnp.where(next_state.done,
                            next_state.additional_carry.final_observation,
                            next_state.observation)
            reward = next_state.reward
            absorbing = next_state.absorbing
            done = next_state.done
            info = next_state.info
            env_state = next_state
        else:
            obs, reward, absorbing, done, info, env_state = self.env.step(env_state, action)
        buffer = jnp.roll(buffer, shift=-1, axis=0)
        buffer = buffer.at[-1].set(obs)
        obs_out = jnp.reshape(buffer, (-1,))
        reset_buffer = jnp.where(done, jnp.zeros_like(buffer), buffer)
        next_state = ObsHistoryState(env_state=env_state, observation_buffer=reset_buffer)
        return obs_out, reward, absorbing, done, info, next_state

    def __getattr__(self, name):
        return getattr(self.env, name)


class MujocoObsHistoryWrapper:
    def __init__(self, env, history_len: int):
        self.env = env
        self.history_len = history_len
        self._buffer = None

        info = env.info
        low = np.tile(info.observation_space.low, history_len)
        high = np.tile(info.observation_space.high, history_len)
        self._mdp_info = MDPInfo(Box(low, high), info.action_space, info.gamma, info.horizon, info.dt)

    @property
    def info(self):
        return self._mdp_info

    @property
    def mdp_info(self):
        return self._mdp_info

    def reset(self):
        obs = self.env.reset()
        self._buffer = np.zeros((self.history_len, obs.shape[0]), dtype=obs.dtype)
        self._buffer[-1] = obs
        return self._buffer.reshape(-1)

    def step(self, action):
        obs, reward, absorbing, done, info = self.env.step(action)
        self._buffer = np.roll(self._buffer, shift=-1, axis=0)
        self._buffer[-1] = obs
        obs_out = self._buffer.reshape(-1)
        if done:
            self._buffer = np.zeros_like(self._buffer)
        return obs_out, reward, absorbing, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


@struct.dataclass
class ObsHistoryState:
    env_state: object
    observation_buffer: jnp.ndarray

    def __getattr__(self, name):
        try:
            return getattr(self.env_state, name)
        except AttributeError as e:
            raise AttributeError(f"Attribute '{name}' not found in any env state nor the MjxState.") from e


def main():
    parser = argparse.ArgumentParser(description="Evaluate AMPJax policy.")
    parser.add_argument("--model_path", required=True, help="Path to AMPJax .pkl file.")
    parser.add_argument("--n_steps", type=int, default=2000, help="Number of steps to run.")
    parser.add_argument("--record", action="store_true", help="Record a video.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy.")
    parser.add_argument("--mjx", action="store_true", help="Evaluate in MJX (no MuJoCo rendering).")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    agent_conf, agent_state = AMPJax.load_agent(args.model_path)
    config = agent_conf.config

    OmegaConf.set_struct(config, False)
    if args.record:
        record_root = os.path.join(os.path.dirname(__file__), "mushroom_rl_recordings")
        tag = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        recorder_params = dict(path=record_root, tag=tag)
        config.experiment.env_params["recorder_params"] = recorder_params

    if "viewer_params" in config.experiment.env_params:
        viewer_params = config.experiment.env_params.pop("viewer_params") or {}
        for key, value in viewer_params.items():
            config.experiment.env_params[key] = value
    if args.mjx:
        config.experiment.env_params["headless"] = True
    else:
        config.experiment.env_params["env_name"] = _strip_mjx_prefix(config.experiment.env_params["env_name"])
        config.experiment.env_params["headless"] = not args.record

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    history_len = int(getattr(config.experiment, "obs_history_len",
                               getattr(config.experiment, "len_obs_history", 1)) or 1)
    if history_len > 1:
        env = MjxObsHistoryWrapper(env, history_len) if args.mjx else MujocoObsHistoryWrapper(env, history_len)

    if args.mjx:
        AMPJax.play_policy(env, agent_conf, agent_state,
                           n_envs=1,
                           n_steps=args.n_steps,
                           record=args.record,
                           deterministic=args.deterministic,
                           train_state_seed=0)
    else:
        AMPJax.play_policy_mujoco(env, agent_conf, agent_state,
                                  n_steps=args.n_steps,
                                  record=args.record,
                                  deterministic=args.deterministic,
                                  train_state_seed=0)


if __name__ == "__main__":
    main()
