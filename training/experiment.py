import os

import numpy as np
import jax.numpy as jnp
from flax import struct
from omegaconf import OmegaConf, open_dict
from torch.utils.tensorboard import SummaryWriter
from experiment_launcher import run_experiment

from loco_mujoco import TaskFactory
from amp_override import AMPJax
from loco_mujoco.core.utils import Box, MDPInfo
from loco_mujoco.trajectory import Trajectory
from loco_mujoco.trajectory.dataclasses import TrajectoryTransitions


def _parse_env_id(env_id: str):
    parts = env_id.split(".")
    env_name = parts[0] if parts else env_id
    task = parts[1] if len(parts) > 1 else "walk"
    dataset_type = parts[2] if len(parts) > 2 else "mocap"
    return env_name, task, dataset_type


def _load_amp_config(conf_path: str):
    return OmegaConf.load(conf_path)


def _apply_amp_overrides(config,
                         env_id: str,
                         total_timesteps: int,
                         num_envs: int,
                         num_steps: int,
                         num_minibatches: int,
                         update_epochs: int,
                         disc_minibatch_size: int,
                         n_disc_epochs: int,
                         proportion_env_reward: float,
                         lr: float,
                         disc_lr: float,
                         gamma: float,
                         disable_arms: bool,
                         use_box_feet: bool):
    env_name, task, dataset_type = _parse_env_id(env_id)
    target_velocity = 2.5 if task == "run" else 1.25
    mjx_env_name = env_name if env_name.startswith("Mjx") else f"Mjx{env_name}"

    with open_dict(config):
        config.experiment.env_params.env_name = mjx_env_name
        config.experiment.env_params.disable_arms = disable_arms
        config.experiment.env_params.use_box_feet = use_box_feet
        config.experiment.env_params.reward_params.target_velocity = target_velocity
        config.experiment.task_factory.params.default_dataset_conf.task = task
        config.experiment.task_factory.params.default_dataset_conf.dataset_type = dataset_type

        config.experiment.total_timesteps = int(total_timesteps)
        config.experiment.num_envs = int(num_envs)
        config.experiment.num_steps = int(num_steps)
        config.experiment.num_minibatches = int(num_minibatches)
        config.experiment.update_epochs = int(update_epochs)
        config.experiment.disc_minibatch_size = int(disc_minibatch_size)
        config.experiment.n_disc_epochs = int(n_disc_epochs)
        config.experiment.proportion_env_reward = float(proportion_env_reward)
        config.experiment.lr = float(lr)
        config.experiment.disc_lr = float(disc_lr)
        config.experiment.gamma = float(gamma)
        # Workaround for loco-mujoco anneal_lr schedule referencing root-level lr.
        config.lr = float(lr)
        config.disc_lr = float(disc_lr)

    return target_velocity


def _maybe_set_jax_platform(use_cuda: bool):
    if not use_cuda:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"


def _load_or_build_expert_dataset(env, cache_path: str):
    if os.path.exists(cache_path):
        traj = Trajectory.load(cache_path)
        env.load_trajectory(traj)
        return env.create_dataset()

    expert_dataset = env.create_dataset()
    traj = Trajectory(info=env.th.traj.info,
                      data=env.th.traj.data,
                      obs_container=env.obs_container,
                      transitions=expert_dataset)
    traj.save(cache_path)
    return expert_dataset


def _stack_transitions_with_history(observations, next_observations, dones, history_len: int):
    observations = np.asarray(observations)
    next_observations = np.asarray(next_observations)
    if history_len <= 1:
        return observations, next_observations

    dones_arr = None
    if dones is not None:
        dones_arr = np.asarray(dones).astype(bool)
        if dones_arr.size == 0:
            dones_arr = None
        elif dones_arr.ndim > 1:
            dones_arr = np.squeeze(dones_arr)

    n_samples, obs_dim = observations.shape
    stacked_obs = np.zeros((n_samples, obs_dim * history_len), dtype=observations.dtype)
    stacked_next = np.zeros((n_samples, obs_dim * history_len), dtype=observations.dtype)
    buffer = np.zeros((history_len, obs_dim), dtype=observations.dtype)

    for i in range(n_samples):
        if i == 0 or (dones_arr is not None and dones_arr[i - 1]):
            buffer.fill(0)

        buffer = np.roll(buffer, shift=-1, axis=0)
        buffer[-1] = observations[i]
        stacked_obs[i] = buffer.reshape(-1)

        buffer_next = np.roll(buffer, shift=-1, axis=0)
        buffer_next[-1] = next_observations[i]
        stacked_next[i] = buffer_next.reshape(-1)

        if dones_arr is not None and dones_arr[i]:
            buffer.fill(0)
        else:
            buffer = buffer_next

    return stacked_obs, stacked_next


def _stack_expert_dataset(expert_dataset, history_len: int):
    if history_len <= 1:
        return expert_dataset

    dataset_np = expert_dataset.to_np()
    stacked_obs, stacked_next = _stack_transitions_with_history(
        dataset_np.observations, dataset_np.next_observations, dataset_np.dones, history_len)

    import jax.numpy as jnp
    return TrajectoryTransitions(
        observations=jnp.array(stacked_obs),
        next_observations=jnp.array(stacked_next),
        absorbings=jnp.array(dataset_np.absorbings),
        dones=jnp.array(dataset_np.dones),
        actions=jnp.array(dataset_np.actions) if dataset_np.actions.size else dataset_np.actions,
        rewards=jnp.array(dataset_np.rewards) if dataset_np.rewards.size else dataset_np.rewards,
    )


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


@struct.dataclass
class ObsHistoryState:
    env_state: object
    observation_buffer: jnp.ndarray

    def __getattr__(self, name):
        try:
            return getattr(self.env_state, name)
        except AttributeError as e:
            raise AttributeError(f"Attribute '{name}' not found in any env state nor the MjxState.") from e


def _maybe_mean(values, n_seeds: int):
    if values is None:
        return None
    values = np.asarray(values)
    if n_seeds > 1 and values.ndim > 1:
        return np.mean(values, axis=0)
    return values


def _log_metrics(sw, metrics, prefix: str, n_seeds: int):
    steps = _maybe_mean(metrics.max_timestep, n_seeds)
    mean_return = _maybe_mean(metrics.mean_episode_return, n_seeds)
    mean_length = _maybe_mean(metrics.mean_episode_length, n_seeds)
    disc_policy = _maybe_mean(getattr(metrics, "discriminator_output_policy", None), n_seeds)
    disc_expert = _maybe_mean(getattr(metrics, "discriminator_output_expert", None), n_seeds)
    reward_env = _maybe_mean(getattr(metrics, "reward_env", None), n_seeds)
    reward_disc = _maybe_mean(getattr(metrics, "reward_disc", None), n_seeds)
    reward_total = _maybe_mean(getattr(metrics, "reward_total", None), n_seeds)
    disc_loss = _maybe_mean(getattr(metrics, "disc_loss", None), n_seeds)
    disc_acc = _maybe_mean(getattr(metrics, "disc_acc", None), n_seeds)
    disc_entropy = _maybe_mean(getattr(metrics, "disc_entropy", None), n_seeds)
    disc_gap = _maybe_mean(getattr(metrics, "disc_output_gap", None), n_seeds)
    policy_loss = _maybe_mean(getattr(metrics, "policy_loss", None), n_seeds)
    value_loss = _maybe_mean(getattr(metrics, "value_loss", None), n_seeds)
    policy_entropy = _maybe_mean(getattr(metrics, "policy_entropy", None), n_seeds)
    approx_kl = _maybe_mean(getattr(metrics, "approx_kl", None), n_seeds)
    clip_frac = _maybe_mean(getattr(metrics, "clip_frac", None), n_seeds)

    for i in range(len(steps)):
        step = int(steps[i])
        sw.add_scalar(f"{prefix}/MeanEpisodeReturn", float(mean_return[i]), step)
        sw.add_scalar(f"{prefix}/MeanEpisodeLength", float(mean_length[i]), step)
        if reward_env is not None:
            sw.add_scalar(f"{prefix}/Reward/Env", float(reward_env[i]), step)
        if reward_disc is not None:
            sw.add_scalar(f"{prefix}/Reward/Disc", float(reward_disc[i]), step)
        if reward_total is not None:
            sw.add_scalar(f"{prefix}/Reward/Total", float(reward_total[i]), step)
        if disc_policy is not None:
            sw.add_scalar(f"{prefix}/DiscriminatorPolicy", float(disc_policy[i]), step)
        if disc_expert is not None:
            sw.add_scalar(f"{prefix}/DiscriminatorExpert", float(disc_expert[i]), step)
        if disc_loss is not None:
            sw.add_scalar(f"{prefix}/DiscriminatorLoss", float(disc_loss[i]), step)
        if disc_acc is not None:
            sw.add_scalar(f"{prefix}/DiscriminatorAcc", float(disc_acc[i]), step)
        if disc_entropy is not None:
            sw.add_scalar(f"{prefix}/DiscriminatorEntropy", float(disc_entropy[i]), step)
        if disc_gap is not None:
            sw.add_scalar(f"{prefix}/DiscriminatorOutputGap", float(disc_gap[i]), step)
        if policy_loss is not None:
            sw.add_scalar(f"{prefix}/PPO/PolicyLoss", float(policy_loss[i]), step)
        if value_loss is not None:
            sw.add_scalar(f"{prefix}/PPO/ValueLoss", float(value_loss[i]), step)
        if policy_entropy is not None:
            sw.add_scalar(f"{prefix}/PPO/Entropy", float(policy_entropy[i]), step)
        if approx_kl is not None:
            sw.add_scalar(f"{prefix}/PPO/ApproxKL", float(approx_kl[i]), step)
        if clip_frac is not None:
            sw.add_scalar(f"{prefix}/PPO/ClipFrac", float(clip_frac[i]), step)


def experiment(env_id: str = "SkeletonTorque.walk.mocap",
               total_timesteps: int = 20000000,
               num_envs: int = 256,
               num_steps: int = 16,
               num_minibatches: int = 16,
               update_epochs: int = 4,
               disc_minibatch_size: int = 2048,
               n_disc_epochs: int = 10,
               proportion_env_reward: float = 0.1,
               lr: float = 1.0e-4,
               disc_lr: float = 5.0e-5,
               gamma: float = 0.99,
               results_dir: str = './logs',
               use_cuda: bool = True,
               seed: int = 0,
               disable_arms: bool = True,
               use_box_feet: bool = True):
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _maybe_set_jax_platform(use_cuda)

    if not os.path.isabs(results_dir):
        results_dir = os.path.join(os.path.dirname(__file__), results_dir)
    results_dir = os.path.join(results_dir, str(seed))
    os.makedirs(results_dir, exist_ok=True)
    sw = SummaryWriter(log_dir=results_dir)

    conf_path = os.path.join(os.path.dirname(__file__), "amp_conf.yaml")
    config = _load_amp_config(conf_path)
    history_len = int(getattr(config.experiment, "obs_history_len",
                               getattr(config.experiment, "len_obs_history", 1)) or 1)
    with open_dict(config):
        config.experiment.obs_history_len = history_len
        config.experiment.len_obs_history = 1

    target_velocity = _apply_amp_overrides(
        config,
        env_id=env_id,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        num_steps=num_steps,
        num_minibatches=num_minibatches,
        update_epochs=update_epochs,
        disc_minibatch_size=disc_minibatch_size,
        n_disc_epochs=n_disc_epochs,
        proportion_env_reward=proportion_env_reward,
        lr=lr,
        disc_lr=disc_lr,
        gamma=gamma,
        disable_arms=disable_arms,
        use_box_feet=use_box_feet,
    )
    OmegaConf.save(config, os.path.join(results_dir, "amp_conf.yaml"))

    print(f"Starting AMP+PPO training {env_id} (target_velocity={target_velocity})...")

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    expert_cache = os.path.join(results_dir, "expert_traj.npz")
    expert_dataset = _load_or_build_expert_dataset(env, expert_cache)
    expert_dataset = _stack_expert_dataset(expert_dataset, history_len)
    if history_len > 1:
        env = MjxObsHistoryWrapper(env, history_len)

    import jax
    agent_conf = AMPJax.init_agent_conf(env, config)
    agent_conf = agent_conf.add_expert_dataset(expert_dataset)

    mh = None
    if config.experiment.validation.active:
        from loco_mujoco.utils import MetricsHandler
        mh = MetricsHandler(config, env)
    train_fn = AMPJax.build_train_fn(env, agent_conf, mh=mh)

    if config.experiment.n_seeds > 1:
        train_fn = jax.jit(jax.vmap(train_fn))
        rngs = jax.random.split(jax.random.PRNGKey(seed), config.experiment.n_seeds)
        out = train_fn(rngs)
    else:
        train_fn = jax.jit(train_fn)
        rng = jax.random.PRNGKey(seed)
        out = train_fn(rng)

    agent_state = out["agent_state"]
    AMPJax.save_agent(results_dir, agent_conf, agent_state)

    training_metrics = jax.tree.map(lambda x: np.asarray(x), out["training_metrics"])
    _log_metrics(sw, training_metrics, "Train", config.experiment.n_seeds)

    if config.experiment.validation.active:
        validation_metrics = jax.tree.map(lambda x: np.asarray(x), out["validation_metrics"])
        _log_metrics(sw, validation_metrics, "Validation", config.experiment.n_seeds)

    sw.flush()
    sw.close()
    print("Finished.")


if __name__ == "__main__":
    run_experiment(experiment)
