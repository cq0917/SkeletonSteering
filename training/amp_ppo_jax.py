import dataclasses
import os
import pickle
import time
from typing import Any, Dict, Iterable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from omegaconf import OmegaConf

import mujoco

from loco_mujoco.task_factories import TaskFactory
from loco_mujoco.core.wrappers.mjx import LogWrapper, VecEnv
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid, mj_jntname2qvelid
from loco_mujoco.trajectory import Trajectory

try:
    from tensorboardX import SummaryWriter as _SummaryWriter
except Exception:
    try:
        from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    except Exception:
        _SummaryWriter = None


@struct.dataclass
class NormalizerState:
    mean: jnp.ndarray
    mean_sq: jnp.ndarray
    std: jnp.ndarray
    count: jnp.ndarray
    clip: float = 10.0
    min_std: float = 1.0e-4


@struct.dataclass
class DiscReplayBuffer:
    data: jnp.ndarray
    size: jnp.ndarray
    index: jnp.ndarray


@struct.dataclass
class DiscHistory:
    root_pos: jnp.ndarray
    root_rot: jnp.ndarray
    root_vel: jnp.ndarray
    root_ang_vel: jnp.ndarray
    joint_rot: jnp.ndarray
    dof_vel: jnp.ndarray
    key_pos: jnp.ndarray


@struct.dataclass
class ObsHistory:
    obs: jnp.ndarray


@struct.dataclass
class TrainState:
    params: Dict[str, Any]
    opt_state: optax.OptState
    obs_norm: NormalizerState
    disc_norm: NormalizerState
    action_mean: jnp.ndarray
    action_std: jnp.ndarray
    replay: DiscReplayBuffer
    rng: jnp.ndarray


@dataclasses.dataclass(frozen=True)
class DiscObsSpec:
    root_qpos_idx: np.ndarray
    root_qvel_idx: np.ndarray
    joint_qpos_idx: np.ndarray
    joint_qvel_idx: np.ndarray
    joint_axis: np.ndarray
    key_body_idx: np.ndarray
    key_site_idx: np.ndarray


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    activation: str = "relu"
    out_init_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        act = _get_activation(self.activation)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = act(x)
        x = nn.Dense(self.out_dim, kernel_init=nn.initializers.uniform(self.out_init_scale))(x)
        return x


def _get_activation(name: str):
    name = (name or "relu").lower()
    if name == "tanh":
        return jnp.tanh
    if name == "relu":
        return jax.nn.relu
    if name == "swish":
        return jax.nn.swish
    raise ValueError(f"Unsupported activation: {name}")


def _init_normalizer(shape: Iterable[int], clip: float, min_std: float):
    shape = tuple(shape)
    mean = jnp.zeros(shape, dtype=jnp.float32)
    mean_sq = jnp.zeros(shape, dtype=jnp.float32)
    std = jnp.ones(shape, dtype=jnp.float32)
    count = jnp.zeros((), dtype=jnp.float32)
    return NormalizerState(mean=mean, mean_sq=mean_sq, std=std, count=count, clip=float(clip), min_std=float(min_std))


def _normalize(norm: NormalizerState, x: jnp.ndarray):
    x = (x - norm.mean) / norm.std
    return jnp.clip(x, -norm.clip, norm.clip)


def _update_normalizer(norm: NormalizerState, batch: jnp.ndarray):
    flat = batch.reshape((-1, batch.shape[-1]))
    batch_count = flat.shape[0]
    batch_mean = jnp.mean(flat, axis=0)
    batch_mean_sq = jnp.mean(jnp.square(flat), axis=0)

    total = norm.count + batch_count
    w_old = jnp.where(total > 0, norm.count / total, 0.0)
    w_new = jnp.where(total > 0, batch_count / total, 0.0)
    mean = w_old * norm.mean + w_new * batch_mean
    mean_sq = w_old * norm.mean_sq + w_new * batch_mean_sq
    var = jnp.maximum(mean_sq - jnp.square(mean), norm.min_std * norm.min_std)
    std = jnp.sqrt(var)
    return norm.replace(mean=mean, mean_sq=mean_sq, std=std, count=total)


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(jax.device_get(x))


def _init_replay_buffer(capacity: int, obs_dim: int, on_cpu: bool):
    if on_cpu:
        data = np.zeros((capacity, obs_dim), dtype=np.float32)
        size = np.int32(0)
        index = np.int32(0)
    else:
        data = jnp.zeros((capacity, obs_dim), dtype=jnp.float32)
        size = jnp.zeros((), dtype=jnp.int32)
        index = jnp.zeros((), dtype=jnp.int32)
    return DiscReplayBuffer(data=data, size=size, index=index)


def _replay_add(buffer: DiscReplayBuffer, samples: jnp.ndarray):
    capacity = buffer.data.shape[0]
    if isinstance(buffer.data, np.ndarray):
        samples_np = _to_numpy(samples)
        n = min(int(samples_np.shape[0]), int(capacity))
        if n <= 0:
            return buffer
        samples_np = samples_np[-n:]
        idx = (np.arange(n) + int(buffer.index)) % int(capacity)
        buffer.data[idx] = samples_np
        size = min(int(capacity), int(buffer.size) + n)
        index = (int(buffer.index) + n) % int(capacity)
        return buffer.replace(size=np.int32(size), index=np.int32(index))

    n = samples.shape[0]
    n = jnp.minimum(n, capacity)
    samples = samples[-n:]
    idx = (jnp.arange(n) + buffer.index) % capacity
    data = buffer.data.at[idx].set(samples)
    size = jnp.minimum(capacity, buffer.size + n)
    index = (buffer.index + n) % capacity
    return buffer.replace(data=data, size=size, index=index)


def _replay_sample(buffer: DiscReplayBuffer, rng: jnp.ndarray, n: int):
    if isinstance(buffer.data, np.ndarray):
        max_idx = max(int(buffer.size), 1)
        idx = jax.random.randint(rng, (n,), 0, max_idx)
        idx_host = _to_numpy(idx)
        return jnp.asarray(buffer.data[idx_host])

    max_idx = jnp.maximum(buffer.size, 1)
    idx = jax.random.randint(rng, (n,), 0, max_idx)
    return buffer.data[idx]


def _quat_wxyz_to_xyzw(q):
    return jnp.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _quat_mul(q, r):
    qx, qy, qz, qw = jnp.split(q, 4, axis=-1)
    rx, ry, rz, rw = jnp.split(r, 4, axis=-1)
    x = qw * rx + qx * rw + qy * rz - qz * ry
    y = qw * ry - qx * rz + qy * rw + qz * rx
    z = qw * rz + qx * ry - qy * rx + qz * rw
    w = qw * rw - qx * rx - qy * ry - qz * rz
    return jnp.concatenate([x, y, z, w], axis=-1)


def _quat_rotate(q, v):
    q_xyz = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * jnp.cross(q_xyz, v)
    return v + qw * t + jnp.cross(q_xyz, t)


def _axis_angle_to_quat(axis, angle):
    axis = axis / (jnp.linalg.norm(axis, axis=-1, keepdims=True) + 1.0e-8)
    half = angle * 0.5
    sin_half = jnp.sin(half)[..., None]
    cos_half = jnp.cos(half)[..., None]
    xyz = axis * sin_half
    return jnp.concatenate([xyz, cos_half], axis=-1)


def _quat_to_tan_norm(q):
    ref_tan = jnp.zeros_like(q[..., :3])
    ref_tan = ref_tan.at[..., 0].set(1.0)
    tan = _quat_rotate(q, ref_tan)

    ref_norm = jnp.zeros_like(q[..., :3])
    ref_norm = ref_norm.at[..., 2].set(1.0)
    norm = _quat_rotate(q, ref_norm)
    return jnp.concatenate([tan, norm], axis=-1)


def _calc_heading(q):
    ref_dir = jnp.zeros_like(q[..., :3])
    ref_dir = ref_dir.at[..., 0].set(1.0)
    rot_dir = _quat_rotate(q, ref_dir)
    return jnp.arctan2(rot_dir[..., 1], rot_dir[..., 0])


def _calc_heading_quat_inv(q):
    heading = _calc_heading(q)
    axis = jnp.zeros_like(q[..., :3])
    axis = axis.at[..., 2].set(1.0)
    return _axis_angle_to_quat(axis, -heading)


def _build_disc_obs_spec(env, key_body_names: Optional[Sequence[str]],
                         key_site_names: Optional[Sequence[str]]) -> DiscObsSpec:
    model = env.unwrapped()._model
    root_qpos_idx = np.array(mj_jntname2qposid("root", model), dtype=np.int32)
    root_qvel_idx = np.array(mj_jntname2qvelid("root", model), dtype=np.int32)

    joint_qpos_idx = []
    joint_qvel_idx = []
    joint_axis = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("AMP disc obs expects hinge joints only.")
        joint_qpos_idx.append(int(model.jnt_qposadr[j]))
        joint_qvel_idx.append(int(model.jnt_dofadr[j]))
        joint_axis.append(np.array(model.jnt_axis[j], dtype=np.float32))

    key_body_idx = []
    if key_body_names:
        for name in key_body_names:
            key_body_idx.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))

    key_site_idx = []
    if key_site_names:
        for name in key_site_names:
            key_site_idx.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))

    return DiscObsSpec(
        root_qpos_idx=root_qpos_idx,
        root_qvel_idx=root_qvel_idx,
        joint_qpos_idx=np.array(joint_qpos_idx, dtype=np.int32),
        joint_qvel_idx=np.array(joint_qvel_idx, dtype=np.int32),
        joint_axis=np.array(joint_axis, dtype=np.float32),
        key_body_idx=np.array(key_body_idx, dtype=np.int32),
        key_site_idx=np.array(key_site_idx, dtype=np.int32),
    )


def _extract_disc_inputs_from_data(data, spec: DiscObsSpec):
    qpos = data.qpos
    qvel = data.qvel
    root_pos = qpos[..., spec.root_qpos_idx[:3]]
    root_rot = _quat_wxyz_to_xyzw(qpos[..., spec.root_qpos_idx[3:7]])
    root_vel = qvel[..., spec.root_qvel_idx[:3]]
    root_ang_vel = qvel[..., spec.root_qvel_idx[3:6]]

    joint_angles = qpos[..., spec.joint_qpos_idx]
    joint_rot = _axis_angle_to_quat(jnp.asarray(spec.joint_axis), joint_angles)
    dof_vel = qvel[..., spec.joint_qvel_idx]

    key_pos_parts = []
    if spec.key_body_idx.size > 0 and data.xpos.size:
        key_pos_parts.append(data.xpos[..., spec.key_body_idx, :])
    if spec.key_site_idx.size > 0 and data.site_xpos.size:
        key_pos_parts.append(data.site_xpos[..., spec.key_site_idx, :])
    if key_pos_parts:
        key_pos = jnp.concatenate(key_pos_parts, axis=-2)
    else:
        key_pos = jnp.zeros((qpos.shape[0], 0, 3), dtype=jnp.float32)
    return root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos


def _init_disc_history(state, spec: DiscObsSpec, history_len: int):
    root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos = _extract_disc_inputs_from_data(
        state.data, spec)
    def _tile(x):
        return jnp.tile(x[:, None, ...], (1, history_len) + (1,) * (x.ndim - 1))
    return DiscHistory(
        root_pos=_tile(root_pos),
        root_rot=_tile(root_rot),
        root_vel=_tile(root_vel),
        root_ang_vel=_tile(root_ang_vel),
        joint_rot=_tile(joint_rot),
        dof_vel=_tile(dof_vel),
        key_pos=_tile(key_pos),
    )


def _push_disc_history(hist: DiscHistory, state, spec: DiscObsSpec):
    root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos = _extract_disc_inputs_from_data(
        state.data, spec)
    def _shift(buf, new):
        return jnp.concatenate([buf[:, 1:], new[:, None, ...]], axis=1)
    return DiscHistory(
        root_pos=_shift(hist.root_pos, root_pos),
        root_rot=_shift(hist.root_rot, root_rot),
        root_vel=_shift(hist.root_vel, root_vel),
        root_ang_vel=_shift(hist.root_ang_vel, root_ang_vel),
        joint_rot=_shift(hist.joint_rot, joint_rot),
        dof_vel=_shift(hist.dof_vel, dof_vel),
        key_pos=_shift(hist.key_pos, key_pos),
    )


def _reset_disc_history(hist: DiscHistory, state, done, spec: DiscObsSpec):
    root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos = _extract_disc_inputs_from_data(
        state.data, spec)
    done = done.astype(bool)
    def _reset(buf, new):
        reset = jnp.tile(new[:, None, ...], (1, buf.shape[1]) + (1,) * (new.ndim - 1))
        mask = done[:, None]
        while mask.ndim < buf.ndim:
            mask = mask[..., None]
        return jnp.where(mask, reset, buf)
    return DiscHistory(
        root_pos=_reset(hist.root_pos, root_pos),
        root_rot=_reset(hist.root_rot, root_rot),
        root_vel=_reset(hist.root_vel, root_vel),
        root_ang_vel=_reset(hist.root_ang_vel, root_ang_vel),
        joint_rot=_reset(hist.joint_rot, joint_rot),
        dof_vel=_reset(hist.dof_vel, dof_vel),
        key_pos=_reset(hist.key_pos, key_pos),
    )


def _compute_disc_obs(hist: DiscHistory, global_obs: bool, root_height_obs: bool):
    root_pos = hist.root_pos
    root_rot = hist.root_rot
    root_vel = hist.root_vel
    root_ang_vel = hist.root_ang_vel
    joint_rot = hist.joint_rot
    dof_vel = hist.dof_vel
    key_pos = hist.key_pos

    ref_root_pos = root_pos[..., -1, :]
    ref_root_rot = root_rot[..., -1, :]

    root_pos_obs = root_pos - ref_root_pos[:, None, :]
    if key_pos.shape[-2] > 0:
        key_pos = key_pos - root_pos[:, :, None, :]

    if not global_obs:
        heading_inv = _calc_heading_quat_inv(ref_root_rot)
        heading_inv_expand = jnp.repeat(heading_inv[:, None, :], root_pos.shape[1], axis=1)
        root_pos_obs = _quat_rotate(heading_inv_expand, root_pos_obs)
        root_rot = _quat_mul(heading_inv_expand, root_rot)

        if key_pos.shape[-2] > 0:
            heading_key = heading_inv_expand[:, :, None, :]
            key_pos = _quat_rotate(heading_key, key_pos)

    if root_height_obs:
        root_pos_obs = root_pos_obs.at[..., 2].set(root_pos[..., 2])
    else:
        root_pos_obs = root_pos_obs[..., :2]

    root_rot_flat = root_rot.reshape((-1, root_rot.shape[-1]))
    root_rot_obs = _quat_to_tan_norm(root_rot_flat).reshape((root_rot.shape[0], root_rot.shape[1], -1))

    joint_rot_flat = joint_rot.reshape((-1, joint_rot.shape[-1]))
    joint_rot_obs = _quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = joint_rot_obs.reshape((joint_rot.shape[0], joint_rot.shape[1], joint_rot.shape[2], -1))
    joint_rot_obs = joint_rot_obs.reshape((joint_rot.shape[0], joint_rot.shape[1], -1))

    pos_parts = [root_pos_obs, root_rot_obs, joint_rot_obs]
    if key_pos.shape[-2] > 0:
        key_pos_flat = key_pos.reshape((key_pos.shape[0], key_pos.shape[1], -1))
        pos_parts.append(key_pos_flat)
    pos_obs = jnp.concatenate(pos_parts, axis=-1)

    if not global_obs:
        heading_inv = _calc_heading_quat_inv(ref_root_rot)
        heading_inv_expand = jnp.repeat(heading_inv[:, None, :], root_vel.shape[1], axis=1)
        root_vel_obs = _quat_rotate(heading_inv_expand, root_vel)
        root_ang_vel_obs = _quat_rotate(heading_inv_expand, root_ang_vel)
    else:
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel

    vel_obs = jnp.concatenate([root_vel_obs, root_ang_vel_obs, dof_vel], axis=-1)
    disc_obs = jnp.concatenate([pos_obs, vel_obs], axis=-1)
    return disc_obs.reshape((disc_obs.shape[0], -1))


def _init_obs_history(obs: jnp.ndarray, history_len: int):
    obs = obs.astype(jnp.float32)
    buf = jnp.tile(obs[:, None, :], (1, history_len, 1))
    return ObsHistory(obs=buf)


def _push_obs_history(hist: ObsHistory, obs: jnp.ndarray):
    obs = obs.astype(jnp.float32)
    buf = jnp.concatenate([hist.obs[:, 1:], obs[:, None, :]], axis=1)
    return ObsHistory(obs=buf)


def _reset_obs_history(hist: ObsHistory, obs: jnp.ndarray, done: jnp.ndarray):
    obs = obs.astype(jnp.float32)
    reset = jnp.tile(obs[:, None, :], (1, hist.obs.shape[1], 1))
    mask = done[:, None, None]
    buf = jnp.where(mask, reset, hist.obs)
    return ObsHistory(obs=buf)


def _flatten_obs_history(hist: ObsHistory):
    return hist.obs.reshape((hist.obs.shape[0], -1))


def _gaussian_log_prob(mean, log_std, action):
    var = jnp.exp(2.0 * log_std)
    logp = -0.5 * jnp.sum(jnp.square(action - mean) / var, axis=-1)
    logp += -0.5 * mean.shape[-1] * jnp.log(2.0 * jnp.pi) - jnp.sum(log_std, axis=-1)
    return logp


def _gaussian_entropy(log_std):
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)


def _print_dataset_params(params: Dict[str, Any]):
    def _format_value(val):
        if val is None:
            return "None"
        if isinstance(val, dict) and "traj" in val:
            return "in-memory Trajectory"
        return str(val)

    print("[AMP PPO] dataset params:",
          "default=", _format_value(params.get("default_dataset_conf")),
          "amass=", _format_value(params.get("amass_dataset_conf")),
          "lafan1=", _format_value(params.get("lafan1_dataset_conf")),
          "custom=", _format_value(params.get("custom_dataset_conf")))


def _print_root_quat_stats(env):
    try:
        traj = env.th.traj
        root_name = env.root_free_joint_xml_name
        if root_name not in traj.info.joint_name2ind_qpos:
            print(f"[AMP PPO] root joint '{root_name}' not found in trajectory.")
            return
        root_ind = traj.info.joint_name2ind_qpos[root_name]
        root_quats = np.asarray(traj.data.qpos)[:, root_ind[3:7]]
        norms = np.linalg.norm(root_quats, axis=1)
        zero_cnt = int(np.sum(norms == 0.0))
        print(f"[AMP PPO] root_quat min_norm={norms.min():.6f} zero_cnt={zero_cnt}")
    except Exception as exc:
        print(f"[AMP PPO] root_quat stats failed: {exc}")


def _build_env(config):
    factory = TaskFactory.get_factory_cls(config.env.task_factory.name)
    params = OmegaConf.to_container(config.env.task_factory.params, resolve=True) or {}
    custom_path = params.pop("custom_traj_path", None) or params.pop("custom_dataset_path", None)
    if custom_path:
        # Load a fixed local trajectory to avoid HF downloads and caching issues.
        try:
            stat = os.stat(custom_path)
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
            print(f"[AMP PPO] Using custom_traj_path: {custom_path} (size={stat.st_size} mtime={mtime})")
        except FileNotFoundError:
            print(f"[AMP PPO] custom_traj_path not found: {custom_path}")
        except Exception as exc:
            print(f"[AMP PPO] custom_traj_path stat failed: {exc}")
        params["custom_dataset_conf"] = {"traj": Trajectory.load(custom_path)}
        params["default_dataset_conf"] = None
        params["amass_dataset_conf"] = None
        params["lafan1_dataset_conf"] = None
    else:
        print("[AMP PPO] No custom_traj_path set; using dataset confs as-is.")
    _print_dataset_params(params)
    try:
        env = factory.make(**config.env.env_params, **params)
    except Exception as exc:
        print(f"[AMP PPO] Env init failed: {exc}")
        debug_params = dict(params)
        debug_params["terminal_state_type"] = "NoTerminalStateHandler"
        try:
            debug_env = factory.make(**config.env.env_params, **debug_params)
            _print_root_quat_stats(debug_env)
        except Exception as debug_exc:
            print(f"[AMP PPO] Debug env init failed: {debug_exc}")
        raise
    env = LogWrapper(env)
    env = VecEnv(env)
    return env


def _init_models(rng, obs_dim: int, action_dim: int, disc_obs_dim: int, config):
    actor = MLP(config.model.actor_hidden_layers, action_dim,
                activation=config.model.activation, out_init_scale=config.model.actor_init_scale)
    critic = MLP(config.model.critic_hidden_layers, 1,
                 activation=config.model.activation, out_init_scale=1.0)
    disc = MLP(config.model.disc_hidden_layers, 1,
               activation=config.model.activation, out_init_scale=config.model.disc_init_scale)

    rng, actor_key, critic_key, disc_key = jax.random.split(rng, 4)
    dummy_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    dummy_disc = jnp.zeros((1, disc_obs_dim), dtype=jnp.float32)
    actor_params = actor.init(actor_key, dummy_obs)["params"]
    critic_params = critic.init(critic_key, dummy_obs)["params"]
    disc_params = disc.init(disc_key, dummy_disc)["params"]

    params = {
        "actor": actor_params,
        "critic": critic_params,
        "disc": disc_params,
    }
    return params, actor, critic, disc, rng


def _build_optimizer(config):
    grad_clip = float(config.optim.grad_clip)
    lr = float(config.optim.learning_rate)
    weight_decay = float(config.optim.weight_decay)
    opt_type = str(config.optim.type).lower()
    if opt_type == "adam":
        opt = optax.adamw(lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        opt = optax.sgd(lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optim.type}")
    if grad_clip > 0:
        opt = optax.chain(optax.clip_by_global_norm(grad_clip), opt)
    return opt


def _compute_gae(rewards, values, dones, last_value, gamma, lam):
    def _step(carry, inputs):
        gae, next_value = carry
        reward, value, done = inputs
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * lam * (1.0 - done) * gae
        return (gae, value), gae

    inputs = (rewards[::-1], values[::-1], dones[::-1])
    (_, _), adv_rev = jax.lax.scan(_step, (jnp.zeros_like(last_value), last_value), inputs)
    adv = adv_rev[::-1]
    returns = adv + values
    return adv, returns


def _accumulate_episode_returns(rewards, dones, carry):
    def _step(ret_carry, inputs):
        reward, done = inputs
        ret_carry = ret_carry + reward
        ep_return = ret_carry * done
        ret_carry = ret_carry * (1.0 - done)
        return ret_carry, (ep_return, done)

    carry, (ep_returns, ep_dones) = jax.lax.scan(_step, carry, (rewards, dones))
    return carry, jnp.sum(ep_returns), jnp.sum(ep_dones)


def _bce_with_logits(logits, targets):
    return jnp.mean(jnp.maximum(logits, 0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits))))


def _disc_grad_penalty(disc_apply, params, obs):
    def _single_grad(x):
        return jax.grad(lambda y: disc_apply(params, y).squeeze(-1))(x)
    grads = jax.vmap(_single_grad)(obs)
    return jnp.mean(jnp.sum(jnp.square(grads), axis=-1))


def _loss_fn(params, actor, critic, disc, batch, disc_batch, obs_norm, disc_norm, action_mean, action_std, log_std,
             config):
    norm_obs = _normalize(obs_norm, batch["obs"])
    norm_actions = (batch["actions"] - action_mean) / action_std

    mean = actor.apply({"params": params["actor"]}, norm_obs)
    logp = _gaussian_log_prob(mean, log_std, norm_actions)
    ratio = jnp.exp(logp - batch["logp"])

    adv = batch["adv"]
    clip_ratio = config.ppo.clip_ratio
    actor_loss0 = ratio * adv
    actor_loss1 = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    actor_loss = -jnp.mean(jnp.minimum(actor_loss0, actor_loss1))

    value = critic.apply({"params": params["critic"]}, norm_obs).squeeze(-1)
    value_loss = jnp.mean(jnp.square(value - batch["returns"]))

    agent_obs = _normalize(disc_norm, disc_batch["agent"])
    replay_obs = _normalize(disc_norm, disc_batch["replay"])
    demo_obs = _normalize(disc_norm, disc_batch["demo"])

    disc_agent_obs = jnp.concatenate([agent_obs, replay_obs], axis=0)
    logits_agent = disc.apply({"params": params["disc"]}, disc_agent_obs).squeeze(-1)
    logits_demo = disc.apply({"params": params["disc"]}, demo_obs).squeeze(-1)

    disc_loss_agent = _bce_with_logits(logits_agent, jnp.zeros_like(logits_agent))
    disc_loss_demo = _bce_with_logits(logits_demo, jnp.ones_like(logits_demo))
    disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)
    disc_loss += config.disc.grad_penalty * _disc_grad_penalty(disc.apply, {"params": params["disc"]}, demo_obs)

    total_loss = actor_loss + value_loss + config.disc.loss_weight * disc_loss

    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_ratio).astype(jnp.float32))
    approx_kl = jnp.mean(batch["logp"] - logp)
    entropy = jnp.mean(_gaussian_entropy(log_std))
    disc_agent_acc = jnp.mean((logits_agent < 0).astype(jnp.float32))
    disc_demo_acc = jnp.mean((logits_demo > 0).astype(jnp.float32))

    metrics = {
        "actor_loss": actor_loss,
        "value_loss": value_loss,
        "disc_loss": disc_loss,
        "clip_frac": clip_frac,
        "approx_kl": approx_kl,
        "entropy": entropy,
        "disc_agent_acc": disc_agent_acc,
        "disc_demo_acc": disc_demo_acc,
    }
    return total_loss, metrics


def _update_params(train_state: TrainState, actor, critic, disc, batch, disc_batch, log_std, config, optimizer):
    num_samples = batch["obs"].shape[0]
    minibatch_size = int(config.ppo.minibatch_size)
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive.")
    minibatch_size = min(minibatch_size, num_samples)
    num_batches = max(1, num_samples // minibatch_size)

    def _minibatch_update(params, opt_state, idx):
        mb = {k: v[idx] for k, v in batch.items()}
        loss_fn = lambda p: _loss_fn(p, actor, critic, disc, mb, disc_batch,
                                     train_state.obs_norm, train_state.disc_norm,
                                     train_state.action_mean, train_state.action_std, log_std, config)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    def _epoch_update(carry, _):
        rng, params, opt_state = carry
        rng, perm_key = jax.random.split(rng)
        perm = jax.random.permutation(perm_key, num_samples)

        def _batch_loop(batch_idx, carry):
            params, opt_state, metrics_acc = carry
            start = batch_idx * minibatch_size
            idx = jax.lax.dynamic_slice(perm, (start,), (minibatch_size,))
            params, opt_state, metrics = _minibatch_update(params, opt_state, idx)
            metrics_acc = jax.tree_util.tree_map(lambda a, b: a + b, metrics_acc, metrics)
            return params, opt_state, metrics_acc

        metrics_acc = {k: jnp.zeros(()) for k in ["actor_loss", "value_loss", "disc_loss",
                                                  "clip_frac", "approx_kl", "entropy",
                                                  "disc_agent_acc", "disc_demo_acc"]}
        params, opt_state, metrics_acc = jax.lax.fori_loop(
            0, num_batches, _batch_loop, (params, opt_state, metrics_acc))
        return (rng, params, opt_state), metrics_acc

    (rng, params, opt_state), metrics_acc = jax.lax.scan(
        _epoch_update, (train_state.rng, train_state.params, train_state.opt_state),
        None, length=int(config.ppo.update_epochs))

    metrics_acc = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), metrics_acc)
    denom = float(int(config.ppo.update_epochs) * num_batches)
    metrics = {k: v / denom for k, v in metrics_acc.items()}
    return train_state.replace(params=params, opt_state=opt_state, rng=rng), metrics


def _rollout(env, train_state: TrainState, actor, critic, obs, state, obs_hist, disc_hist,
             spec: DiscObsSpec, log_std, config):
    def _step(carry, _):
        rng, state, obs, obs_hist, disc_hist = carry

        disc_hist = _push_disc_history(disc_hist, state, spec)
        disc_obs = _compute_disc_obs(disc_hist, config.disc_obs.global_obs, config.disc_obs.root_height_obs)

        obs_hist = _push_obs_history(obs_hist, obs)
        flat_obs = _flatten_obs_history(obs_hist)
        norm_obs = _normalize(train_state.obs_norm, flat_obs)

        rng, act_key = jax.random.split(rng)
        mean = actor.apply({"params": train_state.params["actor"]}, norm_obs)
        norm_action = mean + jax.random.normal(act_key, mean.shape) * jnp.exp(log_std)
        logp = _gaussian_log_prob(mean, log_std, norm_action)
        action = norm_action * train_state.action_std + train_state.action_mean

        value = critic.apply({"params": train_state.params["critic"]}, norm_obs).squeeze(-1)

        next_obs, reward, absorbing, done, info, next_state = env.step(state, action)
        carry_obs = next_state.observation

        obs_hist = _reset_obs_history(obs_hist, carry_obs, done)
        disc_hist = _reset_disc_history(disc_hist, next_state, done, spec)
        done_f = done.astype(jnp.float32)

        transition = {
            "obs": flat_obs,
            "actions": action,
            "logp": logp,
            "value": value,
            "reward": reward,
            "done": done_f,
            "ep_return": next_state.metrics.returned_episode_returns * done_f,
            "disc_obs": disc_obs,
        }
        return (rng, next_state, carry_obs, obs_hist, disc_hist), transition

    carry = (train_state.rng, state, obs, obs_hist, disc_hist)
    (rng, next_state, next_obs, obs_hist, disc_hist), traj = jax.lax.scan(
        _step, carry, None, length=int(config.ppo.steps_per_iter))
    train_state = train_state.replace(rng=rng)
    return train_state, next_state, next_obs, obs_hist, disc_hist, traj


def _compute_disc_reward(disc, params, disc_obs, disc_norm, reward_scale):
    norm_disc_obs = _normalize(disc_norm, disc_obs)
    logits = disc.apply({"params": params["disc"]}, norm_disc_obs).squeeze(-1)
    prob = jax.nn.sigmoid(logits)
    reward = -jnp.log(jnp.maximum(1.0 - prob, 1.0e-4))
    return reward * reward_scale


def _sample_expert_disc_obs(expert_disc_obs, rng: jnp.ndarray, n: int):
    idx = jax.random.randint(rng, (n,), 0, expert_disc_obs.shape[0])
    if isinstance(expert_disc_obs, np.ndarray):
        idx_host = _to_numpy(idx)
        return jnp.asarray(expert_disc_obs[idx_host])
    return expert_disc_obs[idx]


def _build_expert_disc_obs(traj: Trajectory, spec: DiscObsSpec, history_len: int,
                           global_obs: bool, root_height_obs: bool):
    data = traj.data
    qpos = np.asarray(data.qpos)
    qvel = np.asarray(data.qvel)
    xpos = np.asarray(data.xpos) if data.xpos.size else np.empty((qpos.shape[0], 0, 3))
    site_xpos = np.asarray(data.site_xpos) if data.site_xpos.size else np.empty((qpos.shape[0], 0, 3))
    key_body_idx = spec.key_body_idx if xpos.size else np.array([], dtype=np.int32)
    key_site_idx = spec.key_site_idx if site_xpos.size else np.array([], dtype=np.int32)

    joint_axis = spec.joint_axis
    joint_axis = joint_axis / (np.linalg.norm(joint_axis, axis=-1, keepdims=True) + 1.0e-8)

    def _axis_angle_to_quat_np(axis, angle):
        half = angle * 0.5
        sin_half = np.sin(half)[..., None]
        cos_half = np.cos(half)[..., None]
        return np.concatenate([axis * sin_half, cos_half], axis=-1)

    def _quat_rotate_np(q, v):
        q_xyz = q[..., :3]
        qw = q[..., 3:4]
        t = 2.0 * np.cross(q_xyz, v)
        return v + qw * t + np.cross(q_xyz, t)

    def _quat_mul_np(q, r):
        qx, qy, qz, qw = np.split(q, 4, axis=-1)
        rx, ry, rz, rw = np.split(r, 4, axis=-1)
        x = qw * rx + qx * rw + qy * rz - qz * ry
        y = qw * ry - qx * rz + qy * rw + qz * rx
        z = qw * rz + qx * ry - qy * rx + qz * rw
        w = qw * rw - qx * rx - qy * ry - qz * rz
        return np.concatenate([x, y, z, w], axis=-1)

    def _quat_to_tan_norm_np(q):
        ref_tan = np.zeros_like(q[..., :3])
        ref_tan[..., 0] = 1.0
        tan = _quat_rotate_np(q, ref_tan)
        ref_norm = np.zeros_like(q[..., :3])
        ref_norm[..., 2] = 1.0
        norm = _quat_rotate_np(q, ref_norm)
        return np.concatenate([tan, norm], axis=-1)

    def _calc_heading_quat_inv_np(q):
        ref_dir = np.zeros_like(q[..., :3])
        ref_dir[..., 0] = 1.0
        rot_dir = _quat_rotate_np(q, ref_dir)
        heading = np.arctan2(rot_dir[..., 1], rot_dir[..., 0])
        axis = np.zeros_like(q[..., :3])
        axis[..., 2] = 1.0
        return _axis_angle_to_quat_np(axis, -heading)

    def _compute_disc_obs_np(hist):
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos = hist
        ref_root_pos = root_pos[:, -1, :]
        ref_root_rot = root_rot[:, -1, :]
        root_pos_obs = root_pos - ref_root_pos[:, None, :]
        if key_pos.shape[-2] > 0:
            key_pos = key_pos - root_pos[:, :, None, :]
        if not global_obs:
            heading_inv = _calc_heading_quat_inv_np(ref_root_rot)
            heading_inv_expand = np.repeat(heading_inv[:, None, :], root_pos.shape[1], axis=1)
            root_pos_obs = _quat_rotate_np(heading_inv_expand, root_pos_obs)
            root_rot = _quat_mul_np(heading_inv_expand, root_rot)
            if key_pos.shape[-2] > 0:
                heading_key = heading_inv_expand[:, :, None, :]
                key_pos = _quat_rotate_np(heading_key, key_pos)
        if root_height_obs:
            root_pos_obs[..., 2] = root_pos[..., 2]
        else:
            root_pos_obs = root_pos_obs[..., :2]
        root_rot_obs = _quat_to_tan_norm_np(root_rot.reshape((-1, 4))).reshape(
            (root_rot.shape[0], root_rot.shape[1], -1))
        joint_rot_obs = _quat_to_tan_norm_np(joint_rot.reshape((-1, 4)))
        joint_rot_obs = joint_rot_obs.reshape((joint_rot.shape[0], joint_rot.shape[1], joint_rot.shape[2], -1))
        joint_rot_obs = joint_rot_obs.reshape((joint_rot.shape[0], joint_rot.shape[1], -1))

        pos_parts = [root_pos_obs, root_rot_obs, joint_rot_obs]
        if key_pos.shape[-2] > 0:
            key_pos_flat = key_pos.reshape((key_pos.shape[0], key_pos.shape[1], -1))
            pos_parts.append(key_pos_flat)
        pos_obs = np.concatenate(pos_parts, axis=-1)

        if not global_obs:
            heading_inv = _calc_heading_quat_inv_np(ref_root_rot)
            heading_inv_expand = np.repeat(heading_inv[:, None, :], root_vel.shape[1], axis=1)
            root_vel_obs = _quat_rotate_np(heading_inv_expand, root_vel)
            root_ang_vel_obs = _quat_rotate_np(heading_inv_expand, root_ang_vel)
        else:
            root_vel_obs = root_vel
            root_ang_vel_obs = root_ang_vel

        vel_obs = np.concatenate([root_vel_obs, root_ang_vel_obs, dof_vel], axis=-1)
        disc_obs = np.concatenate([pos_obs, vel_obs], axis=-1)
        return disc_obs.reshape((disc_obs.shape[0], -1))

    disc_obs = []
    history = None
    split_points = data.split_points.tolist() if data.split_points.size else [0, qpos.shape[0]]
    split_set = set(split_points)

    for idx in range(qpos.shape[0]):
        if idx in split_set or history is None:
            root_pos = qpos[idx:idx + 1, spec.root_qpos_idx[:3]]
            root_rot = qpos[idx:idx + 1, spec.root_qpos_idx[3:7]]
            root_rot = root_rot[..., [1, 2, 3, 0]]
            root_vel = qvel[idx:idx + 1, spec.root_qvel_idx[:3]]
            root_ang_vel = qvel[idx:idx + 1, spec.root_qvel_idx[3:6]]
            joint_angles = qpos[idx:idx + 1, spec.joint_qpos_idx]
            joint_rot = _axis_angle_to_quat_np(joint_axis, joint_angles)
            dof_vel = qvel[idx:idx + 1, spec.joint_qvel_idx]
            key_pos_parts = []
            if key_body_idx.size > 0:
                key_pos_parts.append(xpos[idx:idx + 1, key_body_idx, :])
            if key_site_idx.size > 0:
                key_pos_parts.append(site_xpos[idx:idx + 1, key_site_idx, :])
            if key_pos_parts:
                key_pos = np.concatenate(key_pos_parts, axis=-2)
            else:
                key_pos = np.zeros((1, 0, 3), dtype=np.float32)

            def _tile(x):
                return np.tile(x[:, None, ...], (1, history_len) + (1,) * (x.ndim - 1))

            history = (
                _tile(root_pos),
                _tile(root_rot),
                _tile(root_vel),
                _tile(root_ang_vel),
                _tile(joint_rot),
                _tile(dof_vel),
                _tile(key_pos),
            )
        else:
            root_pos = qpos[idx:idx + 1, spec.root_qpos_idx[:3]]
            root_rot = qpos[idx:idx + 1, spec.root_qpos_idx[3:7]]
            root_rot = root_rot[..., [1, 2, 3, 0]]
            root_vel = qvel[idx:idx + 1, spec.root_qvel_idx[:3]]
            root_ang_vel = qvel[idx:idx + 1, spec.root_qvel_idx[3:6]]
            joint_angles = qpos[idx:idx + 1, spec.joint_qpos_idx]
            joint_rot = _axis_angle_to_quat_np(joint_axis, joint_angles)
            dof_vel = qvel[idx:idx + 1, spec.joint_qvel_idx]
            key_pos_parts = []
            if key_body_idx.size > 0:
                key_pos_parts.append(xpos[idx:idx + 1, key_body_idx, :])
            if key_site_idx.size > 0:
                key_pos_parts.append(site_xpos[idx:idx + 1, key_site_idx, :])
            if key_pos_parts:
                key_pos = np.concatenate(key_pos_parts, axis=-2)
            else:
                key_pos = np.zeros((1, 0, 3), dtype=np.float32)

            def _shift(buf, new):
                return np.concatenate([buf[:, 1:], new[:, None, ...]], axis=1)

            history = (
                _shift(history[0], root_pos),
                _shift(history[1], root_rot),
                _shift(history[2], root_vel),
                _shift(history[3], root_ang_vel),
                _shift(history[4], joint_rot),
                _shift(history[5], dof_vel),
                _shift(history[6], key_pos),
            )

        disc_obs.append(_compute_disc_obs_np(history)[0])

    return np.stack(disc_obs, axis=0)


def train(config, results_dir: str, seed: int = 0):
    os.makedirs(results_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(results_dir, "amp_ppo_jax_conf.yaml"))

    env = _build_env(config)
    obs_dim = int(np.prod(env.info.observation_space.shape))
    action_dim = int(np.prod(env.info.action_space.shape))

    key_bodies = list(config.disc_obs.key_bodies) if config.disc_obs.key_bodies else []
    key_sites = list(config.disc_obs.key_sites) if config.disc_obs.key_sites else []
    if not key_sites and hasattr(env.unwrapped(), "sites_for_mimic"):
        key_sites = list(env.unwrapped().sites_for_mimic)

    traj = env.unwrapped().th.traj
    if key_bodies and not traj.data.xpos.size:
        print("[AMP PPO] Trajectory has no xpos; ignoring key_bodies for disc obs.")
        key_bodies = []
    if key_sites and not traj.data.site_xpos.size:
        print("[AMP PPO] Trajectory has no site_xpos; ignoring key_sites for disc obs.")
        key_sites = []

    spec = _build_disc_obs_spec(env, key_bodies, key_sites)

    num_envs = int(config.ppo.num_envs)
    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, num_envs)
    obs, state = env.reset(reset_keys)
    obs_hist_len = int(config.disc_obs.obs_history_len)
    disc_hist_len = int(config.disc_obs.history_len)
    obs_hist = _init_obs_history(obs, obs_hist_len)
    disc_hist = _init_disc_history(state, spec, disc_hist_len)
    disc_obs = _compute_disc_obs(disc_hist, config.disc_obs.global_obs, config.disc_obs.root_height_obs)
    disc_obs_dim = disc_obs.shape[-1]

    params, actor, critic, disc, rng = _init_models(rng, obs_dim * obs_hist_len,
                                                   action_dim, disc_obs_dim, config)
    optimizer = _build_optimizer(config)
    opt_state = optimizer.init(params)

    action_mean = jnp.asarray(0.5 * (env.info.action_space.high + env.info.action_space.low), dtype=jnp.float32)
    action_std = jnp.asarray(0.5 * (env.info.action_space.high - env.info.action_space.low), dtype=jnp.float32)
    log_std = jnp.log(jnp.ones((action_dim,), dtype=jnp.float32) * config.model.action_std)

    obs_norm = _init_normalizer((obs_dim * obs_hist_len,), config.normalizer.obs_clip,
                                config.normalizer.min_std)
    disc_norm = _init_normalizer((disc_obs_dim,), config.normalizer.disc_clip,
                                 config.normalizer.min_std)

    buffer_on_cpu = bool(getattr(config.disc, "buffer_on_cpu", True))
    demo_on_cpu = bool(getattr(config.disc, "demo_on_cpu", True))
    replay = _init_replay_buffer(int(config.disc.buffer_size), disc_obs_dim, buffer_on_cpu)
    train_state = TrainState(
        params=params,
        opt_state=opt_state,
        obs_norm=obs_norm,
        disc_norm=disc_norm,
        action_mean=action_mean,
        action_std=action_std,
        replay=replay,
        rng=rng,
    )

    expert_traj_path = os.path.join(results_dir, "expert_traj.npz")
    if os.path.exists(expert_traj_path):
        traj = Trajectory.load(expert_traj_path)
    else:
        traj = env.unwrapped().th.traj
        traj.save(expert_traj_path)

    expert_disc_obs_path = os.path.join(results_dir, "expert_disc_obs.npz")
    if os.path.exists(expert_disc_obs_path):
        expert_disc_obs = np.load(expert_disc_obs_path)["disc_obs"]
    else:
        expert_disc_obs = _build_expert_disc_obs(
            traj, spec, disc_hist_len, config.disc_obs.global_obs, config.disc_obs.root_height_obs)
        np.savez(expert_disc_obs_path, disc_obs=np.asarray(expert_disc_obs))
    if demo_on_cpu:
        expert_disc_obs = np.asarray(expert_disc_obs, dtype=np.float32)
    else:
        expert_disc_obs = jnp.asarray(expert_disc_obs, dtype=jnp.float32)

    total_timesteps = int(config.ppo.total_timesteps)
    steps_per_iter = int(config.ppo.steps_per_iter)
    num_iters = total_timesteps // (steps_per_iter * num_envs)

    log_interval = int(OmegaConf.select(config, "train.log_interval", default=10))
    jit_rollout = bool(OmegaConf.select(config, "train.jit_rollout", default=True))
    jit_update = bool(OmegaConf.select(config, "train.jit_update", default=True))
    block_until_ready = bool(OmegaConf.select(config, "train.block_until_ready", default=True))
    tb_enabled = bool(OmegaConf.select(config, "train.tensorboard", default=True))
    tb_subdir = str(OmegaConf.select(config, "train.tensorboard_dir", default="tensorboard"))
    save_best = bool(OmegaConf.select(config, "train.save_best", default=True))
    writer = None
    if tb_enabled:
        if _SummaryWriter is None:
            print("TensorBoard不可用：未安装tensorboardX或torch。将跳过日志写入。")
        else:
            writer = _SummaryWriter(os.path.join(results_dir, tb_subdir))

    if jit_rollout or jit_update:
        print("已开启JIT，首次迭代会编译，可能耗时较长。")

    def _rollout_impl(train_state, obs, state, obs_hist, disc_hist):
        return _rollout(env, train_state, actor, critic, obs, state, obs_hist, disc_hist, spec, log_std, config)

    def _update_impl(train_state, batch, disc_batch):
        return _update_params(train_state, actor, critic, disc, batch, disc_batch, log_std, config, optimizer)

    rollout_fn = jax.jit(_rollout_impl) if jit_rollout else _rollout_impl
    update_fn = jax.jit(_update_impl) if jit_update else _update_impl

    sample_count = 0
    best_episode_return = -np.inf
    episode_return_carry = jnp.zeros((num_envs,), dtype=jnp.float32)
    best_path = os.path.join(results_dir, "amp_ppo_jax_best.pkl")
    for iter_idx in range(num_iters):
        iter_start = time.perf_counter()
        train_state, state, obs, obs_hist, disc_hist, traj = rollout_fn(
            train_state, obs, state, obs_hist, disc_hist)

        disc_reward = _compute_disc_reward(
            disc, train_state.params, traj["disc_obs"], train_state.disc_norm, config.disc.reward_scale)
        total_reward = config.reward.task_weight * traj["reward"] + config.reward.disc_weight * disc_reward
        reward_mean = jnp.mean(total_reward)
        task_reward_mean = jnp.mean(traj["reward"])
        disc_reward_mean = jnp.mean(disc_reward)
        episode_return_carry, episode_return_sum, episode_count = _accumulate_episode_returns(
            total_reward, traj["done"], episode_return_carry)
        episode_return_mean = jnp.where(
            episode_count > 0, episode_return_sum / episode_count, jnp.nan)

        last_obs_hist = _push_obs_history(obs_hist, obs)
        last_obs_flat = _flatten_obs_history(last_obs_hist)
        last_value = critic.apply({"params": train_state.params["critic"]},
                                  _normalize(train_state.obs_norm, last_obs_flat)).squeeze(-1)

        adv, returns = _compute_gae(total_reward, traj["value"], traj["done"], last_value,
                                    config.ppo.discount, config.ppo.td_lambda)
        adv_mean = jnp.mean(adv)
        adv_std = jnp.std(adv) + 1.0e-5
        adv = (adv - adv_mean) / adv_std

        batch = {
            "obs": traj["obs"].reshape((-1, traj["obs"].shape[-1])),
            "actions": traj["actions"].reshape((-1, traj["actions"].shape[-1])),
            "logp": traj["logp"].reshape((-1,)),
            "adv": adv.reshape((-1,)),
            "returns": returns.reshape((-1,)),
        }

        disc_obs_flat = traj["disc_obs"].reshape((-1, traj["disc_obs"].shape[-1]))
        rng, agent_key, demo_key, replay_key, add_key = jax.random.split(train_state.rng, 5)
        train_state = train_state.replace(rng=rng)

        disc_batch_size = int(config.disc.batch_size)
        agent_samples = int(np.ceil(disc_batch_size / 2))
        agent_idx = jax.random.randint(agent_key, (agent_samples,), 0, disc_obs_flat.shape[0])
        agent_disc_obs = disc_obs_flat[agent_idx]
        replay_obs = _replay_sample(train_state.replay, replay_key, agent_samples)
        demo_obs = _sample_expert_disc_obs(expert_disc_obs, demo_key, disc_batch_size)

        disc_batch = {
            "agent": agent_disc_obs,
            "replay": replay_obs,
            "demo": demo_obs,
        }

        sample_count += steps_per_iter * num_envs
        if sample_count < config.normalizer.samples:
            train_state = train_state.replace(
                obs_norm=_update_normalizer(train_state.obs_norm, traj["obs"]),
                disc_norm=_update_normalizer(train_state.disc_norm,
                                             jnp.concatenate([disc_obs_flat, demo_obs], axis=0)),
            )

        replay_size = int(train_state.replay.size)
        if replay_size >= int(config.disc.buffer_size):
            replay_take = min(int(config.disc.replay_samples), disc_obs_flat.shape[0])
            replay_idx = jax.random.choice(add_key, disc_obs_flat.shape[0], (replay_take,), replace=False)
            replay_samples = disc_obs_flat[replay_idx]
        else:
            replay_samples = disc_obs_flat
        train_state = train_state.replace(replay=_replay_add(train_state.replay, replay_samples))

        pre_update_state = train_state
        train_state, metrics = update_fn(train_state, batch, disc_batch)

        if block_until_ready:
            jax.block_until_ready(metrics["actor_loss"])
        iter_time = time.perf_counter() - iter_start

        should_log = log_interval > 0 and (iter_idx % log_interval == 0 or iter_idx == num_iters - 1)
        need_reward_vals = save_best or should_log
        reward_mean_val = None
        episode_return_mean_val = None
        episode_count_val = None
        if need_reward_vals:
            reward_mean_val = float(jax.device_get(reward_mean))
            task_reward_mean_val = float(jax.device_get(task_reward_mean))
            disc_reward_mean_val = float(jax.device_get(disc_reward_mean))
            episode_return_mean_val = float(jax.device_get(episode_return_mean))
            episode_count_val = float(jax.device_get(episode_count))
            if save_best and episode_count_val > 0 and episode_return_mean_val > best_episode_return:
                best_episode_return = episode_return_mean_val
                payload = _build_checkpoint_payload(config, pre_update_state)
                _save_checkpoint(best_path, payload)
                if writer is not None:
                    writer.add_scalar("reward/best_episode_mean", best_episode_return, int(sample_count))
                print(
                    f"Saved best checkpoint: episode_mean={best_episode_return:.3f} "
                    f"episodes={int(episode_count_val)} path={best_path}"
                )

        if should_log:
            log_data = dict(jax.device_get(metrics))
            log_data.update({
                "reward_mean": reward_mean_val,
                "task_reward_mean": task_reward_mean_val,
                "disc_reward_mean": disc_reward_mean_val,
                "episode_return_mean": episode_return_mean_val,
                "episode_count": episode_count_val,
            })
            sps = (steps_per_iter * num_envs) / max(iter_time, 1.0e-6)
            ep_str = ""
            if episode_count_val and episode_count_val > 0:
                ep_str = f"ep_r={float(log_data['episode_return_mean']):.3f} "
            print(
                f"[{iter_idx + 1}/{num_iters}] "
                f"samples={sample_count} "
                f"reward={float(log_data['reward_mean']):.3f} "
                f"{ep_str}"
                f"disc_r={float(log_data['disc_reward_mean']):.3f} "
                f"actor={float(log_data['actor_loss']):.4f} "
                f"value={float(log_data['value_loss']):.4f} "
                f"disc={float(log_data['disc_loss']):.4f} "
                f"kl={float(log_data['approx_kl']):.4f} "
                f"clip={float(log_data['clip_frac']):.3f} "
                f"{sps:.0f} samples/s "
                f"{iter_time:.2f}s/iter"
            )
            if writer is not None:
                step = int(sample_count)
                writer.add_scalar("reward/total_mean", float(log_data["reward_mean"]), step)
                writer.add_scalar("reward/task_mean", float(log_data["task_reward_mean"]), step)
                writer.add_scalar("reward/disc_mean", float(log_data["disc_reward_mean"]), step)
                if episode_count_val and episode_count_val > 0:
                    writer.add_scalar("reward/episode_mean", float(log_data["episode_return_mean"]), step)
                    writer.add_scalar("reward/episode_count", float(log_data["episode_count"]), step)
                writer.add_scalar("loss/actor", float(log_data["actor_loss"]), step)
                writer.add_scalar("loss/value", float(log_data["value_loss"]), step)
                writer.add_scalar("loss/disc", float(log_data["disc_loss"]), step)
                writer.add_scalar("stats/approx_kl", float(log_data["approx_kl"]), step)
                writer.add_scalar("stats/clip_frac", float(log_data["clip_frac"]), step)
                writer.add_scalar("perf/samples_per_s", float(sps), step)
                writer.add_scalar("perf/iter_time_s", float(iter_time), step)
                writer.flush()

    if writer is not None:
        writer.close()

    save_path = os.path.join(results_dir, "amp_ppo_jax.pkl")
    payload = _build_checkpoint_payload(config, train_state)
    _save_checkpoint(save_path, payload)

    return save_path


def _build_checkpoint_payload(config, train_state: TrainState) -> Dict[str, Any]:
    return dict(
        config=OmegaConf.to_container(config, resolve=True),
        params=train_state.params,
        obs_norm=dict(
            mean=np.asarray(train_state.obs_norm.mean),
            mean_sq=np.asarray(train_state.obs_norm.mean_sq),
            std=np.asarray(train_state.obs_norm.std),
            count=float(train_state.obs_norm.count),
            clip=float(train_state.obs_norm.clip),
            min_std=float(train_state.obs_norm.min_std),
        ),
        disc_norm=dict(
            mean=np.asarray(train_state.disc_norm.mean),
            mean_sq=np.asarray(train_state.disc_norm.mean_sq),
            std=np.asarray(train_state.disc_norm.std),
            count=float(train_state.disc_norm.count),
            clip=float(train_state.disc_norm.clip),
            min_std=float(train_state.disc_norm.min_std),
        ),
        action_mean=np.asarray(train_state.action_mean),
        action_std=np.asarray(train_state.action_std),
    )


def _save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: str):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    config = OmegaConf.create(payload["config"])
    return config, payload
