import os

import numpy as np
import torch
import yaml
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
from torch.utils.tensorboard import SummaryWriter
from experiment_launcher import run_experiment
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
from mushroom_rl.core import Core
from mushroom_rl.core.logger.logger import Logger
from loco_mujoco import ImitationFactory
from loco_mujoco.task_factories import DefaultDatasetConf
from tqdm import tqdm

from utils import get_agent, compute_mean_speed, wrap_mdp_for_mushroom

def _parse_env_id(env_id: str):
    parts = env_id.split(".")
    env_name = parts[0] if parts else env_id
    task = parts[1] if len(parts) > 1 else "walk"
    dataset_type = parts[2] if len(parts) > 2 else "mocap"
    return env_name, task, dataset_type

def _load_training_config(conf_path, env_id):
    try:
        with open(conf_path, 'r') as f:
            confs = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

    env_key = env_id.split('.')[0]
    conf = confs.get(env_key)
    if conf is None:
        env_key = ".".join(env_id.split('.')[:2])
        conf = confs.get(env_key, {})
    return conf.get("training_config", {})

def experiment(reward_ratio: float = 0.1,
               env_id: str = "SkeletonTorque.walk.mocap",
               n_epochs: int = 500,
               n_steps_per_epoch: int = 10000,
               n_steps_per_fit: int = 1024,
               n_eval_episodes: int = 50,
               n_epochs_save: int = 500,
               gamma: float = 0.99,
               results_dir: str = './logs',
               use_cuda: bool = False,
               seed: int = 0):
    os.environ['MUJOCO_GL'] = 'egl'
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    results_dir = os.path.join(results_dir, str(seed))
    conf_path = os.path.join(os.path.dirname(__file__), "confs.yaml")
    training_config = _load_training_config(conf_path, env_id)
    save_every_n_epochs = training_config.get("save_every_n_epochs", n_epochs_save)

    # logging
    sw = SummaryWriter(log_dir=results_dir)     # tensorboard
    logger = Logger(results_dir=results_dir, log_name="logging", seed=seed, append=True)    # numpy
    best_reward = -float("inf")
    best_agent_path = os.path.join(results_dir, "agent_best.msh")

    print(f"Starting training {env_id}...")
    # create environment, agent and core
    env_name, task, dataset_type = _parse_env_id(env_id)
    target_velocity = 1.25
    if task == "run":
        target_velocity = 2.5
    mdp = ImitationFactory.make(
        env_name,
        default_dataset_conf=DefaultDatasetConf(task=task, dataset_type=dataset_type),
        reward_type="TargetXVelocityReward",
        reward_params=dict(target_velocity=target_velocity),
        timestep=0.001,
        n_substeps=10,
        horizon=1000,
        use_box_feet=True,
        disable_arms=True,
        headless=True)
    mdp = wrap_mdp_for_mushroom(mdp)
    _ = mdp.reset()

    agent = get_agent(env_id, mdp, use_cuda, sw, conf_path=conf_path)
    agent._env_reward_frac = reward_ratio
    print(f'env_reward_frac = {agent._env_reward_frac}')
    core = Core(agent, mdp)
    if n_steps_per_fit is not None and hasattr(agent, "_demonstrations"):
        demos = agent._demonstrations
        if isinstance(demos, dict) and "states" in demos:
            demo_len = demos["states"].shape[0]
            if demo_len < n_steps_per_fit:
                print(f'Adjusting n_steps_per_fit from {n_steps_per_fit} to {demo_len} to match demo length')
                n_steps_per_fit = demo_len

    for epoch in tqdm(range(n_epochs)):
        # train
        core.learn(
            n_steps=n_steps_per_epoch,
            n_steps_per_fit=n_steps_per_fit,
            quiet=True,
            render=False)
        # evaluate
        dataset = core.evaluate(n_episodes=n_eval_episodes)
        R_mean = np.mean(compute_J(dataset))
        J_mean = np.mean(compute_J(dataset, gamma=gamma))
        L = np.mean(compute_episodes_length(dataset))
        S_mean = compute_mean_speed(mdp, dataset)
        logger.log_numpy(
            Epoch=epoch,
            R_mean=R_mean,
            J_mean=J_mean,
            L=L,
            S_mean=S_mean)
        sw.add_scalar("Eval_R-stochastic", R_mean, epoch)
        sw.add_scalar("Eval_J-stochastic", J_mean, epoch)
        sw.add_scalar("Eval_L-stochastic", L, epoch)
        sw.add_scalar("Eval_S-stochastic", S_mean, epoch)
        if R_mean > best_reward:
            best_reward = R_mean
            core.agent.save(best_agent_path, full_save=True)
        if save_every_n_epochs and save_every_n_epochs > 0 and (epoch + 1) % save_every_n_epochs == 0:
            snapshot_path = os.path.join(
                results_dir,
                f"agent_epoch_{epoch}_R_{R_mean:.6f}.msh")
            core.agent.save(snapshot_path, full_save=True)

    print("Finished.")


if __name__ == "__main__":
    run_experiment(experiment)
