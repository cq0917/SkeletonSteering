import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from experiment_launcher import run_experiment
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
from mushroom_rl.core import Core
from mushroom_rl.core.logger.logger import Logger
from imitation_lib.utils import BestAgentSaver
from loco_mujoco import LocoEnv
from tqdm import tqdm

from utils import get_agent, compute_mean_speed

def experiment(reward_ratio: float = 0.3,
               env_id: str = "HumanoidTorque.walk.real",
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

    # logging
    sw = SummaryWriter(log_dir=results_dir)     # tensorboard
    logger = Logger(results_dir=results_dir, log_name="logging", seed=seed, append=True)    # numpy
    agent_saver = BestAgentSaver(save_path=results_dir, n_epochs_save=n_epochs_save)

    print(f"Starting training {env_id}...")
    # create environment, agent and core
    mdp = LocoEnv.make(env_id, headless=True)
    _ = mdp.reset()

    agent = get_agent(env_id, mdp, use_cuda, sw)
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
        agent_saver.save(core.agent, R_mean)

    agent_saver.save_curr_best_agent()
    print("Finished.")


if __name__ == "__main__":
    run_experiment(experiment)
