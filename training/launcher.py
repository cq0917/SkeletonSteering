"""
This script launches experiments for training agents with different parameters.
"""
from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    N_SEEDS = 1

    launcher = Launcher(
        exp_name='single_speed_reward_ratio',
        exp_file='experiment',
        n_seeds=N_SEEDS,
        n_exps_in_parallel=1,
        use_timestamp=True,)

    default_params = dict(
        n_epochs=4000,
        n_steps_per_epoch=5000,
        n_epochs_save=100,
        n_eval_episodes=5,
        n_steps_per_fit=1000,
        use_cuda=USE_CUDA,
        env_id="HumanoidTorque.walk.real",)

    reward_ratios = [0.3]

    for reward_ratio in reward_ratios:
        launcher.add_experiment(reward_ratio__=reward_ratio, **default_params)

    launcher.run(LOCAL, TEST)



'''
训练：
python launcher.py
tensorboard --logdir training/logs

评估:
export MUJOCO_GL=glfw
python eval.py --model_path logs/single_speed_reward_ratio_2026-01-19_21-24-15/reward_ratio___0.3/0/agent_epoch_79_J_64.528577.msh
'''

