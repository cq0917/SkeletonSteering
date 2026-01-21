from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = True

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
        n_eval_episodes=2,
        n_steps_per_fit=1000,
        use_cuda=USE_CUDA,
        env_id="SkeletonTorque.walk.mocap",)

    reward_ratios = [0.1]

    for reward_ratio in reward_ratios:
        launcher.add_experiment(reward_ratio__=reward_ratio, **default_params)

    launcher.run(LOCAL, TEST)



'''
训练：
python launcher.py
tensorboard --logdir training/logs

评估:
export MUJOCO_GL=glfw
python eval.py --record --print_joint_stats --model_path logs/1.0.1_01_21/reward_ratio___0.1/0/agent_best.msh
--compare_expert  用于加载专家速度用于对比
'''
