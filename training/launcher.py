import os

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = True

    N_SEEDS = 1

    base_dir = os.path.join(os.path.dirname(__file__), "logs")
    launcher = Launcher(
        exp_name='amp_ppo_single_speed',
        exp_file='experiment',
        n_seeds=N_SEEDS,
        n_exps_in_parallel=1,
        use_timestamp=True,
        base_dir=base_dir,)

    default_params = dict(
        total_timesteps=50000000,
        num_envs=256,
        num_steps=32,
        num_minibatches=16,
        update_epochs=4,
        disc_minibatch_size=2048,
        n_disc_epochs=20,
        lr=1.0e-4,
        disc_lr=1.0e-4,
        use_cuda=USE_CUDA,
        env_id="SkeletonTorque.walk.mocap",)

    proportion_env_rewards = [0.1]

    for proportion_env_reward in proportion_env_rewards:
        launcher.add_experiment(proportion_env_reward__=proportion_env_reward, **default_params)

    launcher.run(LOCAL, TEST)



'''
训练：
python training/train_amp_ppo_jax.py --seed 42 \
    --config training/amp_ppo_jax_conf.yaml --env_id SkeletonTorque.walk.mocap

tensorboard --logdir 'training/logs'



评估 (AMP 保存为 .pkl):
export MUJOCO_GL=glfw
python training/eval_amp_ppo_jax.py \
    --record --n_steps 2000 --deterministic \
    --model_path training/logs/amp_ppo_jax_2026-01-25_02-05-45/amp_ppo_jax_best.pkl
'''
