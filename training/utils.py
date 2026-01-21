from copy import deepcopy
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.core import Core, Agent
from mushroom_rl.utils.torch import to_float_tensor
from imitation_lib.imitation import VAIL_TRPO
from imitation_lib.utils import FullyConnectedNetwork, NormcInitializer, Standardizer, VariationalNet, VDBLoss
import mujoco
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

class MushroomEnvWrapper:
    def __init__(self, env):
        self._env = env

    @property
    def info(self):
        return self._env.info

    def reset(self, initial_state=None):
        # MushroomRL passes an initial_state; LocoMuJoCo reset expects a PRNG key.
        return self._env.reset()

    def step(self, action):
        obs, reward, absorbing, done, info = self._env.step(action)
        time_limit = False
        if done and not absorbing:
            try:
                time_limit = self._env._cur_step_in_episode >= self._env.info.horizon
            except Exception:
                time_limit = False
        terminal = bool(absorbing or (done and not time_limit))
        return obs, reward, terminal, info

    def render(self, record=False):
        if hasattr(self._env, "render"):
            try:
                return self._env.render(record)
            except TypeError:
                return self._env.render()
        return None

    def stop(self):
        if hasattr(self._env, "stop"):
            return self._env.stop()
        return None

    def __getattr__(self, name):
        return getattr(self._env, name)

def wrap_mdp_for_mushroom(env):
    return env if isinstance(env, MushroomEnvWrapper) else MushroomEnvWrapper(env)

def _get_obs_index(env, obs_name, entry_index=0):
    obs = env.obs_container[obs_name]
    obs_ind = np.array(obs.obs_ind).reshape(-1)
    return int(obs_ind[entry_index])

def _build_demonstrations(mdp):
    demos = mdp.create_dataset()
    if hasattr(demos, "to_np"):
        demos = demos.to_np()
    if isinstance(demos, dict):
        return demos
    if hasattr(demos, "observations"):
        demo_dict = dict(
            states=np.array(demos.observations),
            next_states=np.array(demos.next_observations),
            absorbing=np.array(demos.absorbings),
        )
        actions = getattr(demos, "actions", None)
        if actions is not None and np.size(actions) > 0:
            demo_dict["actions"] = np.array(actions)
        return demo_dict
    return demos

def get_agent(env_id, mdp, use_cuda, sw, conf_path=None):

    if conf_path is None:
        conf_path = 'confs.yaml'    # use default one

    with open(conf_path, 'r') as f:
        confs = yaml.safe_load(f)

    # get conf for environment
    try:
        # get the default conf (task agnostic)
        env_id_short = env_id.split('.')[0]
        conf = confs[env_id_short]
    except KeyError:
        # get the conf for the specific environment and task
        env_id_short = ".".join(env_id.split('.')[:2])
        conf = confs[env_id_short]

    if conf["algorithm"] == "VAIL":
        agent = create_vail_agent(mdp, sw, use_cuda, **conf["algorithm_config"])
    else:
        raise ValueError(f"Invalid algorithm: {conf['algorithm']}")

    return agent

def create_vail_agent(mdp, sw, use_cuda, std_0, info_constraint, lr_beta, z_dim, disc_only_states,
                      disc_use_next_states, train_disc_n_th_epoch, disc_batch_size, learning_rate_critic,
                      learning_rate_disc, policy_entr_coef, max_kl, n_epochs_cg, use_noisy_targets,
                      last_policy_activation, disc_exclude_root=False):
    mdp_info = deepcopy(mdp.info)
    expert_data = _build_demonstrations(mdp)

    trpo_standardizer = Standardizer(use_cuda=use_cuda)
    policy_params = dict(network=FullyConnectedNetwork,
                         input_shape=mdp_info.observation_space.shape,
                         output_shape=mdp_info.action_space.shape,
                         std_0=std_0,
                         n_features=[512, 256],
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         activations=['relu', 'relu', last_policy_activation],
                         standardizer=trpo_standardizer,
                         use_cuda=use_cuda)

    critic_params = dict(network=FullyConnectedNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': learning_rate_critic,
                                               'weight_decay': 0.0}},
                         loss=F.mse_loss,
                         batch_size=256,
                         input_shape=mdp_info.observation_space.shape,
                         activations=['relu', 'relu', 'identity'],
                         standardizer=trpo_standardizer,
                         squeeze_out=False,
                         output_shape=(1,),
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         n_features=[512, 256],
                         use_cuda=use_cuda)

    discrim_obs_mask = np.arange(mdp_info.observation_space.shape[0])
    if disc_exclude_root:
        root_indices = []
        for obs_name in ("q_root", "dq_root"):
            if hasattr(mdp, "obs_container") and obs_name in mdp.obs_container:
                obs_ind = np.array(mdp.obs_container[obs_name].obs_ind).reshape(-1)
                root_indices.extend(obs_ind.tolist())
        if root_indices:
            root_set = set(root_indices)
            discrim_obs_mask = np.array([i for i in discrim_obs_mask if i not in root_set], dtype=int)
    discrim_act_mask = [] if disc_only_states else np.arange(mdp_info.action_space.shape[0])
    discrim_input_shape = (len(discrim_obs_mask) + len(discrim_act_mask),) if not disc_use_next_states else \
        (2 * len(discrim_obs_mask) + len(discrim_act_mask),)
    discrim_standardizer = Standardizer(use_cuda=use_cuda)
    z_size = z_dim
    encoder_net = FullyConnectedNetwork(input_shape=discrim_input_shape, output_shape=(128,), n_features=[256],
                                        activations=['relu', 'relu'], standardizer=None,
                                        squeeze_out=False, use_cuda=use_cuda)
    decoder_net = FullyConnectedNetwork(input_shape=(z_size,), output_shape=(1,), n_features=[],
                                        # no features mean no hidden layer -> one layer
                                        activations=['identity'], standardizer=None,
                                        initializers=[NormcInitializer(std=0.1)],
                                        squeeze_out=False, use_cuda=use_cuda)

    discriminator_params = dict(optimizer={'class': optim.Adam,
                                           'params': {'lr': learning_rate_disc,
                                                      'weight_decay': 0.0}},
                                batch_size=disc_batch_size,
                                network=VariationalNet,
                                input_shape=discrim_input_shape,
                                output_shape=(1,),
                                z_size=z_size,
                                encoder_net=encoder_net,
                                decoder_net=decoder_net,
                                use_next_states=disc_use_next_states,
                                use_actions=not disc_only_states,
                                standardizer=discrim_standardizer,
                                use_cuda=use_cuda)

    alg_params = dict(train_D_n_th_epoch=train_disc_n_th_epoch,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask,
                      n_epochs_cg=n_epochs_cg,
                      trpo_standardizer=trpo_standardizer,
                      D_standardizer=discrim_standardizer,
                      loss=VDBLoss(info_constraint=info_constraint, lr_beta=lr_beta),
                      ent_coeff=policy_entr_coef,
                      use_noisy_targets=use_noisy_targets,
                      max_kl=max_kl,
                      use_next_states=disc_use_next_states)

    agent = VAIL_TRPO(mdp_info=mdp_info, policy_class=GaussianTorchPolicy, policy_params=policy_params, sw=sw,
                      discriminator_params=discriminator_params, critic_params=critic_params,
                      demonstrations=expert_data, **alg_params)
    return agent

def compute_mean_speed(env, dataset):
    x_vel_idx = _get_obs_index(env, "dq_root", entry_index=0)
    speeds = []
    for i in range(len(dataset)):
        speeds.append(dataset[i][0][x_vel_idx])
    return np.mean(speeds)

def _set_agent_deterministic(agent):
    policy = agent.policy
    if not hasattr(policy, "get_mean_and_chol"):
        return

    def _draw_action_det(state, _agent=agent, _policy=policy):
        if _agent.phi is not None:
            state = _agent.phi(state)
        with torch.no_grad():
            s = to_float_tensor(np.atleast_2d(state), _policy.use_cuda)
            mu, _ = _policy.get_mean_and_chol(s)
        return torch.squeeze(mu, dim=0).detach().cpu().numpy()

    agent.draw_action = _draw_action_det

def process_data(
        agent,
        env,
        n_trials=5,
        n_episodes=1,
        cycle_length_cutoff=60,
        record=False,
        is_wrapped=False,
        min_peak=True):
    core = Core(agent, env)
    datasets = []
    for _ in range(n_trials):
        dataset = core.evaluate(n_episodes=n_episodes, render=record, record=record)
        datasets.append(dataset)

    if is_wrapped:
        motor_indices = env.env._action_indices
        motor_names = [mujoco.mj_id2name(env.env._model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in motor_indices]
    else:
        motor_indices = env._action_indices
        motor_names = [mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in motor_indices]
    obs_keys = dict(
        q_hip_flexion_l=_get_obs_index(env, "q_hip_flexion_l"),
        q_knee_angle_l=_get_obs_index(env, "q_knee_angle_l"),
        q_ankle_angle_l=_get_obs_index(env, "q_ankle_angle_l"),
        dq_hip_flexion_l=_get_obs_index(env, "dq_hip_flexion_l"),
        dq_knee_angle_l=_get_obs_index(env, "dq_knee_angle_l"),
        dq_ankle_angle_l=_get_obs_index(env, "dq_ankle_angle_l"),
    )
    act_keys = dict(
        mot_hip_flexion_l=motor_names.index('mot_hip_flexion_l'),
        mot_knee_angle_l=motor_names.index('mot_knee_angle_l'),
        mot_ankle_angle_l=motor_names.index('mot_ankle_angle_l'),
    )

    datas = []
    for dataset in datasets:
        data = {}
        for i in range(len(dataset)):
            for key, value in obs_keys.items():
                if key not in data:
                    data[key] = []
                data[key].append(dataset[i][0][value])
            for key, value in act_keys.items():
                if key not in data:
                    data[key] = []
                data[key].append(dataset[i][1][value])
        for key in obs_keys.keys():
            data[key] = np.array(data[key])
        for key in act_keys.keys():
            data[key] = np.array(data[key])
        datas.append(data)
    cycless = []
    for data in datas:
        if min_peak:
            heel_strikes, _ = find_peaks(-1*data['q_hip_flexion_l'], height=0.2, distance=40)
        else:
            heel_strikes, _ = find_peaks(data['q_hip_flexion_l'], height=np.deg2rad(8),  distance=40)
        cycles = {}
        cycle_lengths = np.diff(heel_strikes)
        for i, cycle_length in enumerate(cycle_lengths):
            if cycle_length > cycle_length_cutoff:
                for key in data.keys():
                    if key not in cycles:
                        cycles[key] = []
                    cycles[key].append(data[key][heel_strikes[i]:heel_strikes[i+1]])
        effective_cycles = 0
        if cycles:
            sample_key = next(iter(cycles))
            effective_cycles = len(cycles[sample_key])
        print(f'Number of recorded cycle: {len(cycle_lengths)}')
        print(f'Number of effective cycle: {effective_cycles}')
        if effective_cycles == 0:
            raise ValueError(
                "No effective gait cycles detected. Try increasing n_episodes, "
                "lowering cycle_length_cutoff, or adjusting peak detection.")
        cycless.append(cycles)
    x_vel_idx = _get_obs_index(env, "dq_root", entry_index=0)
    speedss = []
    for dataset in datasets:
        speeds = []
        for i in range(len(dataset)):
            speeds.append(dataset[i][0][x_vel_idx])
        speeds = np.array(speeds)
        print(f'Average speed: {np.mean(speeds)}')
        speedss.append(speeds)
    interpolated_cycless = []
    mean_lengths = []
    for cycles in cycless:
        sample_key = next(iter(cycles))
        cycles_lengths = [len(cycle) for cycle in cycles[sample_key]]
        mean_length = round(np.mean(cycles_lengths))
        mean_lengths.append(mean_length)
        interpolated_cycles = {}
        for key, cycle in cycles.items():
            for i in range(len(cycle)):
                cycle_len = len(cycle[i])
                x = np.linspace(0, cycle_len-1, num=cycle_len)
                xnew = np.linspace(0, cycle_len-1, num=mean_length)
                y = np.array([cycle[i][j] for j in range(cycle_len)])
                spl = CubicSpline(x, y)
                if key not in interpolated_cycles:
                    interpolated_cycles[key] = []
                interpolated_cycles[key].append(np.array(spl(xnew)))
        for key in interpolated_cycles.keys():
            interpolated_cycles[key] = np.array(interpolated_cycles[key])
        interpolated_cycless.append(interpolated_cycles)

    return interpolated_cycless, mean_lengths, speedss

def interpolate_data(data, mean_length):
    """
    Interpolates the given data to a specified mean length.

    Args:
        data (dict): A dictionary containing the data to be interpolated. The keys are speeds,
                     and the values are dictionaries with joint names as keys and lists of cycles as values.
        mean_length (int): The target length for interpolation.

    Returns:
        dict: A dictionary with the same structure as the input data, but with interpolated cycles.
    """
    interpolated_data = dict()
    for speed in tqdm(data, desc='Interpolating Data'):
        interpolated_cycles = {}
        for joint in data[speed]:
            interpolated_cycles[joint] = []
            for joint_cycle in data[speed][joint]:
                x = np.linspace(0, len(joint_cycle)-1, num=len(joint_cycle))
                xnew = np.linspace(0, len(joint_cycle)-1, num=mean_length)
                spl = CubicSpline(x, joint_cycle)
                interpolated_cycles[joint].append(spl(xnew))
        for joint in interpolated_cycles:
            interpolated_cycles[joint] = np.array(interpolated_cycles[joint])
        interpolated_data[speed] = {
            'cycles': interpolated_cycles,
            'mean_length': mean_length,}

    return interpolated_data

def calculate_metrics(eval_data, ground_truth):
    """
    Calculates evaluation metrics such as RMSE and R2 score.

    Args:
        eval_data (dict): A dictionary containing the evaluation data.
        ground_truth (dict): A dictionary containing the ground truth data.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    metric = {}
    for i, eval_data_ in enumerate(eval_data):
        metric[i] = {}
        for speed in eval_data_:
            metric[i][speed] = {}
            RMSE, R2 = 0, 0
            for data_joint, eval_joint in zip(['q_hip_flexion_r', 'q_knee_angle_r', 'q_ankle_angle_r'], ['Hip', 'Knee', 'Ankle']):
                avg_data = np.mean(ground_truth[speed]['cycles'][data_joint], axis=0)
                avg_eval = np.mean(eval_data_[speed]['data'][eval_joint]['Angle (deg)'], axis=0)
                avg_eval_interpolated = np.interp(np.linspace(0, 1, num=len(avg_data)), np.linspace(0, 1, num=len(avg_eval)), avg_eval)
                RMSE += root_mean_squared_error(avg_data, avg_eval_interpolated)
                R2 += r2_score(avg_data, avg_eval_interpolated)
            metric[i][speed]['RMSE'] = RMSE/3
            metric[i][speed]['R2'] = R2/3
            metric[i][speed]['actual_speed'] = eval_data_[speed]['mean_speed']

    return metric

def eval_model(mdp, model_file, speed_range, mass, n_trials=5, n_episodes=1, cycle_length_cutoff=40, record=False,
               deterministic=False, print_joint_stats=False):
    agent = Agent.load(model_file)
    if deterministic:
        _set_agent_deterministic(agent)
    data_dict = {}
    use_speed_wrapper = hasattr(mdp, "set_operate_speed")
    for target_speed in speed_range:
        data_dict[target_speed] = {}
        if use_speed_wrapper:
            mdp.set_operate_speed(target_speed)
        _ = mdp.reset()
        data, mean_length, speeds = process_data(
            agent,
            mdp,
            n_trials=n_trials,
            n_episodes=n_episodes,
            cycle_length_cutoff=cycle_length_cutoff,
            record=record,
            is_wrapped=use_speed_wrapper)
        data_dict[target_speed]['data'] = data
        data_dict[target_speed]['mean_length'] = mean_length
        data_dict[target_speed]['speeds'] = speeds
        if print_joint_stats:
            for j in range(n_trials):
                hip = np.rad2deg(np.asarray(data[j]["q_hip_flexion_l"]))
                knee = np.rad2deg(np.asarray(data[j]["q_knee_angle_l"]))
                ankle = np.rad2deg(np.asarray(data[j]["q_ankle_angle_l"]))
                p10_h, p50_h, p90_h = np.percentile(hip, [10, 50, 90])
                p10_k, p50_k, p90_k = np.percentile(knee, [10, 50, 90])
                p10_a, p50_a, p90_a = np.percentile(ankle, [10, 50, 90])
                print(f"Policy joint angle stats (deg) trial {j} speed {target_speed}:")
                print(f"  q_hip_flexion_l: min={hip.min():.2f} p10={p10_h:.2f} p50={p50_h:.2f} p90={p90_h:.2f} max={hip.max():.2f}")
                print(f"  q_knee_angle_l: min={knee.min():.2f} p10={p10_k:.2f} p50={p50_k:.2f} p90={p90_k:.2f} max={knee.max():.2f}")
                print(f"  q_ankle_angle_l: min={ankle.min():.2f} p10={p10_a:.2f} p50={p50_a:.2f} p90={p90_a:.2f} max={ankle.max():.2f}")
    processed_datas = []
    for i in range(n_trials):
        processed_data = {}
        HIP_GEAR, KNEE_GEAR, ANKLE_GEAR = 275, 600, 500
        for speed in data_dict:
            processed_data[speed] = {
                'data': {
                    'Hip': {},
                    'Knee': {},
                    'Ankle': {}},
                'idxs': None,}
            processed_data[speed]['data']['Hip']['Angle (deg)'] = np.rad2deg(data_dict[speed]['data'][i]['q_hip_flexion_l'])
            processed_data[speed]['data']['Knee']['Angle (deg)'] = -1*np.rad2deg(data_dict[speed]['data'][i]['q_knee_angle_l'])
            processed_data[speed]['data']['Ankle']['Angle (deg)'] = np.rad2deg(data_dict[speed]['data'][i]['q_ankle_angle_l'])
            processed_data[speed]['data']['Hip']['Torque (Nm/kg)'] = -1*data_dict[speed]['data'][i]['mot_hip_flexion_l'] * HIP_GEAR / mass
            processed_data[speed]['data']['Knee']['Torque (Nm/kg)'] = data_dict[speed]['data'][i]['mot_knee_angle_l'] * KNEE_GEAR / mass
            processed_data[speed]['data']['Ankle']['Torque (Nm/kg)'] = -1*data_dict[speed]['data'][i]['mot_ankle_angle_l'] * ANKLE_GEAR / mass
            processed_data[speed]['data']['Hip']['Power (W/kg)'] = np.multiply(data_dict[speed]['data'][i]['dq_hip_flexion_l'], data_dict[speed]['data'][i]['mot_hip_flexion_l'] * HIP_GEAR) / mass
            processed_data[speed]['data']['Knee']['Power (W/kg)'] = np.multiply(data_dict[speed]['data'][i]['dq_knee_angle_l'], data_dict[speed]['data'][i]['mot_knee_angle_l'] * KNEE_GEAR) / mass
            processed_data[speed]['data']['Ankle']['Power (W/kg)'] = np.multiply(data_dict[speed]['data'][i]['dq_ankle_angle_l'], data_dict[speed]['data'][i]['mot_ankle_angle_l'] * ANKLE_GEAR) / mass
            processed_data[speed]['idxs'] = np.linspace(0, 1, data_dict[speed]['mean_length'][i])*100
            processed_data[speed]['mean_speed'] = round(np.mean(data_dict[speed]['speeds'][i]), 2)
        processed_datas.append(processed_data)
    return processed_datas
