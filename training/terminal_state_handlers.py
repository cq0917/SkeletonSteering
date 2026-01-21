import math
import numpy as np

from loco_mujoco.core.terminal_state_handler.base import TerminalStateHandler


class LegacyHumanoidFallTerminalStateHandler(TerminalStateHandler):
    """
    Replica of the 0.4.1 HumanoidTorque _has_fallen logic, adapted to SkeletonTorque observations.
    """

    def __init__(self, env, height_offset=0.0, **handler_config):
        super().__init__(env, **handler_config)
        q_root_idx = self._get_obs_indices(env, "q_root")
        self._root_height_idx = int(q_root_idx[0])
        self._root_quat_idx = tuple(int(i) for i in q_root_idx[1:5])
        self._lumbar_ext_idx = self._get_obs_index(env, "q_lumbar_extension")
        self._lumbar_bend_idx = self._get_obs_index(env, "q_lumbar_bending")
        self._lumbar_rot_idx = self._get_obs_index(env, "q_lumbar_rotation")

        self._height_offset = float(height_offset)
        self._height_min = -0.46
        self._height_max = 0.1
        self._baseline_height = None
        self._baseline_quat = None

    @staticmethod
    def _get_obs_indices(env, obs_name):
        obs = env.obs_container[obs_name]
        return np.array(obs.obs_ind).reshape(-1)

    @staticmethod
    def _get_obs_index(env, obs_name, entry_index=0):
        obs = env.obs_container[obs_name]
        obs_ind = np.array(obs.obs_ind).reshape(-1)
        return int(obs_ind[entry_index])

    @staticmethod
    def _normalize_quat(quat_sf):
        w, x, y, z = quat_sf
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm == 0.0:
            return 1.0, 0.0, 0.0, 0.0
        inv = 1.0 / norm
        return w * inv, x * inv, y * inv, z * inv

    @staticmethod
    def _quat_dot(q1, q2):
        return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]

    @staticmethod
    def _quat_conjugate(quat_sf):
        w, x, y, z = quat_sf
        return w, -x, -y, -z

    @staticmethod
    def _quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    @staticmethod
    def _quat_to_euler_zxy(quat_sf):
        w, x, y, z = quat_sf
        r12 = 2.0 * (x * y - w * z)
        r22 = 1.0 - 2.0 * (x * x + z * z)
        r31 = 2.0 * (x * z - w * y)
        r32 = 2.0 * (y * z + w * x)
        r33 = 1.0 - 2.0 * (x * x + y * y)

        r32 = max(-1.0, min(1.0, r32))
        beta = math.asin(r32)
        alpha = math.atan2(-r12, r22)
        gamma = math.atan2(-r31, r33)
        return alpha, beta, gamma

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def reset(self, env, model, data, carry, backend):
        self._baseline_height = None
        self._baseline_quat = None
        return data, carry

    def is_absorbing(self, env, obs, info, data, carry):
        height = float(obs[self._root_height_idx])
        qw = float(obs[self._root_quat_idx[0]])
        qx = float(obs[self._root_quat_idx[1]])
        qy = float(obs[self._root_quat_idx[2]])
        qz = float(obs[self._root_quat_idx[3]])
        quat_sf = self._normalize_quat((qw, qx, qy, qz))

        if self._baseline_height is None:
            self._baseline_height = height
            self._baseline_quat = quat_sf
            return False, carry

        if self._quat_dot(self._baseline_quat, quat_sf) < 0.0:
            quat_sf = (-quat_sf[0], -quat_sf[1], -quat_sf[2], -quat_sf[3])
        rel_quat = self._quat_multiply(self._quat_conjugate(self._baseline_quat), quat_sf)
        pelvis_tilt, pelvis_list, pelvis_rotation = self._quat_to_euler_zxy(rel_quat)

        height_delta = height - self._baseline_height + self._height_offset
        pelvis_tilt = self._wrap_to_pi(pelvis_tilt)
        pelvis_list = self._wrap_to_pi(pelvis_list)
        pelvis_rotation = self._wrap_to_pi(pelvis_rotation)

        pelvis_height_condition = (height_delta < self._height_min) or (height_delta > self._height_max)
        pelvis_tilt_condition = (pelvis_tilt < (-np.pi / 4.5)) or (pelvis_tilt > (np.pi / 12))
        pelvis_list_condition = (pelvis_list < -np.pi / 12) or (pelvis_list > np.pi / 8)
        pelvis_rotation_condition = (pelvis_rotation < (-np.pi / 9)) or (pelvis_rotation > (np.pi / 9))

        lumbar_extension = obs[self._lumbar_ext_idx]
        lumbar_bending = obs[self._lumbar_bend_idx]
        lumbar_rotation = obs[self._lumbar_rot_idx]

        lumbar_extension_condition = (lumbar_extension < (-np.pi / 4)) or (lumbar_extension > (np.pi / 10))
        lumbar_bending_condition = (lumbar_bending < -np.pi / 10) or (lumbar_bending > np.pi / 10)
        lumbar_rotation_condition = (lumbar_rotation < (-np.pi / 4.5)) or (lumbar_rotation > (np.pi / 4.5))

        pelvis_condition = (pelvis_height_condition or pelvis_tilt_condition
                            or pelvis_list_condition or pelvis_rotation_condition)
        lumbar_condition = (lumbar_extension_condition or lumbar_bending_condition or lumbar_rotation_condition)

        return bool(pelvis_condition or lumbar_condition), carry


LegacyHumanoidFallTerminalStateHandler.register()
