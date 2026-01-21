#!/usr/bin/env python3
import argparse
import os
import numpy as np
import mujoco

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from loco_mujoco.trajectory import Trajectory


DEFAULT_JOINT_KEYS = [
    "q_hip_flexion_l", "q_knee_angle_l", "q_ankle_angle_l",
    "q_hip_flexion_r", "q_knee_angle_r", "q_ankle_angle_r",
]


def _flatten(arr):
    return np.asarray(arr).reshape(-1)


def _summary_stats(values):
    values = _flatten(values)
    if values.size == 0:
        return None
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return dict(
        count=int(values.size),
        mean=float(values.mean()),
        std=float(values.std()),
        min=float(values.min()),
        p10=float(p10),
        p50=float(p50),
        p90=float(p90),
        max=float(values.max()),
    )


def _print_stats(label, stats, unit=""):
    if stats is None:
        print(f"{label}: no data")
        return
    suffix = f" {unit}" if unit else ""
    print(f"{label}:")
    print(f"  count={stats['count']}")
    print(f"  mean={stats['mean']:.4f}{suffix} std={stats['std']:.4f}{suffix}")
    print(f"  min={stats['min']:.4f}{suffix} p10={stats['p10']:.4f}{suffix} "
          f"p50={stats['p50']:.4f}{suffix} p90={stats['p90']:.4f}{suffix} max={stats['max']:.4f}{suffix}")


def _find_root_joint(traj):
    jnt_type = np.array(traj.info.model.jnt_type)
    free_ids = np.where(jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))[0]
    if free_ids.size > 0:
        return traj.info.joint_names[int(free_ids[0])]
    for name in traj.info.joint_names:
        if "root" in name or "free" in name:
            return name
    raise ValueError("Could not find a free/root joint in trajectory.")


def _compute_joint_stats_from_keys(data, joint_keys):
    per_joint = {}
    stds = []
    ranges = []
    for key in joint_keys:
        if key not in data:
            continue
        angles = np.rad2deg(_flatten(data[key]))
        if angles.size == 0:
            continue
        per_joint[key] = _summary_stats(angles)
        stds.append(angles.std())
        ranges.append(angles.max() - angles.min())
    joint_std_mean = float(np.mean(stds)) if stds else None
    joint_range_mean = float(np.mean(ranges)) if ranges else None
    return per_joint, joint_std_mean, joint_range_mean


def _compute_joint_stats_from_traj(traj, joint_keys):
    qpos = np.array(traj.data.qpos)
    per_joint = {}
    stds = []
    ranges = []
    for key in joint_keys:
        name = key.replace("q_", "")
        if name not in traj.info.joint_name2ind_qpos:
            continue
        idx = np.array(traj.info.joint_name2ind_qpos[name]).reshape(-1)[0]
        angles = np.rad2deg(qpos[:, idx])
        per_joint[key] = _summary_stats(angles)
        stds.append(angles.std())
        ranges.append(angles.max() - angles.min())
    joint_std_mean = float(np.mean(stds)) if stds else None
    joint_range_mean = float(np.mean(ranges)) if ranges else None
    return per_joint, joint_std_mean, joint_range_mean


def _variability_score(speed_cv, lateral_std, joint_std_mean):
    if speed_cv is None or lateral_std is None or joint_std_mean is None:
        return None
    return float(speed_cv + (lateral_std / 0.1) + (joint_std_mean / 30.0))


def analyze_new_dataset(path, joint_keys):
    traj = Trajectory.load(path, backend=np)
    root_name = _find_root_joint(traj)
    root_qpos_idx = np.array(traj.info.joint_name2ind_qpos[root_name]).reshape(-1)
    root_qvel_idx = np.array(traj.info.joint_name2ind_qvel[root_name]).reshape(-1)

    qpos = np.array(traj.data.qpos)
    qvel = np.array(traj.data.qvel)
    split_points = np.array(traj.data.split_points).astype(int)

    x = qpos[:, root_qpos_idx[0]]
    y = qpos[:, root_qpos_idx[1]]
    z = qpos[:, root_qpos_idx[2]]
    vx = qvel[:, root_qvel_idx[0]]
    vy = qvel[:, root_qvel_idx[1]]
    vz = qvel[:, root_qvel_idx[2]]

    return dict(
        kind="new",
        frequency=float(traj.info.frequency),
        n_samples=len(qpos),
        n_traj=len(split_points) - 1,
        root_name=root_name,
        x=x, y=y, z=z,
        vx=vx, vy=vy, vz=vz,
        joint_stats=_compute_joint_stats_from_traj(traj, joint_keys),
    )


def analyze_old_dataset(path, joint_keys):
    data = np.load(path, allow_pickle=True)
    keys = {k: data[k] for k in data.keys()}

    split_points = keys.get("split_points", None)
    if split_points is None:
        first_key = next(iter(keys))
        split_points = np.array([0, len(_flatten(keys[first_key]))])
    split_points = np.array(split_points).astype(int)

    def _get(name):
        return _flatten(keys.get(name, np.array([])))

    x = _get("q_pelvis_tx")
    y = _get("q_pelvis_ty")
    z = _get("q_pelvis_tz")
    vx = _get("dq_pelvis_tx")
    vy = _get("dq_pelvis_ty")
    vz = _get("dq_pelvis_tz")

    frequency = None
    for k in ("frequency", "freq", "traj_frequency", "traj_freq", "traj_dt", "dt"):
        if k in keys:
            val = float(np.array(keys[k]).reshape(-1)[0])
            frequency = val if k != "traj_dt" and k != "dt" else (1.0 / val)
            break

    return dict(
        kind="old",
        frequency=frequency,
        n_samples=len(x) if x.size else len(vx),
        n_traj=len(split_points) - 1,
        root_name="pelvis",
        x=x, y=y, z=z,
        vx=vx, vy=vy, vz=vz,
        joint_stats=_compute_joint_stats_from_keys(keys, joint_keys),
    )


def print_report(label, info):
    print(f"\n=== {label} ===")
    print(f"Samples: {info['n_samples']}  Trajectories: {info['n_traj']}")
    if info["frequency"] is not None:
        dt = 1.0 / info["frequency"]
        print(f"Frequency: {info['frequency']:.3f} Hz (dt={dt:.6f}s)")
    else:
        print("Frequency: unknown")
    _print_stats("Root x velocity", _summary_stats(info["vx"]), "m/s")
    _print_stats("Root y velocity", _summary_stats(info["vy"]), "m/s")
    _print_stats("Root z velocity", _summary_stats(info["vz"]), "m/s")
    speed = np.sqrt(info["vx"] * info["vx"] + info["vy"] * info["vy"])
    _print_stats("Planar speed (xy)", _summary_stats(speed), "m/s")

    dx = float(info["x"][-1] - info["x"][0]) if info["x"].size else 0.0
    dy = float(info["y"][-1] - info["y"][0]) if info["y"].size else 0.0
    straight_ratio = abs(dy) / (abs(dx) + 1e-8) if info["x"].size else None
    if straight_ratio is not None:
        print(f"Overall displacement: dx={dx:.4f} m dy={dy:.4f} m straight_ratio={straight_ratio:.6f}")

    per_joint, joint_std_mean, joint_range_mean = info["joint_stats"]
    if per_joint:
        print("Joint angle variability (deg):")
        for key, stats in per_joint.items():
            print(f"  {key}: std={stats['std']:.2f} range={stats['max'] - stats['min']:.2f}")
    if joint_std_mean is not None:
        print(f"Mean joint std (deg): {joint_std_mean:.2f}")
    if joint_range_mean is not None:
        print(f"Mean joint range (deg): {joint_range_mean:.2f}")

    speed_stats = _summary_stats(info["vx"])
    speed_cv = (speed_stats["std"] / (abs(speed_stats["mean"]) + 1e-8)) if speed_stats else None
    lateral_std = _summary_stats(info["vy"])["std"] if _summary_stats(info["vy"]) else None
    score = _variability_score(speed_cv, lateral_std, joint_std_mean)
    if score is not None:
        print(f"Variability score (heuristic, higher => harder): {score:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare expert datasets (old 0.4.1 vs new 1.0.1) with simple variability metrics.")
    parser.add_argument("--old_path", required=True, help="Path to old (0.4.1) expert dataset npz.")
    parser.add_argument("--new_path", required=True, help="Path to new (1.0.1) expert dataset npz.")
    parser.add_argument(
        "--joint_keys",
        nargs="*",
        default=DEFAULT_JOINT_KEYS,
        help="Joint keys to compare (default: hip/knee/ankle).")
    args = parser.parse_args()

    if not os.path.isfile(args.old_path):
        raise FileNotFoundError(f"Old dataset not found: {args.old_path}")
    if not os.path.isfile(args.new_path):
        raise FileNotFoundError(f"New dataset not found: {args.new_path}")

    old_info = analyze_old_dataset(args.old_path, args.joint_keys)
    new_info = analyze_new_dataset(args.new_path, args.joint_keys)

    print_report("Old dataset (0.4.1)", old_info)
    print_report("New dataset (1.0.1)", new_info)


if __name__ == "__main__":
    main()


'''
python compare_expert_datasets.py \
    --old_path /home/robot/RL/loco-mujoco-0.4.1/loco_mujoco/datasets/humanoids/real/02-constspeed_reduced_humanoid.npz \
    --new_path /home/robot/.cache/huggingface/hub/datasets--robfiras--loco-mujoco-datasets/snapshots/75479db3e07a86a4477982ca8a706b8a45828974/DefaultDatasets/mocap/SkeletonTorque/walk.npz

运行结果：
=== Old dataset (0.4.1) ===
Samples: 449000  Trajectories: 1
Frequency: unknown
Root x velocity:
  count=449000
  mean=1.2815 m/s std=0.1072 m/s
  min=0.9250 m/s p10=1.1492 m/s p50=1.2698 m/s p90=1.4299 m/s max=1.7316 m/s
Root y velocity:
  count=449000
  mean=0.0001 m/s std=0.2738 m/s
  min=-0.4441 m/s p10=-0.3371 m/s p50=-0.0070 m/s p90=0.3559 m/s max=0.5073 m/s
Root z velocity:
  count=449000
  mean=-0.0000 m/s std=0.1043 m/s
  min=-0.2976 m/s p10=-0.1665 m/s p50=0.0124 m/s p90=0.1295 m/s max=0.2660 m/s
Planar speed (xy):
  count=449000
  mean=1.3100 m/s std=0.1120 m/s
  min=0.9270 m/s p10=1.1619 m/s p50=1.3076 m/s p90=1.4580 m/s max=1.7555 m/s
Overall displacement: dx=1150.8101 m dy=0.0672 m straight_ratio=0.000058
Joint angle variability (deg):
  q_hip_flexion_l: std=16.63 range=59.83
  q_knee_angle_l: std=18.56 range=75.55
  q_ankle_angle_l: std=10.78 range=59.22
  q_hip_flexion_r: std=16.80 range=55.97
  q_knee_angle_r: std=17.21 range=67.91
  q_ankle_angle_r: std=10.02 range=55.70
Mean joint std (deg): 15.00
Mean joint range (deg): 62.36
Variability score (heuristic, higher => harder): 3.321

=== New dataset (1.0.1) ===
Samples: 35200  Trajectories: 1
Frequency: 40.000 Hz (dt=0.025000s)
Root x velocity:
  count=35200
  mean=1.2814 m/s std=0.1084 m/s
  min=0.9474 m/s p10=1.1466 m/s p50=1.2700 m/s p90=1.4315 m/s max=1.6763 m/s
Root y velocity:
  count=35200
  mean=0.0000 m/s std=0.1053 m/s
  min=-0.2840 m/s p10=-0.1290 m/s p50=-0.0124 m/s p90=0.1673 m/s max=0.3071 m/s
Root z velocity:
  count=35200
  mean=0.0000 m/s std=0.2747 m/s
  min=-0.4406 m/s p10=-0.3375 m/s p50=-0.0089 m/s p90=0.3562 m/s max=0.5146 m/s
Planar speed (xy):
  count=35200
  mean=1.2854 m/s std=0.1116 m/s
  min=0.9474 m/s p10=1.1472 m/s p50=1.2720 m/s p90=1.4414 m/s max=1.6830 m/s
Overall displacement: dx=1127.7059 m dy=0.0331 m straight_ratio=0.000029
Joint angle variability (deg):
  q_hip_flexion_l: std=16.64 range=59.80
  q_knee_angle_l: std=18.56 range=75.00
  q_ankle_angle_l: std=10.77 range=58.97
  q_hip_flexion_r: std=16.80 range=55.61
  q_knee_angle_r: std=17.21 range=67.68
  q_ankle_angle_r: std=10.02 range=55.60
Mean joint std (deg): 15.00
Mean joint range (deg): 62.11
Variability score (heuristic, higher => harder): 1.638

分析：
新旧版专家数据很可能是同一段走路轨迹,只是采样频率不同。理由：
  - 速度分布几乎一致：旧版 mean≈1.2815, 新版 mean≈1.2814; 关节角波动范围也非常接近。
  - 新版频率明确是 40Hz、样本 35200(总时长 880s)。旧版样本 449000,如果按 500Hz 估算，总时长约 898s,比新版略长。
  - 样本比 449000/35200≈12.76,接近 500/40=12.5,但不完全一致，说明很可能是同源轨迹、经过不同裁剪/重定向/平滑/重采样后的版本。
'''