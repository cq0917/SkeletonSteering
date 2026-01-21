#!/usr/bin/env python3
import argparse
import os
import numpy as np
import mujoco

from loco_mujoco.trajectory import Trajectory


def _find_root_joint(traj):
    jnt_type = np.array(traj.info.model.jnt_type)
    free_ids = np.where(jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))[0]
    if free_ids.size > 0:
        return traj.info.joint_names[int(free_ids[0])]
    for name in traj.info.joint_names:
        if "root" in name or "free" in name:
            return name
    raise ValueError("Could not find a free/root joint in trajectory.")


def _summary_stats(values):
    values = np.asarray(values)
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
    suffix = f" {unit}" if unit else ""
    print(f"{label}:")
    print(f"  count={stats['count']}")
    print(f"  mean={stats['mean']:.4f}{suffix} std={stats['std']:.4f}{suffix}")
    print(f"  min={stats['min']:.4f}{suffix} p10={stats['p10']:.4f}{suffix} "
          f"p50={stats['p50']:.4f}{suffix} p90={stats['p90']:.4f}{suffix} max={stats['max']:.4f}{suffix}")


def _segment_stats(vx, vy, x, y):
    speed = np.sqrt(vx * vx + vy * vy)
    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    mean_vx = float(vx.mean())
    cv_vx = float(vx.std() / (abs(mean_vx) + 1e-8))
    return dict(
        mean_vx=mean_vx,
        std_vx=float(vx.std()),
        mean_vy=float(vy.mean()),
        std_vy=float(vy.std()),
        mean_speed=float(speed.mean()),
        std_speed=float(speed.std()),
        dx=dx,
        dy=dy,
        straight_ratio=float(abs(dy) / (abs(dx) + 1e-8)),
        cv_vx=cv_vx,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LocoMuJoCo expert dataset for speed and straightness.")
    parser.add_argument(
        "--path",
        default=os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--robfiras--loco-mujoco-datasets/"
            "snapshots/75479db3e07a86a4477982ca8a706b8a45828974/"
            "DefaultDatasets/mocap/SkeletonTorque/walk.npz"
        ),
        help="Path to expert dataset npz file.")
    parser.add_argument(
        "--per_traj",
        action="store_true",
        help="Print per-trajectory stats (limited by --max_traj).")
    parser.add_argument(
        "--max_traj",
        type=int,
        default=5,
        help="Maximum number of trajectories to print when --per_traj is set.")
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        raise FileNotFoundError(f"Dataset not found: {args.path}")

    traj = Trajectory.load(args.path, backend=np)
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

    dt = 1.0 / float(traj.info.frequency)
    print(f"Trajectory frequency: {traj.info.frequency} Hz (dt={dt:.6f}s)")
    print(f"Total samples: {len(qpos)}")
    print(f"Number of trajectories: {len(split_points) - 1}")
    print(f"Root joint: {root_name}")

    _print_stats("Root x velocity", _summary_stats(vx), "m/s")
    _print_stats("Root y velocity", _summary_stats(vy), "m/s")
    _print_stats("Root z velocity", _summary_stats(vz), "m/s")

    speed = np.sqrt(vx * vx + vy * vy)
    _print_stats("Planar speed (xy)", _summary_stats(speed), "m/s")

    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    print("Overall displacement:")
    print(f"  dx={dx:.4f} m dy={dy:.4f} m straight_ratio={abs(dy)/(abs(dx)+1e-8):.6f}")

    if args.per_traj:
        max_traj = min(args.max_traj, len(split_points) - 1)
        for i in range(max_traj):
            start, end = split_points[i], split_points[i + 1]
            stats = _segment_stats(vx[start:end], vy[start:end], x[start:end], y[start:end])
            print(f"Traj {i}: mean_vx={stats['mean_vx']:.4f} std_vx={stats['std_vx']:.4f} "
                  f"cv_vx={stats['cv_vx']:.4f} mean_vy={stats['mean_vy']:.4f} std_vy={stats['std_vy']:.4f} "
                  f"dx={stats['dx']:.3f} dy={stats['dy']:.3f} straight_ratio={stats['straight_ratio']:.6f}")


if __name__ == "__main__":
    main()

'''
python analyze_expert_dataset.py --per_traj --path /home/robot/.cache
/huggingface/hub/datasets--robfiras--loco-mujoco-datasets/snapshots/75479db3e07a86a4477982ca8a706b8a45828974/DefaultDatasets/mocap/SkeletonTorque/walk.npz
'''