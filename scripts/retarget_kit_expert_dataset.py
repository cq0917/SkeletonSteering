#!/usr/bin/env python3
import argparse
import os
import sys

from loco_mujoco.smpl.retargeting import (
    get_amass_dataset_path,
    get_converted_amass_dataset_path,
    get_smpl_model_path,
    load_retargeted_amass_trajectory,
)


def iter_npz_files(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".npz"):
                yield os.path.join(dirpath, name)


def collect_dataset_names(root):
    names = []
    for path in iter_npz_files(root):
        rel = os.path.relpath(path, root)
        rel = os.path.splitext(rel)[0]
        names.append(rel.replace(os.sep, "/"))
    return sorted(names)


def read_list_file(path):
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(".npz"):
                line = os.path.splitext(line)[0]
            names.append(line)
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Offline retarget KIT expert dataset and save a concatenated Trajectory."
    )
    parser.add_argument(
        "--env_name",
        default="MjxSkeletonTorque",
        help="LocoMuJoCo env name used for retargeting (match training env).",
    )
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Root folder of the (subsampled) KIT dataset. Defaults to LOCOMUJOCO_AMASS_PATH.",
    )
    parser.add_argument(
        "--list_file",
        default=None,
        help="Optional text file listing dataset paths (relative to dataset_root).",
    )
    parser.add_argument(
        "--output_traj",
        default="training/kit_expert_traj.npz",
        help="Output Trajectory path for training (custom_traj_path).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print dataset list and paths; do not retarget.",
    )
    parser.add_argument(
        "--ignore_path_mismatch",
        action="store_true",
        help="Proceed even if dataset_root differs from LOCOMUJOCO_AMASS_PATH.",
    )
    args = parser.parse_args()

    try:
        amass_root = get_amass_dataset_path()
        _ = get_converted_amass_dataset_path()
        _ = get_smpl_model_path()
    except AssertionError as exc:
        print(f"[retarget] Config missing: {exc}")
        print("Set paths via:")
        print("  loco-mujoco-set-amass-path --path <amass_root>")
        print("  loco-mujoco-set-conv-amass-path --path <converted_root>")
        print("  loco-mujoco-set-smpl-model-path --path <smpl_model_root>")
        return 2

    dataset_root = args.dataset_root or amass_root
    if not os.path.isdir(dataset_root):
        print(f"[retarget] dataset_root not found: {dataset_root}")
        return 2

    if (os.path.abspath(dataset_root) != os.path.abspath(amass_root)) and not args.ignore_path_mismatch:
        print("[retarget] dataset_root != LOCOMUJOCO_AMASS_PATH")
        print(f"  dataset_root: {dataset_root}")
        print(f"  AMASS path:   {amass_root}")
        print("Either set LOCOMUJOCO_AMASS_PATH to dataset_root or pass --ignore_path_mismatch.")
        return 2

    if args.list_file:
        dataset_names = read_list_file(args.list_file)
    else:
        dataset_names = collect_dataset_names(dataset_root)

    if not dataset_names:
        print("[retarget] No dataset files found.")
        return 2

    print(f"[retarget] env_name: {args.env_name}")
    print(f"[retarget] dataset_root: {dataset_root}")
    print(f"[retarget] clip count: {len(dataset_names)}")
    if args.dry_run:
        for name in dataset_names:
            print(name)
        return 0

    traj = load_retargeted_amass_trajectory(args.env_name, dataset_names)
    if args.output_traj:
        out_path = args.output_traj
        if not os.path.isabs(out_path):
            out_path = os.path.abspath(out_path)
        traj.save(out_path)
        print(f"[retarget] saved Trajectory: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())



'''
loco-mujoco-set-amass-path --path /home/robot/RL/SkeletonSteering/expert_dataset
loco-mujoco-set-conv-amass-path --path /home/robot/RL/amass_conv
loco-mujoco-set-smpl-model-path --path /home/robot/RL/smpl

JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES="" python scripts/retarget_kit_expert_dataset.py \
    --dataset_root /home/robot/RL/SkeletonSteering/expert_dataset \
    --env_name SkeletonTorque \
    --output_traj training/kit_expert_traj.npz

'''