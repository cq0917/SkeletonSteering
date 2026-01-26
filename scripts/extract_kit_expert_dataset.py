#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from collections import defaultdict

import numpy as np


DEFAULT_ACTIONS = [
    "leftturn",
    "rightturn",
    "walking_slow",
    "walking_medium",
    "walking_run",
    "walking_fast",
    "walkingstraightforwards",
    "walking",
    "walkingstraightforward",
    "run",
]


def iter_npz_files(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".npz"):
                yield os.path.join(dirpath, name)


def normalize_label(name):
    base = os.path.basename(name)
    if base.endswith("_poses.npz"):
        base = base[: -len("_poses.npz")]
    else:
        base = os.path.splitext(base)[0]
    base = base.replace("-", "_")
    base = re.sub(r"(_\d+)+$", "", base)
    base = re.sub(r"\d+$", "", base)
    base = base.strip("_-")
    base = re.sub(r"_+", "_", base)
    return base.lower()


def normalize_action(action):
    action = action.strip().replace(" ", "")
    return normalize_label(action)


def parse_actions(values):
    if not values:
        return {normalize_action(a) for a in DEFAULT_ACTIONS}
    if len(values) == 1 and "," in values[0]:
        values = [v.strip() for v in values[0].split(",") if v.strip()]
    return {normalize_action(v) for v in values}


def get_framerate(data):
    for key in ("mocap_framerate", "mocap_frame_rate", "fps"):
        if key in data:
            try:
                return float(data[key])
            except Exception:
                return None
    return None


def get_num_frames(data):
    for key in ("trans", "poses", "pose_aa"):
        if key in data:
            try:
                return int(data[key].shape[0])
            except Exception:
                return None
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract selected KIT actions into a local expert_dataset directory."
    )
    parser.add_argument(
        "--kit_root",
        default="/home/robot/RL/KIT",
        help="Path to KIT dataset root.",
    )
    parser.add_argument(
        "--output_dir",
        default="expert_dataset",
        help="Destination dataset directory (default: ./expert_dataset).",
    )
    parser.add_argument(
        "--actions",
        nargs="*",
        default=None,
        help="Action labels to include (space or comma separated).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report matches, do not copy files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output_dir.",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="Write manifest.json with file lists per action.",
    )
    args = parser.parse_args()

    kit_root = args.kit_root
    if not os.path.isdir(kit_root):
        raise SystemExit(f"KIT root not found: {kit_root}")

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)

    action_set = parse_actions(args.actions)

    matched = 0
    copied = 0
    skipped = 0
    bytes_total = 0
    counts = defaultdict(int)
    frames_by_action = defaultdict(int)
    seconds_by_action = defaultdict(float)
    missing_fps = 0
    missing_frames = 0
    read_errors = 0
    total_frames = 0
    total_seconds = 0.0
    manifest = defaultdict(list)

    for path in iter_npz_files(kit_root):
        label = normalize_label(path)
        if label not in action_set:
            continue
        matched += 1
        counts[label] += 1

        rel_path = os.path.relpath(path, kit_root)
        if args.manifest:
            manifest[label].append(rel_path)

        try:
            with np.load(path, allow_pickle=True) as data:
                framerate = get_framerate(data)
                n_frames = get_num_frames(data)
        except Exception:
            read_errors += 1
            framerate = None
            n_frames = None

        if framerate is None:
            missing_fps += 1
        elif n_frames is None:
            missing_frames += 1
        elif framerate > 0:
            frames_by_action[label] += n_frames
            seconds_by_action[label] += n_frames / framerate
            total_frames += n_frames
            total_seconds += n_frames / framerate

        if args.dry_run:
            continue

        dest_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        if os.path.exists(dest_path) and not args.overwrite:
            skipped += 1
            continue

        shutil.copy2(path, dest_path)
        copied += 1
        try:
            bytes_total += os.path.getsize(dest_path)
        except OSError:
            pass

    print(f"KIT root: {kit_root}")
    print(f"Output dir: {output_dir}")
    print(f"Actions: {', '.join(sorted(action_set))}")
    print(f"Matched files: {matched}")
    print(f"Copied files: {copied}")
    print(f"Skipped files: {skipped}")
    print(f"Total bytes: {bytes_total}")
    print(f"Missing fps: {missing_fps}")
    print(f"Missing frames: {missing_frames}")
    print(f"Read errors: {read_errors}")
    print(f"Total frames: {total_frames}")
    print(f"Total seconds: {total_seconds:.1f}")
    print("By action:")
    for action in sorted(action_set):
        print(
            f"- {action}: clips={counts.get(action, 0)}, "
            f"frames={frames_by_action.get(action, 0)}, "
            f"seconds={seconds_by_action.get(action, 0.0):.1f}"
        )

    if args.manifest and not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "kit_root": kit_root,
                    "output_dir": output_dir,
                    "actions": sorted(action_set),
                    "counts": dict(counts),
                    "frames": dict(frames_by_action),
                    "seconds": dict(seconds_by_action),
                    "missing_fps": missing_fps,
                    "missing_frames": missing_frames,
                    "read_errors": read_errors,
                    "total_frames": total_frames,
                    "total_seconds": total_seconds,
                    "files": dict(manifest),
                },
                f,
                indent=2,
            )
        print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()



'''
(skeleton1.0.1) robot@cq:~/RL/SkeletonSteering$ python scripts/extract_kit_expert_dataset.py --kit_root /home/robot/RL/KIT --output_dir expert_dataset --dry_run
KIT root: /home/robot/RL/KIT
Output dir: /home/robot/RL/SkeletonSteering/expert_dataset
Actions: leftturn, rightturn, run, walking, walking_fast, walking_medium, walking_run, walking_slow, walkingstraightforward, walkingstraightforwards
Matched files: 617
Copied files: 0
Skipped files: 0
Total bytes: 0
Missing fps: 0
Missing frames: 0
Read errors: 0
Total frames: 312573
Total seconds: 3125.7
By action:
- leftturn: clips=53, frames=26476, seconds=264.8
- rightturn: clips=79, frames=41647, seconds=416.5
- run: clips=33, frames=3437, seconds=34.4
- walking: clips=13, frames=8094, seconds=80.9
- walking_fast: clips=74, frames=33787, seconds=337.9
- walking_medium: clips=109, frames=56645, seconds=566.4
- walking_run: clips=90, frames=38127, seconds=381.3
- walking_slow: clips=108, frames=73066, seconds=730.7
- walkingstraightforward: clips=9, frames=5079, seconds=50.8
- walkingstraightforwards: clips=49, frames=26215, seconds=262.1

--dry_run
'''
