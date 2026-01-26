#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np


def iter_npz_files(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".npz"):
                yield os.path.join(dirpath, name)


def normalize_label(path, strip_suffix=True, strip_trailing_numbers=True, lower=True):
    base = os.path.basename(path)
    if strip_suffix and base.endswith("_poses.npz"):
        base = base[:-len("_poses.npz")]
    else:
        base = os.path.splitext(base)[0]
    if strip_trailing_numbers:
        base = re.sub(r"(_\d+)+$", "", base)
        base = re.sub(r"\d+$", "", base)
    if lower:
        base = base.lower()
    return base


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


def sort_items(items, sort_key):
    if sort_key == "name":
        return sorted(items, key=lambda kv: kv[0])
    if sort_key == "count":
        return sorted(items, key=lambda kv: (-kv[1]["clips"], kv[0]))
    return sorted(items, key=lambda kv: (-kv[1]["seconds"], kv[0]))


def main():
    parser = argparse.ArgumentParser(description="Analyze KIT motions: list each motion type and total seconds.")
    parser.add_argument("--kit_root", default="/home/robot/RL/KIT", help="Path to KIT dataset root.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--sort", choices=["seconds", "name", "count"], default="seconds")
    parser.add_argument("--top", type=int, default=0, help="Limit printed labels (0 = all).")
    parser.add_argument("--max_examples", type=int, default=3, help="Example paths per label.")
    parser.add_argument("--keep_case", action="store_true", help="Preserve original label casing.")
    parser.add_argument("--keep_trailing_numbers", action="store_true", help="Keep trailing indices in labels.")
    parser.add_argument("--keep_suffix", action="store_true", help="Keep suffix like _poses in labels.")
    args = parser.parse_args()

    kit_root = args.kit_root
    if not os.path.isdir(kit_root):
        raise SystemExit(f"KIT root not found: {kit_root}")

    strip_suffix = not args.keep_suffix
    strip_trailing_numbers = not args.keep_trailing_numbers
    lower = not args.keep_case

    stats = {}
    totals = {"files": 0, "processed": 0, "missing_fps": 0, "missing_frames": 0, "errors": 0}

    for path in iter_npz_files(kit_root):
        totals["files"] += 1
        label = normalize_label(
            path,
            strip_suffix=strip_suffix,
            strip_trailing_numbers=strip_trailing_numbers,
            lower=lower,
        )
        rel_path = os.path.relpath(path, kit_root)
        try:
            with np.load(path, allow_pickle=True) as data:
                framerate = get_framerate(data)
                n_frames = get_num_frames(data)
        except Exception:
            totals["errors"] += 1
            continue

        if framerate is None:
            totals["missing_fps"] += 1
            continue
        if n_frames is None:
            totals["missing_frames"] += 1
            continue
        if framerate <= 0:
            totals["missing_fps"] += 1
            continue

        seconds = n_frames / framerate
        entry = stats.setdefault(label, {"clips": 0, "frames": 0, "seconds": 0.0, "examples": []})
        entry["clips"] += 1
        entry["frames"] += n_frames
        entry["seconds"] += seconds
        if len(entry["examples"]) < args.max_examples:
            entry["examples"].append(rel_path)
        totals["processed"] += 1

    items = sort_items(list(stats.items()), args.sort)
    if args.top > 0:
        items = items[: args.top]

    total_seconds = sum(v["seconds"] for v in stats.values())
    total_frames = sum(v["frames"] for v in stats.values())

    print(f"KIT root: {kit_root}")
    print(f"Files: {totals['files']}")
    print(f"Processed: {totals['processed']}")
    print(f"Missing fps: {totals['missing_fps']}")
    print(f"Missing frames: {totals['missing_frames']}")
    print(f"Errors: {totals['errors']}")
    print(f"Total frames: {total_frames}")
    print(f"Total seconds: {total_seconds:.1f} ({total_seconds / 3600.0:.2f}h)")
    print("")
    print("Motion types:")
    for label, info in items:
        examples = ", ".join(info["examples"])
        print(
            f"- {label}: clips={info['clips']}, frames={info['frames']}, "
            f"seconds={info['seconds']:.1f} (e.g. {examples})"
        )

    if args.output:
        payload = {
            "kit_root": kit_root,
            "totals": {
                "files": totals["files"],
                "processed": totals["processed"],
                "missing_fps": totals["missing_fps"],
                "missing_frames": totals["missing_frames"],
                "errors": totals["errors"],
                "frames": total_frames,
                "seconds": total_seconds,
            },
            "labels": items,
            "options": {
                "strip_suffix": strip_suffix,
                "strip_trailing_numbers": strip_trailing_numbers,
                "lower": lower,
            },
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON summary to {args.output}")


if __name__ == "__main__":
    main()


'''
(skeleton1.0.1) robot@cq:~/RL/SkeletonSteering$ python scripts/analyze_kit_dataset.py --kit_root /home/robot/RL/KIT --output /tmp/kit_summary.json
KIT root: /home/robot/RL/KIT
Files: 4232
Processed: 4232
Missing fps: 0
Missing frames: 0
Errors: 0
Total frames: 3971045
Total seconds: 39710.5 (11.03h)

Motion types:
- trial: clips=171, frames=342440, seconds=3424.4 (e.g. 1226/Trial_53_poses.npz, 1226/Trial_14_poses.npz, 1226/Trial_30_poses.npz)
- experiment3a: clips=20, frames=173629, seconds=1736.3 (e.g. 987/Experiment3a_05_poses.npz, 987/Experiment3a_06_poses.npz, 987/Experiment3a_02_poses.npz)
- marcuss_adrianm: clips=8, frames=136878, seconds=1368.8 (e.g. 576/MarcusS_AdrianM09_poses.npz, 576/MarcusS_AdrianM11_poses.npz, 576/MarcusS_AdrianM12_poses.npz)
- experiment3b: clips=11, frames=90113, seconds=901.1 (e.g. 987/Experiment3b_07_poses.npz, 987/Experiment3b_02_poses.npz, 987/Experiment3b_05_poses.npz)
- displace_tilt_side: clips=82, frames=80794, seconds=807.9 (e.g. 1717/displace_tilt_side_26_poses.npz, 1717/displace_tilt_side_31_poses.npz, 1717/displace_tilt_side_01_poses.npz)
- walking_slow: clips=108, frames=73066, seconds=730.7 (e.g. 424/walking_slow04_poses.npz, 424/walking_slow02_poses.npz, 424/walking_slow07_poses.npz)
- parkour: clips=56, frames=63001, seconds=630.0 (e.g. 424/parkour06_poses.npz, 424/parkour03_poses.npz, 424/parkour05_poses.npz)
- turn_left: clips=113, frames=61838, seconds=618.4 (e.g. 424/turn_left07_poses.npz, 424/turn_left04_poses.npz, 424/turn_left06_poses.npz)
- step_over_gap: clips=107, frames=59206, seconds=592.1 (e.g. 424/step_over_gap08_poses.npz, 424/step_over_gap10_poses.npz, 424/step_over_gap02_poses.npz)
- walkincounterclockwisecircle: clips=77, frames=57351, seconds=573.5 (e.g. 10/WalkInCounterClockwiseCircle06_poses.npz, 10/WalkInCounterClockwiseCircle02_poses.npz, 10/WalkInCounterClockwiseCircle04_poses.npz)
- walkinclockwisecircle: clips=77, frames=57191, seconds=571.9 (e.g. 10/WalkInClockwiseCircle03_poses.npz, 10/WalkInClockwiseCircle05_poses.npz, 10/WalkInClockwiseCircle10_poses.npz)
- walking_medium: clips=109, frames=56645, seconds=566.4 (e.g. 424/walking_medium01_poses.npz, 424/walking_medium03_poses.npz, 424/walking_medium04_poses.npz)
- go_over_beam: clips=75, frames=55632, seconds=556.3 (e.g. 167/go_over_beam01_poses.npz, 167/go_over_beam05_poses.npz, 167/go_over_beam09_poses.npz)
- turn_right: clips=97, frames=55566, seconds=555.7 (e.g. 424/turn_right02_poses.npz, 424/turn_right08_poses.npz, 424/turn_right09_poses.npz)
- : clips=32, frames=53074, seconds=530.7 (e.g. 3/912_3_12_poses.npz, 3/912_3_15_poses.npz, 3/912_3_04_poses.npz)
- bend_left: clips=99, frames=50534, seconds=505.3 (e.g. 424/bend_left09_poses.npz, 424/bend_left02_poses.npz, 424/bend_left06_poses.npz)
- upstairs: clips=75, frames=50448, seconds=504.5 (e.g. 424/upstairs03_poses.npz, 424/upstairs01_poses.npz, 424/upstairs10_poses.npz)
- bend_right: clips=95, frames=47821, seconds=478.2 (e.g. 424/bend_right05_poses.npz, 424/bend_right03_poses.npz, 424/bend_right09_poses.npz)
- displace_from_left_to_right: clips=96, frames=46750, seconds=467.5 (e.g. 1717/displace_from_left_to_right_11_poses.npz, 1717/displace_from_left_to_right_21_poses.npz, 1717/displace_from_left_to_right_18_poses.npz)
- walkingstraightbackwards: clips=79, frames=42552, seconds=425.5 (e.g. 10/WalkingStraightBackwards03_poses.npz, 10/WalkingStraightBackwards04_poses.npz, 10/WalkingStraightBackwards07_poses.npz)
- rightturn: clips=79, frames=41647, seconds=416.5 (e.g. 10/RightTurn06_poses.npz, 10/RightTurn10_poses.npz, 10/RightTurn03_poses.npz)
- wash_front: clips=5, frames=40206, seconds=402.1 (e.g. 674/wash_front01_poses.npz, 674/wash_front02_poses.npz, 674/wash_front03_poses.npz)
- downstairs: clips=71, frames=40040, seconds=400.4 (e.g. 424/downstairs01_poses.npz, 424/downstairs04_poses.npz, 424/downstairs09_poses.npz)
- experiment3b_subject1347_back: clips=3, frames=39727, seconds=397.3 (e.g. 1347/Experiment3b_subject1347_back_03_poses.npz, 1347/Experiment3b_subject1347_back_01_poses.npz, 1347/Experiment3b_subject1347_back_02_poses.npz)
- wash_left_arm: clips=5, frames=38269, seconds=382.7 (e.g. 675/wash_left_arm_01_poses.npz, 675/wash_left_arm_04_poses.npz, 675/wash_left_arm_05_poses.npz)
- walking_run: clips=90, frames=38127, seconds=381.3 (e.g. 424/walking_run08_poses.npz, 424/walking_run02_poses.npz, 424/walking_run06_poses.npz)
- wash_back: clips=5, frames=36083, seconds=360.8 (e.g. 674/wash_back01_poses.npz, 674/wash_back03_poses.npz, 674/wash_back02_poses.npz)
- double_pour_and_mixing: clips=15, frames=35589, seconds=355.9 (e.g. 675/double_pour_and_mixing03_poses.npz, 675/double_pour_and_mixing11_poses.npz, 675/double_pour_and_mixing10_poses.npz)
- dry_left_arm: clips=15, frames=35083, seconds=350.8 (e.g. 675/dry_left_arm_06_poses.npz, 675/dry_left_arm_04_poses.npz, 675/dry_left_arm_15_poses.npz)
- conversation: clips=8, frames=35058, seconds=350.6 (e.g. 442/conversation03_poses.npz, 442/conversation01_poses.npz, 442/conversation04_poses.npz)
- displace_tilt_middle: clips=38, frames=34992, seconds=349.9 (e.g. 1717/displace_tilt_middle_12_poses.npz, 1717/displace_tilt_middle_10_poses.npz, 1717/displace_tilt_middle_11_poses.npz)
- wash_right_arm: clips=5, frames=34393, seconds=343.9 (e.g. 675/wash_right_arm_01_poses.npz, 675/wash_right_arm_06_poses.npz, 675/wash_right_arm_02_poses.npz)
- walking_fast: clips=74, frames=33787, seconds=337.9 (e.g. 424/walking_fast07_poses.npz, 424/walking_fast09_poses.npz, 424/walking_fast04_poses.npz)
- upstairs_downstairs: clips=34, frames=33227, seconds=332.3 (e.g. 424/upstairs_downstairs04_poses.npz, 424/upstairs_downstairs03_poses.npz, 424/upstairs_downstairs02_poses.npz)
- dry_right_arm: clips=15, frames=32437, seconds=324.4 (e.g. 675/dry_right_arm_011_poses.npz, 675/dry_right_arm_007_poses.npz, 675/dry_right_arm_009_poses.npz)
- wash_head: clips=5, frames=30241, seconds=302.4 (e.g. 674/wash_head01_poses.npz, 674/wash_head02_poses.npz, 674/wash_head03_poses.npz)
- recoverystepping: clips=67, frames=26960, seconds=269.6 (e.g. 1721/RecoveryStepping_50_120_02_poses.npz, 1721/RecoveryStepping_50_60_03_poses.npz, 1721/RecoveryStepping_30_180_01_poses.npz)
- leftturn: clips=53, frames=26476, seconds=264.8 (e.g. 10/LeftTurn06_poses.npz, 10/LeftTurn04_poses.npz, 10/LeftTurn02_poses.npz)
- walkingstraightforwards: clips=49, frames=26215, seconds=262.1 (e.g. 7/WalkingStraightForwards02_poses.npz, 7/WalkingStraightForwards10_poses.npz, 7/WalkingStraightForwards07_poses.npz)
- push_recovery_front: clips=38, frames=25452, seconds=254.5 (e.g. 379/push_recovery_front05_poses.npz, 379/push_recovery_front09_poses.npz, 379/push_recovery_front03_poses.npz)
- wipe: clips=15, frames=24934, seconds=249.3 (e.g. 1297/wipe02_poses.npz, 1297/wipe08_poses.npz, 1297/wipe09_poses.npz)
- step_stones: clips=27, frames=23866, seconds=238.7 (e.g. 424/step_stones07_poses.npz, 424/step_stones09_poses.npz, 424/step_stones08_poses.npz)
- drinking: clips=17, frames=23439, seconds=234.4 (e.g. 442/drinking03_poses.npz, 442/drinking02_poses.npz, 442/drinking04_poses.npz)
- push_recovery_right: clips=37, frames=23278, seconds=232.8 (e.g. 379/push_recovery_right04_poses.npz, 379/push_recovery_right05_poses.npz, 379/push_recovery_right08_poses.npz)
- pizzadelivery: clips=10, frames=21820, seconds=218.2 (e.g. 442/PizzaDelivery04_poses.npz, 442/PizzaDelivery03_poses.npz, 442/PizzaDelivery02_poses.npz)
- wave_both: clips=30, frames=21744, seconds=217.4 (e.g. 572/wave_both07_poses.npz, 572/wave_both02_poses.npz, 572/wave_both11_poses.npz)
- dance_chacha: clips=16, frames=21681, seconds=216.8 (e.g. 572/dance_chacha03_poses.npz, 572/dance_chacha08_poses.npz, 572/dance_chacha04_poses.npz)
- push_recovery_stand_left: clips=40, frames=21658, seconds=216.6 (e.g. 379/push_recovery_stand_left04_poses.npz, 379/push_recovery_stand_left01_poses.npz, 379/push_recovery_stand_left10_poses.npz)
- push_recovery_left: clips=34, frames=21061, seconds=210.6 (e.g. 379/push_recovery_left06_poses.npz, 379/push_recovery_left08_poses.npz, 379/push_recovery_left10_poses.npz)
- wave_left: clips=31, frames=20399, seconds=204.0 (e.g. 572/wave_left04_poses.npz, 572/wave_left10_poses.npz, 572/wave_left16_poses.npz)
- dance_waltz: clips=14, frames=19905, seconds=199.0 (e.g. 572/dance_waltz04_poses.npz, 572/dance_waltz06_poses.npz, 572/dance_waltz14_poses.npz)
- wave_right: clips=28, frames=19471, seconds=194.7 (e.g. 572/wave_right05_poses.npz, 572/wave_right15_poses.npz, 572/wave_right10_poses.npz)
- experiment3b_subject1346_back: clips=2, frames=19455, seconds=194.6 (e.g. 1346/Experiment3b_subject1346_back_03_poses.npz, 1346/Experiment3b_subject1346_back_02_poses.npz)
- seesaw_backwards: clips=20, frames=19149, seconds=191.5 (e.g. 424/seesaw_backwards02_poses.npz, 424/seesaw_backwards01_poses.npz, 424/seesaw_backwards10_poses.npz)
- experiment3b_back: clips=3, frames=19029, seconds=190.3 (e.g. 3/Experiment3b_back_02_poses.npz, 3/Experiment3b_back_03_poses.npz, 3/Experiment3b_back_01_poses.npz)
- experiment3a_back: clips=3, frames=18752, seconds=187.5 (e.g. 3/Experiment3a_back_03_poses.npz, 3/Experiment3a_back_02_poses.npz, 3/Experiment3a_back_01_poses.npz)
- push_recovery_stand_right: clips=41, frames=18228, seconds=182.3 (e.g. 379/push_recovery_stand_right08_poses.npz, 379/push_recovery_stand_right11_poses.npz, 379/push_recovery_stand_right02_poses.npz)
- lean_over_table: clips=19, frames=18116, seconds=181.2 (e.g. 3/lean_over_table05_poses.npz, 3/lean_over_table18_poses.npz, 3/lean_over_table08_poses.npz)
- experiment3b_subject1346_legs: clips=2, frames=17286, seconds=172.9 (e.g. 1346/Experiment3b_subject1346_legs_02_poses.npz, 1346/Experiment3b_subject1346_legs_03_poses.npz)
- walk_slow_with_handrail_table_beam_right: clips=10, frames=17076, seconds=170.8 (e.g. 675/walk_slow_with_handrail_table_beam_right10_poses.npz, 675/walk_slow_with_handrail_table_beam_right05_poses.npz, 675/walk_slow_with_handrail_table_beam_right03_poses.npz)
- shower_left_arm: clips=15, frames=17034, seconds=170.3 (e.g. 674/shower_left_arm02_poses.npz, 674/shower_left_arm09_poses.npz, 674/shower_left_arm04_poses.npz)
- experiment4a: clips=2, frames=16836, seconds=168.4 (e.g. 675/Experiment4a_01_poses.npz, 675/Experiment4a_03_poses.npz)
- push_from_behind: clips=19, frames=16812, seconds=168.1 (e.g. 425/push_from_behind_08_poses.npz, 425/push_from_behind_01_poses.npz, 425/push_from_behind_04_poses.npz)
- push_recovery_stand_back: clips=39, frames=16795, seconds=167.9 (e.g. 379/push_recovery_stand_back01_poses.npz, 379/push_recovery_stand_back04_poses.npz, 379/push_recovery_stand_back02_poses.npz)
- experiment3a_subject1347_legs: clips=2, frames=16724, seconds=167.2 (e.g. 1347/Experiment3a_subject1347_legs_03_poses.npz, 1347/Experiment3a_subject1347_legs_01_poses.npz)
- experiment3a_subject1347_back: clips=3, frames=16626, seconds=166.3 (e.g. 1347/Experiment3a_subject1347_back_02_poses.npz, 1347/Experiment3a_subject1347_back_03_poses.npz, 1347/Experiment3a_subject1347_back_01_poses.npz)
- push_recovery_back: clips=36, frames=16505, seconds=165.0 (e.g. 379/push_recovery_back07_poses.npz, 379/push_recovery_back02_poses.npz, 379/push_recovery_back10_poses.npz)
- experiment3b_legs: clips=3, frames=16077, seconds=160.8 (e.g. 3/Experiment3b_legs_01_poses.npz, 3/Experiment3b_legs_03_poses.npz, 3/Experiment3b_legs_02_poses.npz)
- walk_slow_with_handrail_table_beam_left: clips=10, frames=15920, seconds=159.2 (e.g. 675/walk_slow_with_handrail_table_beam_left07_poses.npz, 675/walk_slow_with_handrail_table_beam_left04_poses.npz, 675/walk_slow_with_handrail_table_beam_left03_poses.npz)
- push_recovery_stand_front: clips=39, frames=15818, seconds=158.2 (e.g. 379/push_recovery_stand_front10_poses.npz, 379/push_recovery_stand_front09_poses.npz, 379/push_recovery_stand_front04_poses.npz)
- experiment3a_subject1346_back: clips=3, frames=15728, seconds=157.3 (e.g. 1346/Experiment3a_subject1346_back_02_poses.npz, 1346/Experiment3a_subject1346_back_01_poses.npz, 1346/Experiment3a_subject1346_back_03_poses.npz)
- violin_right: clips=16, frames=14075, seconds=140.8 (e.g. 572/violin_right06_poses.npz, 572/violin_right02_poses.npz, 572/violin_right09_poses.npz)
- shower_right_arm: clips=15, frames=13780, seconds=137.8 (e.g. 674/shower_right_arm02_poses.npz, 674/shower_right_arm09_poses.npz, 674/shower_right_arm04_poses.npz)
- seesaw: clips=20, frames=13756, seconds=137.6 (e.g. 424/seesaw07_poses.npz, 424/seesaw08_poses.npz, 424/seesaw03_poses.npz)
- wipe_leg_vertical: clips=14, frames=13270, seconds=132.7 (e.g. 955/wipe_leg_vertical01_poses.npz, 952/wipe_leg_vertical01_poses.npz, 883/wipe_leg_vertical02_poses.npz)
- lean_on_table: clips=17, frames=12993, seconds=129.9 (e.g. 3/lean_on_table14_poses.npz, 3/lean_on_table01_poses.npz, 3/lean_on_table06_poses.npz)
- wipe_head_vertical: clips=9, frames=12922, seconds=129.2 (e.g. 955/wipe_head_vertical01_poses.npz, 952/wipe_head_vertical01_poses.npz, 883/wipe_head_vertical03_poses.npz)
- wipe_head_dabbing: clips=12, frames=12688, seconds=126.9 (e.g. 955/wipe_head_dabbing01_poses.npz, 952/wipe_head_dabbing01_poses.npz, 952/wipe_head_dabbing02_poses.npz)
- push_from_the_right_side: clips=18, frames=12289, seconds=122.9 (e.g. 425/push_from_the_right_side_11_poses.npz, 425/push_from_the_right_side_12_poses.npz, 425/push_from_the_right_side_13_poses.npz)
- violin_left: clips=15, frames=12096, seconds=121.0 (e.g. 572/violin_left15_poses.npz, 572/violin_left02_poses.npz, 572/violin_left14_poses.npz)
- slope_up: clips=20, frames=11326, seconds=113.3 (e.g. 513/slope_up10_poses.npz, 513/slope_up04_poses.npz, 513/slope_up06_poses.npz)
- guitar_left: clips=18, frames=11265, seconds=112.6 (e.g. 572/guitar_left12_poses.npz, 572/guitar_left01_poses.npz, 572/guitar_left13_poses.npz)
- stir_left: clips=16, frames=11212, seconds=112.1 (e.g. 572/stir_left04_poses.npz, 572/stir_left08_poses.npz, 572/stir_left15_poses.npz)
- synchron: clips=10, frames=11036, seconds=110.4 (e.g. 442/synchron04_poses.npz, 442/synchron02_poses.npz, 442/synchron05_poses.npz)
- wipe_circular_left: clips=16, frames=11012, seconds=110.1 (e.g. 572/wipe_circular_left16_poses.npz, 572/wipe_circular_left13_poses.npz, 572/wipe_circular_left12_poses.npz)
- experiment3b_subject1347_legs: clips=1, frames=10836, seconds=108.4 (e.g. 1347/Experiment3b_subject1347_legs_01_poses.npz)
- experiment3a_legs: clips=2, frames=10425, seconds=104.2 (e.g. 3/Experiment3a_legs_01_poses.npz, 3/Experiment3a_legs_03_poses.npz)
- guitar_right: clips=15, frames=10376, seconds=103.8 (e.g. 572/guitar_right03_poses.npz, 572/guitar_right04_poses.npz, 572/guitar_right09_poses.npz)
- slope_down: clips=19, frames=10087, seconds=100.9 (e.g. 513/slope_down05_poses.npz, 513/slope_down01_poses.npz, 513/slope_down06_poses.npz)
- wipe_back_dabbing: clips=11, frames=10026, seconds=100.3 (e.g. 955/wipe_back_dabbing02_poses.npz, 955/wipe_back_dabbing01_poses.npz, 955/wipe_back_dabbing03_poses.npz)
- castbox: clips=10, frames=9970, seconds=99.7 (e.g. 442/castBox05_poses.npz, 442/castBox02_poses.npz, 442/castBox04_poses.npz)
- highfive: clips=8, frames=9752, seconds=97.5 (e.g. 442/HighFive01_poses.npz, 442/HighFive05_poses.npz, 442/HighFive03_poses.npz)
- seesaw_up: clips=10, frames=9751, seconds=97.5 (e.g. 513/seesaw_up02_poses.npz, 513/seesaw_up10_poses.npz, 513/seesaw_up07_poses.npz)
- stir_right: clips=15, frames=9692, seconds=96.9 (e.g. 572/stir_right13_poses.npz, 572/stir_right09_poses.npz, 572/stir_right04_poses.npz)
- experiment3a_subject1346_legs: clips=2, frames=9487, seconds=94.9 (e.g. 1346/Experiment3a_subject1346_legs_03_poses.npz, 1346/Experiment3a_subject1346_legs_02_poses.npz)
- wipe_leg_dabbing: clips=11, frames=9483, seconds=94.8 (e.g. 955/wipe_leg_dabbing03_poses.npz, 955/wipe_leg_dabbing02_poses.npz, 955/wipe_leg_dabbing01_poses.npz)
- wipe_back_vertical: clips=9, frames=9036, seconds=90.4 (e.g. 955/wipe_back_vertical02_poses.npz, 952/wipe_back_vertical02_poses.npz, 952/wipe_back_vertical01_poses.npz)
- kneel_down_with_left_hand: clips=10, frames=8987, seconds=89.9 (e.g. 3/kneel_down_with_left_hand10_poses.npz, 3/kneel_down_with_left_hand02_poses.npz, 3/kneel_down_with_left_hand09_poses.npz)
- go_over_beam_n: clips=10, frames=8590, seconds=85.9 (e.g. 183/go_over_beam_n08_poses.npz, 183/go_over_beam_n05_poses.npz, 183/go_over_beam_n04_poses.npz)
- wipe_head_bigcircle: clips=7, frames=8529, seconds=85.3 (e.g. 952/wipe_head_bigcircle01_poses.npz, 952/wipe_head_bigcircle02_poses.npz, 883/wipe_head_bigcircle04_poses.npz)
- wipe_head_horizontal: clips=8, frames=8376, seconds=83.8 (e.g. 955/wipe_head_horizontal01_poses.npz, 883/wipe_head_horizontal03_poses.npz, 883/wipe_head_horizontal01_poses.npz)
- dry_front: clips=6, frames=8123, seconds=81.2 (e.g. 674/dry_front02_poses.npz, 674/dry_front04_poses.npz, 674/dry_front08_poses.npz)
- experiment3_subject1346_random: clips=3, frames=8098, seconds=81.0 (e.g. 1346/Experiment3_subject1346_random_01_poses.npz, 1346/Experiment3_subject1346_random_03_poses.npz, 1346/Experiment3_subject1346_random_02_poses.npz)
- walking: clips=13, frames=8094, seconds=80.9 (e.g. 291/walking02_poses.npz, 291/walking03_poses.npz, 425/walking_10_poses.npz)
- walk_with_table_right: clips=10, frames=8035, seconds=80.3 (e.g. 675/walk_with_table_right05_poses.npz, 675/walk_with_table_right08_poses.npz, 675/walk_with_table_right02_poses.npz)
- transfuse: clips=3, frames=7912, seconds=79.1 (e.g. 3/Transfuse03_poses.npz, 3/Transfuse02_poses.npz, 3/Transfuse07_poses.npz)
- preparing_the_dough: clips=4, frames=7889, seconds=78.9 (e.g. 3/Preparing_the_dough02_poses.npz, 3/Preparing_the_dough01_poses.npz, 3/Preparing_the_dough10_poses.npz)
- crawl: clips=11, frames=7860, seconds=78.6 (e.g. 3/crawl03_poses.npz, 3/crawl05_poses.npz, 3/crawl06_poses.npz)
- kneel_down_with_right_hand: clips=10, frames=7732, seconds=77.3 (e.g. 3/kneel_down_with_right_hand08_poses.npz, 3/kneel_down_with_right_hand04_poses.npz, 3/kneel_down_with_right_hand09_poses.npz)
- walk_with_handrail_table_beam_right: clips=9, frames=7314, seconds=73.1 (e.g. 675/walk_with_handrail_table_beam_right06_poses.npz, 675/walk_with_handrail_table_beam_right07_poses.npz, 675/walk_with_handrail_table_beam_right01_poses.npz)
- kneel_down_with_both_hands: clips=10, frames=7293, seconds=72.9 (e.g. 3/kneel_down_with_both_hands05_poses.npz, 3/kneel_down_with_both_hands02_poses.npz, 3/kneel_down_with_both_hands09_poses.npz)
- balance_on_beam: clips=11, frames=7264, seconds=72.6 (e.g. 513/balance_on_beam11_poses.npz, 513/balance_on_beam10_poses.npz, 513/balance_on_beam08_poses.npz)
- seesaw_down: clips=10, frames=7250, seconds=72.5 (e.g. 513/seesaw_down05_poses.npz, 513/seesaw_down02_poses.npz, 513/seesaw_down09_poses.npz)
- golf_drive: clips=11, frames=7115, seconds=71.1 (e.g. 572/golf_drive02_poses.npz, 572/golf_drive04_poses.npz, 572/golf_drive05_poses.npz)
- experiment3_subject1347_wash_leg_position_smallcircles: clips=2, frames=7021, seconds=70.2 (e.g. 1347/Experiment3_subject1347_wash_leg_position_smallcircles_02_poses.npz, 1347/Experiment3_subject1347_wash_leg_position_smallcircles_01_poses.npz)
- walk_with_table_left: clips=10, frames=6946, seconds=69.5 (e.g. 675/walk_with_table_left07_poses.npz, 675/walk_with_table_left06_poses.npz, 675/walk_with_table_left04_poses.npz)
- wipe_back_horizontal: clips=7, frames=6744, seconds=67.4 (e.g. 952/wipe_back_horizontal03_poses.npz, 952/wipe_back_horizontal02_poses.npz, 952/wipe_back_horizontal01_poses.npz)
- walk_with_handrail_table_left: clips=10, frames=6692, seconds=66.9 (e.g. 675/walk_with_handrail_table_left009_poses.npz, 675/walk_with_handrail_table_left008_poses.npz, 675/walk_with_handrail_table_left001_poses.npz)
- walking_upstairs: clips=7, frames=6596, seconds=66.0 (e.g. 3/walking_upstairs03_poses.npz, 3/walking_upstairs01_poses.npz, 3/walking_upstairs05_poses.npz)
- walk_with_handrail_table_beam_left: clips=10, frames=6584, seconds=65.8 (e.g. 675/walk_with_handrail_table_beam_left08_poses.npz, 675/walk_with_handrail_table_beam_left10_poses.npz, 675/walk_with_handrail_table_beam_left04_poses.npz)
- push_from_the_front: clips=7, frames=6540, seconds=65.4 (e.g. 3/push_from_the_front09_poses.npz, 3/push_from_the_front05_poses.npz, 3/push_from_the_front10_poses.npz)
- balancing: clips=6, frames=6449, seconds=64.5 (e.g. 3/balancing05_poses.npz, 3/balancing_poses.npz, 3/balancing02_poses.npz)
- tennis_smash_right: clips=10, frames=6407, seconds=64.1 (e.g. 572/tennis_smash_right03_poses.npz, 572/tennis_smash_right05_poses.npz, 572/tennis_smash_right01_poses.npz)
- tennis_smash_left: clips=10, frames=6366, seconds=63.7 (e.g. 572/tennis_smash_left02_poses.npz, 572/tennis_smash_left04_poses.npz, 572/tennis_smash_left01_poses.npz)
- inspect_shoe_sole: clips=7, frames=6259, seconds=62.6 (e.g. 3/inspect_shoe_sole02_poses.npz, 3/inspect_shoe_sole03_poses.npz, 3/inspect_shoe_sole08_poses.npz)
- experiment3_subject1346_wash_leg_position_horizontal: clips=2, frames=6235, seconds=62.4 (e.g. 1346/Experiment3_subject1346_wash_leg_position_horizontal_02_poses.npz, 1346/Experiment3_subject1346_wash_leg_position_horizontal_01_poses.npz)
- experiment3_subject1346_wash_back_position_horizontal: clips=2, frames=6152, seconds=61.5 (e.g. 1346/Experiment3_subject1346_wash_back_position_horizontal_02_poses.npz, 1346/Experiment3_subject1346_wash_back_position_horizontal_01_poses.npz)
- walk_with_handrail_table_right: clips=10, frames=6135, seconds=61.4 (e.g. 675/walk_with_handrail_table_right002_poses.npz, 675/walk_with_handrail_table_right008_poses.npz, 675/walk_with_handrail_table_right003_poses.npz)
- kneel_down_with_stool_left_hand: clips=11, frames=6118, seconds=61.2 (e.g. 3/kneel_down_with_stool_left_hand11_poses.npz, 3/kneel_down_with_stool_left_hand04_poses.npz, 3/kneel_down_with_stool_left_hand08_poses.npz)
- kick_high_left: clips=10, frames=5966, seconds=59.7 (e.g. 572/kick_high_left01_poses.npz, 572/kick_high_left05_poses.npz, 572/kick_high_left03_poses.npz)
- experiment3_subject1346_wash_leg_position_smallcircles: clips=2, frames=5959, seconds=59.6 (e.g. 1346/Experiment3_subject1346_wash_leg_position_smallcircles_02_poses.npz, 1346/Experiment3_subject1346_wash_leg_position_smallcircles_01_poses.npz)
- wipe_back_bigcircles: clips=5, frames=5811, seconds=58.1 (e.g. 955/wipe_back_bigcircles02_poses.npz, 955/wipe_back_bigcircles03_poses.npz, 955/wipe_back_bigcircles01_poses.npz)
- kneel_down_with_stool_right_hand: clips=11, frames=5803, seconds=58.0 (e.g. 3/kneel_down_with_stool_right_hand01_poses.npz, 3/kneel_down_with_stool_right_hand04_poses.npz, 3/kneel_down_with_stool_right_hand09_poses.npz)
- pouring_the_bottle: clips=5, frames=5733, seconds=57.3 (e.g. 1487/pouring_the_bottle05_poses.npz, 1487/pouring_the_bottle04_poses.npz, 1487/pouring_the_bottle01_poses.npz)
- kneel_up_with_right_hand: clips=11, frames=5720, seconds=57.2 (e.g. 3/kneel_up_with_right_hand01_poses.npz, 3/kneel_up_with_right_hand08_poses.npz, 3/kneel_up_with_right_hand05_poses.npz)
- wipe_arm_vertical: clips=6, frames=5691, seconds=56.9 (e.g. 955/wipe_arm_vertical01_poses.npz, 952/wipe_arm_vertical01_poses.npz, 883/wipe_arm_vertical05_poses.npz)
- kneel_up_from_crawl: clips=10, frames=5685, seconds=56.9 (e.g. 3/kneel_up_from_crawl08_poses.npz, 3/kneel_up_from_crawl01_poses.npz, 3/kneel_up_from_crawl07_poses.npz)
- experiment3_subject1346_wash_leg_position_bigcircles: clips=2, frames=5651, seconds=56.5 (e.g. 1346/Experiment3_subject1346_wash_leg_position_bigcircles_02_poses.npz, 1346/Experiment3_subject1346_wash_leg_position_bigcircles_01_poses.npz)
- wipe_circular_right: clips=10, frames=5577, seconds=55.8 (e.g. 572/wipe_circular_right01_poses.npz, 572/wipe_circular_right10_poses.npz, 572/wipe_circular_right09_poses.npz)
- tennis_forehand_left: clips=10, frames=5448, seconds=54.5 (e.g. 572/tennis_forehand_left02_poses.npz, 572/tennis_forehand_left03_poses.npz, 572/tennis_forehand_left05_poses.npz)
- wipe_arm_dabbing: clips=7, frames=5391, seconds=53.9 (e.g. 955/wipe_arm_dabbing01_poses.npz, 952/wipe_arm_dabbing01_poses.npz, 883/wipe_arm_dabbing04_poses.npz)
- pour: clips=6, frames=5388, seconds=53.9 (e.g. 551/pour03_poses.npz, 551/pour06_poses.npz, 551/pour05_poses.npz)
- dry_head: clips=5, frames=5383, seconds=53.8 (e.g. 674/dry_head05_poses.npz, 674/dry_head01_poses.npz, 674/dry_head03_poses.npz)
- kneel_up_hold: clips=5, frames=5372, seconds=53.7 (e.g. 3/kneel_up_hold05_poses.npz, 3/kneel_up_hold02_poses.npz, 3/kneel_up_hold03_poses.npz)
- tennis_forehand_right: clips=9, frames=5371, seconds=53.7 (e.g. 572/tennis_forehand_right01_poses.npz, 572/tennis_forehand_right04_poses.npz, 572/tennis_forehand_right03_poses.npz)
- kick: clips=8, frames=5370, seconds=53.7 (e.g. 513/kick09_poses.npz, 513/kick07_poses.npz, 513/kick04_poses.npz)
- push_from_the_left_side: clips=8, frames=5369, seconds=53.7 (e.g. 3/push_from_the_left_side12_poses.npz, 3/push_from_the_left_side03_poses.npz, 3/push_from_the_left_side08_poses.npz)
- experiment3_subject1346_wash_leg_position_vertical: clips=2, frames=5293, seconds=52.9 (e.g. 1346/Experiment3_subject1346_wash_leg_position_vertical_02_poses.npz, 1346/Experiment3_subject1346_wash_leg_position_vertical_01_poses.npz)
- stirring_in_the_bowl: clips=6, frames=5282, seconds=52.8 (e.g. 1487/stirring_in_the_bowl03_poses.npz, 1487/stirring_in_the_bowl07_poses.npz, 1487/stirring_in_the_bowl05_poses.npz)
- experiment3_subject1347_random: clips=3, frames=5265, seconds=52.6 (e.g. 1347/Experiment3_subject1347_random_01_poses.npz, 1347/Experiment3_subject1347_random_03_poses.npz, 1347/Experiment3_subject1347_random_02_poses.npz)
- golf_putting: clips=10, frames=5256, seconds=52.6 (e.g. 572/golf_putting01_poses.npz, 572/golf_putting05_poses.npz, 572/golf_putting03_poses.npz)
- jump_up: clips=10, frames=5226, seconds=52.3 (e.g. 572/jump_up02_poses.npz, 572/jump_up01_poses.npz, 572/jump_up04_poses.npz)
- kick_low_left: clips=10, frames=5087, seconds=50.9 (e.g. 572/kick_low_left02_poses.npz, 572/kick_low_left01_poses.npz, 572/kick_low_left03_poses.npz)
- walkingstraightforward: clips=9, frames=5079, seconds=50.8 (e.g. 4/WalkingStraightForward08_poses.npz, 4/WalkingStraightForward01_poses.npz, 4/WalkingStraightForward03_poses.npz)
- streching_leg: clips=6, frames=5012, seconds=50.1 (e.g. 3/streching_leg01_poses.npz, 3/streching_leg04_poses.npz, 3/streching_leg03_poses.npz)
- kick_low_right: clips=10, frames=4981, seconds=49.8 (e.g. 572/kick_low_right02_poses.npz, 572/kick_low_right01_poses.npz, 572/kick_low_right04_poses.npz)
- walk_with_handrail_beam_right: clips=10, frames=4974, seconds=49.7 (e.g. 675/walk_with_handrail_beam_right08_poses.npz, 675/walk_with_handrail_beam_right01_poses.npz, 675/walk_with_handrail_beam_right02_poses.npz)
- drink: clips=5, frames=4935, seconds=49.3 (e.g. 551/drink05_poses.npz, 551/drink01_poses.npz, 551/drink04_poses.npz)
- squat: clips=10, frames=4921, seconds=49.2 (e.g. 572/squat03_poses.npz, 572/squat05_poses.npz, 572/squat01_poses.npz)
- jump_left: clips=10, frames=4914, seconds=49.1 (e.g. 572/jump_left04_poses.npz, 572/jump_left01_poses.npz, 572/jump_left05_poses.npz)
- throw_left: clips=10, frames=4865, seconds=48.6 (e.g. 572/throw_left01_poses.npz, 572/throw_left03_poses.npz, 572/throw_left05_poses.npz)
- jump_back: clips=10, frames=4830, seconds=48.3 (e.g. 572/jump_back05_poses.npz, 572/jump_back04_poses.npz, 572/jump_back02_poses.npz)
- stomp_right: clips=10, frames=4830, seconds=48.3 (e.g. 572/stomp_right02_poses.npz, 572/stomp_right01_poses.npz, 572/stomp_right03_poses.npz)
- walk_with_handrail_left: clips=10, frames=4787, seconds=47.9 (e.g. 675/walk_with_handrail_left05_poses.npz, 675/walk_with_handrail_left07_poses.npz, 675/walk_with_handrail_left08_poses.npz)
- cup: clips=7, frames=4703, seconds=47.0 (e.g. 551/cup04_poses.npz, 551/cup07_poses.npz, 551/cup05_poses.npz)
- flexion_with_help: clips=5, frames=4671, seconds=46.7 (e.g. 1229/flexion_with_help_02_poses.npz, 1229/flexion_with_help_05_poses.npz, 1229/flexion_with_help_01_poses.npz)
- bow_deep: clips=10, frames=4649, seconds=46.5 (e.g. 572/bow_deep03_poses.npz, 572/bow_deep01_poses.npz, 572/bow_deep02_poses.npz)
- throw_right: clips=10, frames=4615, seconds=46.1 (e.g. 572/throw_right03_poses.npz, 572/throw_right04_poses.npz, 572/throw_right02_poses.npz)
- walk_with_handrail_beam_left: clips=10, frames=4573, seconds=45.7 (e.g. 675/walk_with_handrail_beam_left04_poses.npz, 675/walk_with_handrail_beam_left09_poses.npz, 675/walk_with_handrail_beam_left07_poses.npz)
- jump_right: clips=9, frames=4570, seconds=45.7 (e.g. 572/jump_right03_poses.npz, 572/jump_right05_poses.npz, 572/jump_right02_poses.npz)
- egg_whisk: clips=6, frames=4547, seconds=45.5 (e.g. 551/egg_whisk02_poses.npz, 551/egg_whisk05_poses.npz, 551/egg_whisk04_poses.npz)
- kneel_up_with_left_hand: clips=10, frames=4545, seconds=45.5 (e.g. 3/kneel_up_with_left_hand04_poses.npz, 3/kneel_up_with_left_hand01_poses.npz, 3/kneel_up_with_left_hand05_poses.npz)
- bow_slight: clips=10, frames=4486, seconds=44.9 (e.g. 572/bow_slight03_poses.npz, 572/bow_slight04_poses.npz, 572/bow_slight02_poses.npz)
- kneel_down_to_crawl: clips=8, frames=4451, seconds=44.5 (e.g. 3/kneel_down_to_crawl02_poses.npz, 3/kneel_down_to_crawl03_poses.npz, 3/kneel_down_to_crawl06_poses.npz)
- stomp_left: clips=10, frames=4449, seconds=44.5 (e.g. 572/stomp_left04_poses.npz, 572/stomp_left03_poses.npz, 572/stomp_left02_poses.npz)
- kick_high_right: clips=9, frames=4434, seconds=44.3 (e.g. 572/kick_high_right05_poses.npz, 572/kick_high_right03_poses.npz, 572/kick_high_right02_poses.npz)
- shower_head: clips=5, frames=4410, seconds=44.1 (e.g. 674/shower_head02_poses.npz, 674/shower_head03_poses.npz, 674/shower_head04_poses.npz)
- sponge_big: clips=7, frames=4378, seconds=43.8 (e.g. 551/sponge_big03_poses.npz, 551/sponge_big01_poses.npz, 551/sponge_big04_poses.npz)
- knife: clips=6, frames=4128, seconds=41.3 (e.g. 551/knife02_poses.npz, 551/knife05_poses.npz, 551/knife03_poses.npz)
- push_medium: clips=4, frames=4118, seconds=41.2 (e.g. 471/Push_Medium01_poses.npz, 471/Push_Medium02_poses.npz, 471/Push_Medium04_poses.npz)
- wipe_back_smallcircles: clips=4, frames=4109, seconds=41.1 (e.g. 955/wipe_back_smallcircles02_poses.npz, 955/wipe_back_smallcircles01_poses.npz, 883/wipe_back_smallcircles05_poses.npz)
- downstairs_backwards: clips=5, frames=4094, seconds=40.9 (e.g. 3/downstairs_backwards02_poses.npz, 3/downstairs_backwards01_poses.npz, 3/downstairs_backwards03_poses.npz)
- mixing_bowl_big: clips=6, frames=4081, seconds=40.8 (e.g. 551/mixing_bowl_big03_poses.npz, 551/mixing_bowl_big01_poses.npz, 551/mixing_bowl_big06_poses.npz)
- jump_forward: clips=9, frames=4079, seconds=40.8 (e.g. 572/jump_forward04_poses.npz, 572/jump_forward05_poses.npz, 572/jump_forward02_poses.npz)
- wipe_head_small_circle: clips=5, frames=4040, seconds=40.4 (e.g. 948/Wipe_Head_Small_Circle_01_poses.npz, 948/Wipe_Head_Small_Circle_02_poses.npz, 950/Wipe_Head_Small_Circle_01_poses.npz)
- wipe_leg_smallcircles: clips=3, frames=3811, seconds=38.1 (e.g. 955/wipe_leg_smallcircles02_poses.npz, 955/wipe_leg_smallcircles01_poses.npz, 952/wipe_leg_smallcircles02_poses.npz)
- walk_with_handrail_right: clips=10, frames=3784, seconds=37.8 (e.g. 675/walk_with_handrail_right07_poses.npz, 675/walk_with_handrail_right02_poses.npz, 675/walk_with_handrail_right09_poses.npz)
- walk_with_support: clips=6, frames=3675, seconds=36.8 (e.g. 3/walk_with_support05_poses.npz, 3/walk_with_support06_poses.npz, 3/walk_with_support01_poses.npz)
- supination_with_help: clips=5, frames=3634, seconds=36.3 (e.g. 1229/supination_with_help_04_poses.npz, 1229/supination_with_help_01_poses.npz, 1229/supination_with_help_03_poses.npz)
- wipe_back_biglcircles: clips=3, frames=3597, seconds=36.0 (e.g. 952/wipe_back_biglcircles03_poses.npz, 952/wipe_back_biglcircles01_poses.npz, 952/wipe_back_biglcircles02_poses.npz)
- evasion: clips=8, frames=3575, seconds=35.8 (e.g. 513/evasion01_poses.npz, 513/evasion02_poses.npz, 513/evasion07_poses.npz)
- sponge_big_and_mixing_bowl_big: clips=5, frames=3564, seconds=35.6 (e.g. 551/sponge_big_and_mixing_bowl_big01_poses.npz, 551/sponge_big_and_mixing_bowl_big04_poses.npz, 551/sponge_big_and_mixing_bowl_big03_poses.npz)
- cutting_cucumber: clips=3, frames=3514, seconds=35.1 (e.g. 551/cutting_cucumber03_poses.npz, 551/cutting_cucumber04_poses.npz, 551/cutting_cucumber01_poses.npz)
- wipe_head_smallcircle: clips=3, frames=3513, seconds=35.1 (e.g. 952/wipe_head_smallcircle01_poses.npz, 883/wipe_head_smallcircle04_poses.npz, 883/wipe_head_smallcircle01_poses.npz)
- pour_and_mixing_normal_speed: clips=2, frames=3507, seconds=35.1 (e.g. 425/pour_and_mixing_normal_speed_01_poses.npz, 425/pour_and_mixing_normal_speed_02_poses.npz)
- punch_right: clips=7, frames=3483, seconds=34.8 (e.g. 572/punch_right01_poses.npz, 572/punch_right04_poses.npz, 572/punch_right03_poses.npz)
- run: clips=33, frames=3437, seconds=34.4 (e.g. 424/run02_poses.npz, 424/run03_poses.npz, 424/run04_poses.npz)
- handstand: clips=5, frames=3402, seconds=34.0 (e.g. 200/Handstand01_poses.npz, 200/Handstand05_poses.npz, 200/Handstand06_poses.npz)
- push_soft: clips=3, frames=3359, seconds=33.6 (e.g. 471/Push_Soft03_poses.npz, 471/Push_Soft01_poses.npz, 471/Push_Soft02_poses.npz)
- kniebeuge: clips=6, frames=3309, seconds=33.1 (e.g. 200/Kniebeuge01_poses.npz, 200/Kniebeuge03_poses.npz, 200/Kniebeuge05_poses.npz)
- wipe_arm_smallcircles: clips=3, frames=3309, seconds=33.1 (e.g. 955/wipe_arm_smallcircles02_poses.npz, 955/wipe_arm_smallcircles01_poses.npz, 952/wipe_arm_smallcircles01_poses.npz)
- take_book_from_shelf_with_help: clips=5, frames=3294, seconds=32.9 (e.g. 1229/take_book_from_shelf_with_help_02_poses.npz, 1229/take_book_from_shelf_with_help_03_poses.npz, 1229/take_book_from_shelf_with_help_05_poses.npz)
- wiping_the_table: clips=4, frames=3247, seconds=32.5 (e.g. 1487/wiping_the_table05_poses.npz, 1487/wiping_the_table06_poses.npz, 1487/wiping_the_table01_poses.npz)
- mixing_bowl_big_and_sponge_big: clips=5, frames=3151, seconds=31.5 (e.g. 551/mixing_bowl_big_and_sponge_big04_poses.npz, 551/mixing_bowl_big_and_sponge_big05_poses.npz, 551/mixing_bowl_big_and_sponge_big01_poses.npz)
- wipe_back_smallcircle: clips=3, frames=3130, seconds=31.3 (e.g. 952/wipe_back_smallcircle01_poses.npz, 952/wipe_back_smallcircle02_poses.npz, 952/wipe_back_smallcircle03_poses.npz)
- in_out_rotation_with_help: clips=5, frames=3104, seconds=31.0 (e.g. 1229/in_out_rotation_with_help_03_poses.npz, 1229/in_out_rotation_with_help_05_poses.npz, 1229/in_out_rotation_with_help_01_poses.npz)
- cup_in_bowl: clips=4, frames=3006, seconds=30.1 (e.g. 551/cup_in_bowl01_poses.npz, 551/cup_in_bowl04_poses.npz, 551/cup_in_bowl03_poses.npz)
- punch_left: clips=6, frames=2999, seconds=30.0 (e.g. 572/punch_left02_poses.npz, 572/punch_left05_poses.npz, 572/punch_left01_poses.npz)
- push_left_soft: clips=3, frames=2997, seconds=30.0 (e.g. 513/Push_Left_Soft02_poses.npz, 513/Push_Left_Soft03_poses.npz, 513/Push_Left_Soft01_poses.npz)
- pour_and_mixing_different_speeds: clips=2, frames=2983, seconds=29.8 (e.g. 425/pour_and_mixing_different_speeds_01_poses.npz, 425/pour_and_mixing_different_speeds_02_poses.npz)
- take_of_t-shirt_with_help: clips=5, frames=2950, seconds=29.5 (e.g. 1229/take_of_t-shirt_with_help_04_poses.npz, 1229/take_of_t-shirt_with_help_02_poses.npz, 1229/take_of_t-shirt_with_help_03_poses.npz)
- cutting_zucchini: clips=3, frames=2919, seconds=29.2 (e.g. 551/cutting_zucchini03_poses.npz, 551/cutting_zucchini01_poses.npz, 551/cutting_zucchini02_poses.npz)
- push_left_hard: clips=3, frames=2884, seconds=28.8 (e.g. 513/Push_Left_Hard03_poses.npz, 513/Push_Left_Hard02_poses.npz, 513/Push_Left_Hard01_poses.npz)
- wipe_back_big_circle: clips=3, frames=2883, seconds=28.8 (e.g. 948/Wipe_Back_Big_Circle_01_poses.npz, 948/Wipe_Back_Big_Circle_02_poses.npz, 950/Wipe_Back_Big_Circle_01_poses.npz)
- wipe_leg_smallcircle: clips=3, frames=2883, seconds=28.8 (e.g. 883/wipe_leg_smallcircle03_poses.npz, 883/wipe_leg_smallcircle04_poses.npz, 883/wipe_leg_smallcircle02_poses.npz)
- washing_movement_with_help: clips=5, frames=2846, seconds=28.5 (e.g. 1229/washing_movement_with_help_03_poses.npz, 1229/washing_movement_with_help_05_poses.npz, 1229/washing_movement_with_help_04_poses.npz)
- push_hard: clips=3, frames=2806, seconds=28.1 (e.g. 471/Push_Hard03_poses.npz, 471/Push_Hard02_poses.npz, 471/Push_Hard01_poses.npz)
- push_left_medium: clips=3, frames=2798, seconds=28.0 (e.g. 513/Push_Left_Medium03_poses.npz, 513/Push_Left_Medium01_poses.npz, 513/Push_Left_Medium02_poses.npz)
- hand_through_hair_with_help: clips=5, frames=2730, seconds=27.3 (e.g. 1229/hand_through_hair_with_help_05_poses.npz, 1229/hand_through_hair_with_help_01_poses.npz, 1229/hand_through_hair_with_help_04_poses.npz)
- kickhuefthoch: clips=6, frames=2690, seconds=26.9 (e.g. 200/KickHuefthoch02_poses.npz, 200/KickHuefthoch03_poses.npz, 200/KickHuefthoch06_poses.npz)
- wipe_arm_bigcircles: clips=2, frames=2667, seconds=26.7 (e.g. 955/wipe_arm_bigcircles01_poses.npz, 952/wipe_arm_bigcircles01_poses.npz)
- hand_to_mouth_with_help: clips=5, frames=2540, seconds=25.4 (e.g. 1229/hand_to_mouth_with_help_02_poses.npz, 1229/hand_to_mouth_with_help_03_poses.npz, 1229/hand_to_mouth_with_help_05_poses.npz)
- kopfschulterkniefuss: clips=3, frames=2527, seconds=25.3 (e.g. 442/Kopfschulterkniefuss03_poses.npz, 442/Kopfschulterkniefuss01_poses.npz, 442/Kopfschulterkniefuss02_poses.npz)
- push_right_medium: clips=3, frames=2515, seconds=25.1 (e.g. 513/Push_Right_Medium02_poses.npz, 513/Push_Right_Medium03_poses.npz, 513/Push_Right_Medium01_poses.npz)
- wipe_arm_big_circle: clips=3, frames=2436, seconds=24.4 (e.g. 948/Wipe_Arm_Big_Circle_01_poses.npz, 950/Wipe_Arm_Big_Circle_03_poses.npz, 950/Wipe_Arm_Big_Circle_02_poses.npz)
- cutting_banana: clips=2, frames=2346, seconds=23.5 (e.g. 551/cutting_banana01_poses.npz, 551/cutting_banana02_poses.npz)
- walk_6m_straight_line: clips=2, frames=2239, seconds=22.4 (e.g. 3/walk_6m_straight_line06_poses.npz, 3/walk_6m_straight_line04_poses.npz)
- aufstehen: clips=5, frames=2198, seconds=22.0 (e.g. 442/Aufstehen02_poses.npz, 442/Aufstehen01_poses.npz, 442/Aufstehen04_poses.npz)
- mixing_cooking_spoon: clips=2, frames=2175, seconds=21.8 (e.g. 551/mixing_cooking_spoon01_poses.npz, 551/mixing_cooking_spoon03_poses.npz)
- push_right_hard: clips=3, frames=2155, seconds=21.6 (e.g. 513/Push_Right_Hard02_poses.npz, 513/Push_Right_Hard01_poses.npz, 513/Push_Right_Hard03_poses.npz)
- take_book_from_shelf_right_arm: clips=6, frames=2137, seconds=21.4 (e.g. 1229/take_book_from_shelf_right_arm_06_poses.npz, 1229/take_book_from_shelf_right_arm_02_poses.npz, 1229/take_book_from_shelf_right_arm_03_poses.npz)
- open_pants_right_arm: clips=6, frames=2134, seconds=21.3 (e.g. 1229/open_pants_right_arm_02_poses.npz, 1229/open_pants_right_arm_05_poses.npz, 1229/open_pants_right_arm_03_poses.npz)
- walking_forward_4steps_right: clips=4, frames=2125, seconds=21.2 (e.g. 3/walking_forward_4steps_right_poses.npz, 3/walking_forward_4steps_right_04_poses.npz, 3/walking_forward_4steps_right_05_poses.npz)
- erschreckenderherrbischof: clips=3, frames=2050, seconds=20.5 (e.g. 442/ErschreckenderHerrBischof01_poses.npz, 442/ErschreckenderHerrBischof03_poses.npz, 442/ErschreckenderHerrBischof02_poses.npz)
- washing_movement_right_arm: clips=5, frames=2036, seconds=20.4 (e.g. 1229/washing_movement_right_arm_03_poses.npz, 1229/washing_movement_right_arm_05_poses.npz, 1229/washing_movement_right_arm_02_poses.npz)
- push_behind_hard: clips=3, frames=2005, seconds=20.1 (e.g. 513/Push_Behind_Hard02_poses.npz, 513/Push_Behind_Hard03_poses.npz, 513/Push_Behind_Hard01_poses.npz)
- bouncen: clips=3, frames=1961, seconds=19.6 (e.g. 442/Bouncen02_poses.npz, 442/Bouncen01_poses.npz, 442/Bouncen03_poses.npz)
- washing_movement_left_arm: clips=5, frames=1948, seconds=19.5 (e.g. 1229/washing_movement_left_arm_05_poses.npz, 1229/washing_movement_left_arm_01_poses.npz, 1229/washing_movement_left_arm_04_poses.npz)
- wipe_head_big_circle: clips=2, frames=1936, seconds=19.4 (e.g. 948/Wipe_Head_Big_Circle_01_poses.npz, 950/Wipe_Head_Big_Circle_01_poses.npz)
- take_of_t-shirt_right_arm: clips=5, frames=1904, seconds=19.0 (e.g. 1229/take_of_t-shirt_right_arm_05_poses.npz, 1229/take_of_t-shirt_right_arm_01_poses.npz, 1229/take_of_t-shirt_right_arm_02_poses.npz)
- mixing_egg_whisk: clips=2, frames=1903, seconds=19.0 (e.g. 551/mixing_egg_whisk01_poses.npz, 551/mixing_egg_whisk02_poses.npz)
- cutting_banana_b: clips=2, frames=1901, seconds=19.0 (e.g. 551/cutting_banana_b_04_poses.npz, 551/cutting_banana_b_03_poses.npz)
- wipe_back_small_circle: clips=2, frames=1882, seconds=18.8 (e.g. 948/Wipe_Back_Small_Circle_01_poses.npz, 950/Wipe_Back_Small_Circle_01_poses.npz)
- push_behind_soft: clips=3, frames=1860, seconds=18.6 (e.g. 513/Push_Behind_Soft02_poses.npz, 513/Push_Behind_Soft03_poses.npz, 513/Push_Behind_Soft01_poses.npz)
- hand_through_hair_right_arm: clips=5, frames=1858, seconds=18.6 (e.g. 1229/hand_through_hair_right_arm_02_poses.npz, 1229/hand_through_hair_right_arm_04_poses.npz, 1229/hand_through_hair_right_arm_03_poses.npz)
- hand_to_mouth_right_arm: clips=5, frames=1837, seconds=18.4 (e.g. 1229/hand_to_mouth_right_arm_02_poses.npz, 1229/hand_to_mouth_right_arm_04_poses.npz, 1229/hand_to_mouth_right_arm_03_poses.npz)
- wipe_leg_bigcircle: clips=3, frames=1835, seconds=18.4 (e.g. 883/wipe_leg_bigcircle01_poses.npz, 883/wipe_leg_bigcircle02_poses.npz, 883/wipe_leg_bigcircle03_poses.npz)
- dance_waltz : clips=2, frames=1829, seconds=18.3 (e.g. 572/dance_waltz 01_poses.npz, 572/dance_waltz 02_poses.npz)
- put_objects_in_mixing_bowl: clips=1, frames=1813, seconds=18.1 (e.g. 425/put_objects_in_mixing_bowl_02_poses.npz)
- take_of_t-shirt_left_arm: clips=5, frames=1809, seconds=18.1 (e.g. 1229/take_of_t-shirt_left_arm_02_poses.npz, 1229/take_of_t-shirt_left_arm_03_poses.npz, 1229/take_of_t-shirt_left_arm_05_poses.npz)
- wipe_arm_smallcircle: clips=2, frames=1749, seconds=17.5 (e.g. 883/wipe_arm_smallcircle02_poses.npz, 883/wipe_arm_smallcircle03_poses.npz)
- hand_through_hair_left_arm: clips=5, frames=1748, seconds=17.5 (e.g. 1229/hand_through_hair_left_arm_04_poses.npz, 1229/hand_through_hair_left_arm_05_poses.npz, 1229/hand_through_hair_left_arm_03_poses.npz)
- shower_front: clips=2, frames=1706, seconds=17.1 (e.g. 674/shower_front01_poses.npz, 674/shower_front02_poses.npz)
- take_book_from_shelf_left_arm: clips=5, frames=1696, seconds=17.0 (e.g. 1229/take_book_from_shelf_left_arm_05_poses.npz, 1229/take_book_from_shelf_left_arm_02_poses.npz, 1229/take_book_from_shelf_left_arm_03_poses.npz)
- hand_to_mouth_left_arm: clips=5, frames=1690, seconds=16.9 (e.g. 1229/hand_to_mouth_left_arm_03_poses.npz, 1229/hand_to_mouth_left_arm_04_poses.npz, 1229/hand_to_mouth_left_arm_02_poses.npz)
- wipe_arm_small_circle: clips=2, frames=1687, seconds=16.9 (e.g. 948/Wipe_Arm_Small_Circle_01_poses.npz, 950/Wipe_Arm_Small_Circle_01_poses.npz)
- hinundschuhbinden: clips=3, frames=1683, seconds=16.8 (e.g. 442/HinUndSchuhbinden03_poses.npz, 442/HinUndSchuhbinden02_poses.npz, 442/HinUndSchuhbinden01_poses.npz)
- open_pants_left_arm: clips=5, frames=1683, seconds=16.8 (e.g. 1229/open_pants_left_arm_04_poses.npz, 1229/open_pants_left_arm_03_poses.npz, 1229/open_pants_left_arm_01_poses.npz)
- drehung: clips=3, frames=1651, seconds=16.5 (e.g. 442/Drehung03_poses.npz, 442/Drehung01_poses.npz, 442/Drehung02_poses.npz)
- wipe_leg_biglcircles: clips=1, frames=1584, seconds=15.8 (e.g. 952/wipe_leg_biglcircles01_poses.npz)
- walk_with_left_support: clips=2, frames=1516, seconds=15.2 (e.g. 3/walk_with_left_support02_poses.npz, 3/walk_with_left_support06_poses.npz)
- walk_with_right_support_table: clips=2, frames=1514, seconds=15.1 (e.g. 3/walk_with_right_support_table06_poses.npz, 3/walk_with_right_support_table02_poses.npz)
- walk_with_left_support_table: clips=2, frames=1511, seconds=15.1 (e.g. 3/walk_with_left_support_table01_poses.npz, 3/walk_with_left_support_table08_poses.npz)
- wipe_arm_bigcircle: clips=2, frames=1497, seconds=15.0 (e.g. 883/wipe_arm_bigcircle03_poses.npz, 883/wipe_arm_bigcircle01_poses.npz)
- wipe_head_smallcircles: clips=2, frames=1484, seconds=14.8 (e.g. 955/wipe_head_smallcircles02_poses.npz, 955/wipe_head_smallcircles01_poses.npz)
- push_right_soft: clips=2, frames=1453, seconds=14.5 (e.g. 513/Push_Right_Soft03_poses.npz, 513/Push_Right_Soft01_poses.npz)
- wipe_leg_small_circle: clips=2, frames=1406, seconds=14.1 (e.g. 948/Wipe_Leg_Small_Circle_01_poses.npz, 950/Wipe_Leg_Small_Circle_01_poses.npz)
- dry_back: clips=1, frames=1392, seconds=13.9 (e.g. 674/dry_back01_poses.npz)
- push_behind_medium: clips=2, frames=1384, seconds=13.8 (e.g. 513/Push_Behind_Medium03_poses.npz, 513/Push_Behind_Medium02_poses.npz)
- wipe_head_bigcircles: clips=1, frames=1332, seconds=13.3 (e.g. 955/wipe_head_bigcircles01_poses.npz)
- sponge_in_bowl: clips=2, frames=1301, seconds=13.0 (e.g. 551/sponge_in_bowl02_poses.npz, 551/sponge_in_bowl01_poses.npz)
- dreischritte: clips=2, frames=1217, seconds=12.2 (e.g. 442/Dreischritte02_poses.npz, 442/Dreischritte01_poses.npz)
- downstairs_b: clips=3, frames=1164, seconds=11.6 (e.g. 513/downstairs_b03_poses.npz, 513/downstairs_b02_poses.npz, 513/downstairs_b01_poses.npz)
- punching: clips=1, frames=1153, seconds=11.5 (e.g. 291/punching03_poses.npz)
- push_front_medium: clips=2, frames=1141, seconds=11.4 (e.g. 513/Push_Front_Medium03_poses.npz, 513/Push_Front_Medium02_poses.npz)
- wipe_back: clips=1, frames=1113, seconds=11.1 (e.g. 674/wipe_back01_poses.npz)
- wipe_leg_bigcircles: clips=1, frames=1104, seconds=11.0 (e.g. 955/wipe_leg_bigcircles01_poses.npz)
- push_front_hard: clips=2, frames=1082, seconds=10.8 (e.g. 513/Push_Front_Hard01_poses.npz, 513/Push_Front_Hard03_poses.npz)
- mixing_small_with_egg_whisk: clips=1, frames=1080, seconds=10.8 (e.g. 551/mixing_small_with_egg_whisk01_poses.npz)
- wipe_left_arm: clips=1, frames=1058, seconds=10.6 (e.g. 674/wipe_left_arm01_poses.npz)
- wipe_front: clips=1, frames=1038, seconds=10.4 (e.g. 674/wipe_front01_poses.npz)
- push_front_soft: clips=2, frames=1030, seconds=10.3 (e.g. 513/Push_Front_Soft01_poses.npz, 513/Push_Front_Soft03_poses.npz)
- wipe_right_arm: clips=1, frames=1002, seconds=10.0 (e.g. 674/wipe_right_arm01_poses.npz)
- pouring_the_cup: clips=1, frames=969, seconds=9.7 (e.g. 1487/pouring_the_cup07_poses.npz)
- point_at_left: clips=1, frames=926, seconds=9.3 (e.g. 291/point_at_left03_poses.npz)
- shake_hand: clips=2, frames=900, seconds=9.0 (e.g. 291/shake_hand04_poses.npz, 291/shake_hand05_poses.npz)
- shower_back: clips=1, frames=880, seconds=8.8 (e.g. 674/shower_back01_poses.npz)
- walk_with_right_support: clips=1, frames=880, seconds=8.8 (e.g. 3/walk_with_right_support02_poses.npz)
- salutieren: clips=2, frames=857, seconds=8.6 (e.g. 442/Salutieren02_poses.npz, 442/Salutieren01_poses.npz)
- nordic_walking: clips=1, frames=841, seconds=8.4 (e.g. 291/nordic_walking03_poses.npz)
- point_at_right: clips=1, frames=819, seconds=8.2 (e.g. 291/point_at_right03_poses.npz)
- wipe_leg_big_circle: clips=1, frames=812, seconds=8.1 (e.g. 948/Wipe_Leg_Big_Circle_01_poses.npz)
- walk_like_an_egyptian: clips=2, frames=771, seconds=7.7 (e.g. 291/walk_like_an_egyptian05_poses.npz, 291/walk_like_an_egyptian01_poses.npz)
- walk_with_left_support06_frqoni: clips=1, frames=762, seconds=7.6 (e.g. 3/walk_with_left_support06_FrqONi5_poses.npz)
- point_at: clips=2, frames=732, seconds=7.3 (e.g. 291/point_at05_poses.npz, 291/point_at01_poses.npz)
- wave: clips=2, frames=676, seconds=6.8 (e.g. 291/wave01_poses.npz, 291/wave05_poses.npz)
- air_guitar: clips=1, frames=670, seconds=6.7 (e.g. 291/air_guitar01_poses.npz)
- radschlagen: clips=2, frames=658, seconds=6.6 (e.g. 200/RadSchlagen03_poses.npz, 200/RadSchlagen02_poses.npz)
- downstaris: clips=1, frames=637, seconds=6.4 (e.g. 3/downstaris09_poses.npz)
- chicken: clips=2, frames=627, seconds=6.3 (e.g. 291/chicken04_poses.npz, 291/chicken01_poses.npz)
- punsh_left: clips=1, frames=616, seconds=6.2 (e.g. 572/punsh_left03_poses.npz)
- air_guitar_banging: clips=1, frames=596, seconds=6.0 (e.g. 291/air_guitar_banging01_poses.npz)
- punch: clips=1, frames=577, seconds=5.8 (e.g. 442/Punch01_poses.npz)
- jumping_jack: clips=2, frames=545, seconds=5.5 (e.g. 291/jumping_jack02_poses.npz, 291/jumping_jack01_poses.npz)
- drum: clips=2, frames=544, seconds=5.4 (e.g. 291/drum05_poses.npz, 291/drum03_poses.npz)
- cal: clips=1, frames=425, seconds=4.2 (e.g. 291/cal03_poses.npz)
- jumping_jumping_jack: clips=2, frames=422, seconds=4.2 (e.g. 291/jumping_jumping_jack01_poses.npz, 291/jumping_jumping_jack02_poses.npz)
- chris3mseminar cal : clips=1, frames=259, seconds=2.6 (e.g. 63/Chris3MSeminar Cal 08_poses.npz)

Wrote JSON summary to /tmp/kit_summary.json
'''



"""
[KIT] Reference dataset built
Output: data/kit_walk_run_turn_motion_dict.pkl
Total clips: 608
Total frames: 102406
Total duration: 3413.5s (0.95h)

By category:
- walk_forward: clips=353, frames=65758, seconds=2191.9
- run: clips=123, frames=13895, seconds=463.2
- turn: clips=132, frames=22753, seconds=758.4

Action labels (normalized):
- LeftTurn: 53 clips, 294.8s
- RightTurn: 79 clips, 463.7s
- WalkingStraightForward: 9 clips, 56.5s
- WalkingStraightForwards: 49 clips, 291.9s
- run: 33 clips, 38.5s
- walking_fast: 74 clips, 376.2s
- walking_forward_4steps_right: 4 clips, 23.6s
- walking_medium: 109 clips, 630.7s
- walking_run: 90 clips, 424.6s
- walking_slow: 108 clips, 813.0s


补充：
- LeftTurn / RightTurn：向左/向右转身。
- turn_left / turn_right：原地转弯  删去
- WalkInClockwiseCircle / WalkInCounterClockwiseCircle：沿顺时针/逆时针圆周行走。  删去
- WalkingStraightForward / WalkingStraightForwards：直线向前走（两个拼写版本）。
- walking_slow / walking_medium / walking_fast：慢/中/快走。
- walk_6m_straight_line：直线走 6 米。   删去
- walking_forward_4steps_right：向前走 4 步（右脚起步/或右侧版本，按文件名推断）。
- run：跑步。
- walking_run：走-跑混合过渡/慢跑类（文件名包含 run）。


最终选用：
turn：LeftTurn、RightTurn
walk：walking_slow、walking_medium、walking_run、walking_fast、walkingstraightforwards、walking、walkingstraightforward
run：run
"""