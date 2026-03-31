import realsense_cam
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

from pyAgxArm import create_agx_arm_config, AgxArmFactory
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning) #Due to Chinese text in docstring of pyAgxArm
import cv2
import camera_calibrate

def _to_serializable(values):
    serializable = []
    for value in values:
        if value is None:
            serializable.append(None)
        elif hasattr(value, "tolist"):
            serializable.append(value.tolist())
        else:
            serializable.append(value)
    return serializable


def save_handeye_outputs(output_dir, all_images, all_corners, all_ids, all_charuco_corners, all_charuco_ids, read_poses, input_poses):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = {
        "input_poses": input_poses,
        "read_poses": read_poses,
        "all_corners": _to_serializable(all_corners),
        "all_ids": _to_serializable(all_ids),
        "all_charuco_corners": _to_serializable(all_charuco_corners),
        "all_charuco_ids": _to_serializable(all_charuco_ids),
    }

    dataset_path = output_path / "handeye_capture_data.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    if all_images:
        color_images = [Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)) for color, _ in all_images]
        depth_images = [Image.fromarray(np.asarray(depth, dtype=np.uint16)) for _, depth in all_images]

        color_path = output_path / "color_images.tiff"
        depth_path = output_path / "depth_images.tiff"

        color_images[0].save(str(color_path), save_all=True, append_images=color_images[1:])
        depth_images[0].save(str(depth_path), save_all=True, append_images=depth_images[1:])

        print(f"Saved color TIFF stack to: {color_path}")
        print(f"Saved depth TIFF stack to: {depth_path}")

    print(f"Saved hand-eye JSON data to: {dataset_path}")


def capture_handeye(board_config_path, poses_path):
    # connect to arm
    cfg = create_agx_arm_config(robot="piper", comm="can", channel="can0")
    print(cfg)
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    time.sleep(0.5)
    print("robotic arm is_ok =", robot.is_ok())
    robot.enable()
    robot.set_speed_percent(10)

    # load charuco board configuration and initialize storage arrays
    charuco_board = camera_calibrate.load_charuco_board_from_json(board_config_path)
    all_corners = []
    all_ids = []
    all_charuco_corners = []
    all_charuco_ids = []
    all_images = []

    cam = realsense_cam.RealsenseCam()

    # load poses to move the robot to
    with open(poses_path, "r", encoding="utf-8") as pose_file:
        poses = json.load(pose_file)

    read_poses = []  # Store robot's actual poses for hand-eye calibration
    total_poses = len(poses)
    capture_count = 0
    move_commanded = False
    motion_start_t = None

    while capture_count < total_poses:
        # Capture frames continuously while moving robot through predefined poses.
        depth_image, color_image = cam.grab_frames()
        if depth_image is None or color_image is None:
            continue

        camera_calibrate.display_image_with_charuco_overlay(color_image, charuco_board, capture_count)
        if not move_commanded:
            target_pose = poses[capture_count][0] if isinstance(poses[capture_count], list) and len(poses[capture_count]) > 0 else poses[capture_count]
            robot.move_l(target_pose)
            move_commanded = True
            motion_start_t = time.monotonic()
            continue

        arm_status = robot.get_arm_status()
        if arm_status is None:
            continue

        move_complete = arm_status.msg.motion_status == 0
        if not move_complete:
            if motion_start_t is not None and (time.monotonic() - motion_start_t) > 20.0:
                print(f"Motion timeout at pose {capture_count + 1}, retrying move command.")
                move_commanded = False
            continue

        time.sleep(1.0)  # Give the robot time to settle before capture.
        _ , charuco_corners, charuco_ids = camera_calibrate.display_image_with_charuco_overlay(
            color_image, charuco_board, capture_count
        )

        flange_pose = robot.get_flange_pose()
        joint_angles = robot.get_joint_angles()
        if flange_pose is None or joint_angles is None:
            print(f"Skipping pose {capture_count + 1}: failed to read robot state.")
            move_commanded = False
            motion_start_t = None
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        all_images.append([color_image, depth_image])
        read_poses.append([flange_pose.msg, joint_angles.msg])

        capture_count += 1
        move_commanded = False
        motion_start_t = None
        print(f"Captured pose {capture_count}/{total_poses}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return all_images, all_corners, all_ids, all_charuco_corners, all_charuco_ids, read_poses, poses


def main():
    parser = argparse.ArgumentParser(description="Capture hand-eye calibration dataset")
    parser.add_argument(
        "--board-params-json",
        type=str,
        required=True,
        help="Path to board parameters JSON used to load ChArUco board",
    )
    parser.add_argument(
        "--poses-file",
        type=str,
        required=True,
        help="Path to input robot poses JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="calibrationdata/handeye_capture",
        help="Directory where JSON dataset and TIFF stacks are written",
    )
    args = parser.parse_args()

    (
        all_images,
        all_corners,
        all_ids,
        all_charuco_corners,
        all_charuco_ids,
        read_poses,
        input_poses,
    ) = capture_handeye(args.board_params_json, args.poses_file)

    save_handeye_outputs(
        args.output_dir,
        all_images,
        all_corners,
        all_ids,
        all_charuco_corners,
        all_charuco_ids,
        read_poses,
        input_poses,
    )


if __name__ == "__main__":
    main()
