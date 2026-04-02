
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning) #Due to Chinese text in docstring of pyAgxArm
import cv2

from multimode_gripper import realsense_cam
from multimode_gripper import robot_motion
from multimode_gripper.calibration import camera_calibrate

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
    robot_mot = robot_motion.RobotMotion()
    robot = robot_mot.robot
    robot.set_speed_percent(30)

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
    
    # State machine for calibration process
    STATE_MOVING = 0
    STATE_SETTLING = 1
    STATE_CAPTURING = 2
    state = STATE_MOVING
    settle_start_time = None

    while capture_count < total_poses:
        # Capture frames continuously while moving robot through predefined poses.
        depth_image, color_image = cam.grab_frames()
        if depth_image is None or color_image is None:
            continue

        camera_calibrate.display_image_with_charuco_overlay(color_image.copy(), charuco_board, capture_count)

        if state == STATE_MOVING:
            # Send move command and check if movement is complete
            move_complete = robot_mot.move_non_blocking(poses[capture_count][0])
            if move_complete:
                print(f"Robot reached pose {capture_count + 1}/{total_poses}")
                state = STATE_SETTLING
                settle_start_time = time.time()
                
        elif state == STATE_SETTLING:
            # Wait for robot to settle
            if time.time() - settle_start_time >= 2.0:  # 2 second settling time
                print(f"Robot settled at pose {capture_count + 1}, capturing...")
                state = STATE_CAPTURING
                
        elif state == STATE_CAPTURING:
            # Capture calibration data
            _, charuco_corners, charuco_ids = camera_calibrate.display_image_with_charuco_overlay(
                color_image, charuco_board, capture_count
            )

            flange_pose = robot.get_flange_pose()
            joint_angles = robot.get_joint_angles()
            if flange_pose is None or joint_angles is None:
                print(f"Skipping pose {capture_count + 1}: failed to read robot state.")
                state = STATE_MOVING  # Try again
                continue

            all_charuco_corners.append(charuco_corners.copy())
            all_charuco_ids.append(charuco_ids.copy())
            all_images.append([color_image.copy(), depth_image.copy()])
            read_poses.append([flange_pose.msg.copy(), joint_angles.msg.copy()])

            capture_count += 1
            print(f"Captured pose {capture_count}/{total_poses}")
            
            # Move to next pose
            if capture_count < total_poses:
                state = STATE_MOVING
            else:
                break  # All poses captured

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
