'''
This script is used to teach the robot the pose of the gripper. The user can move the robot to the desired range of poses, and then save the poses to a file. 
The saved poses can be used later for camera calibration and other tasks.
'''
from pyAgxArm import create_agx_arm_config, AgxArmFactory
import warnings
import time
import realsense_cam
import json
import argparse
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning) #Due to Chinese text in docstring of pyAgxArm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a series of poses and save to a json file. These poses can be used for camera calibration and other tasks."
        )
    )
    parser.add_argument(
        "-- output-json",
        type=Path,
        required=True,
        help="What file to save the poses to",
    )
    parser.add_argument(
        "-- overwrite-existing",
        type = bool,
        required=False,
        default=False,
        help="Overwrite the existing JSON file if it exists."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = args.output_json
    '''
    open and connect robot and camera
    '''
    cfg = create_agx_arm_config(robot="piper", comm="can", channel="can0")
    print(cfg)
    robot = AgxArmFactory.create_arm(cfg)

    robot.connect()
    cam = realsense_cam.RealsenseCam()
    cam.display_realsense()
    '''
    Capture the series of poses
    '''
    poses = [['flange_pose', 'joint_states']]

    print("Move the robot to the desired pose and press 's' to save the pose, 'q' to finish.")
    while True:
        key = input("Press 's' to save the current pose, 'q' to finish: ")
        if key == 's':
            flange_pose = robot.get_flange_pose()
            joint_states = robot.get_joint_angles()
            if flange_pose is not None:
                poses.append([flange_pose, joint_states])
                print(f"Pose saved  ")
            else:
                print("Failed to get flange pose.")
        elif key == 'q':
            break
        else:
            print("Invalid input. Please press 's' to save or 'q' to finish.")

        with open(output_file, "w") as f:
            json.dump(poses, f, indent=4)

        print(f"Saved {len(poses)} poses.")


if __name__ == "__main__":
    main()