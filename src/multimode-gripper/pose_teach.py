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
import cv2
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
    '''
    Capture the series of poses
    '''
    poses = []
    pose_num = 1
    print("Move the robot to the desired pose and press 's' to save the pose, 'q' to finish.")
    while True:
        depth_image, color_image = cam.grab_frames()
        display = color_image.copy()
        flange_pose = robot.get_flange_pose()
        joint_states = robot.get_joint_angles()
        if depth_image is None or color_image is None or flange_pose is None or joint_states is None:
            continue
        cv2.imshow("Charuco Calibration", display)
        cv2.putText(display, f"positions (x,y,z,thet,phi,psi): {flange_pose}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if flange_pose is not None:
                poses.append([flange_pose, joint_states])
                print(f"Pose {pose_num} captured.")
                pose_num += 1
            else:
                print("Failed to get flange pose.")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break

    with open(output_file, "w") as f:
        json.dump(poses, f, indent=4)

    print(f"Saved {len(poses)} poses.")


if __name__ == "__main__":
    main()