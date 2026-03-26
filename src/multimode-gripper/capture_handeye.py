import realsense_cam
import json
import argparse
from pathlib import Path

from pyAgxArm import create_agx_arm_config, AgxArmFactory
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning) #Due to Chinese text in docstring of pyAgxArm
import cv2
import camera_calibrate

cfg = create_agx_arm_config(robot="piper", comm="can", channel="can0")
print(cfg)
robot = AgxArmFactory.create_arm(cfg)

robot.connect()
time.sleep(0.5)
print("robotic arm is_ok =", robot.is_ok())






def move_and_wait(robot, flange_pose):
    robot.move_p(flange_pose)
    start_t = time.monotonic()
    time.sleep(0.1)
    while True:
        status = robot.get_arm_status()
        if status is not None and status.msg.motion_status == 0:
            print("done")
            break
        if time.monotonic() - start_t > 20.0:
            print("timeout（20s）")
            break
        time.sleep(0.1)



charuco_board = camera_calibrate.load_charuco_board_from_json(board_config_path)
all_corners = []
all_ids = []
all_charuco_corners = []
all_charuco_ids = []
cam = realsense_cam.RealsenseCam()
capture_count = 0
poses_loc = 'poses.json'
poses = json.load(open(poses_loc, "r", encoding="utf-8"))

read_poses = []

total_poses = len(poses)
move_complete = False

while True:
    depth_image, color_image = cam.grab_frames()
    if depth_image is None or color_image is None:
        continue
    display, corners, ids, charuco_corners, charuco_ids = camera_calibrate.display_image_with_charuco_overlay(color_image, charuco_board)
    move_complete = robot.get_arm_status().msg.motion_status
    if not move_complete:
        robot.movepose(poses[capture_count][0])
    if move_complete:
        capture_count += 1
        time.sleep(1.0) # Add a short delay to ensure the robot has fully settled before capturing the frame
        display, corners, ids, charuco_corners, charuco_ids = camera_calibrate.display_image_with_charuco_overlay(color_image, charuco_board)

        all_corners.append(corners)
        all_ids.append(ids)
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

        read_poses.append([robot.get_flange_pose(), robot.get_joint_angles()])
        print(f"Captured pose {capture_count}/{total_poses}")
         

    key = cv2.waitKey(1) & 0xFF
    if capture_count == total_poses - 1:
        cv2.destroyAllWindows()
        break

'''


#move_and_wait(robot, new_flange_pose)
#move_and_wait(robot, newer_flange_pose)

end_effector.move_gripper(width=0.06, force=1.0)
time.sleep(1.0)

end_effector.move_gripper(width=0.02, force=1.0)
'''