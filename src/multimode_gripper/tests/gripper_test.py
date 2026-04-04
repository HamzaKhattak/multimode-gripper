import argparse
import json
import multiprocessing
import time
import warnings
from pathlib import Path

import cv2

from multimode_gripper import realsense_cam
from multimode_gripper import robot_motion
from multimode_gripper import sensor_camera

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Due to Chinese text in docstring of pyAgxArm


def sensorgrab(stop_event, cap_params, q, save=False, save_path=None, live_view=False):
    # Create camera inside the worker process (required for Windows multiprocessing spawn).
    cam = sensor_camera.SensorCamera()
    try:
        while not stop_event.is_set():
            low_image, high_image = cam.lowhighframecap(*cap_params)
            timestamp = time.time()
            low_image_path = None
            high_image_path = None

            if save and save_path is not None:
                frame_id = int(timestamp * 1000)
                low_image_path = save_path / f"sensor_color_{frame_id}.png"
                high_image_path = save_path / f"sensor_depth_{frame_id}.png"
                cv2.imwrite(str(low_image_path), low_image)
                cv2.imwrite(str(high_image_path), high_image)

            if live_view:
                cv2.imshow("Sensor Color Image", low_image)
                cv2.waitKey(1)

            q.put(
                {
                    "timestamp": timestamp,
                    "sensor_color_path": str(low_image_path) if low_image_path else None,
                    "sensor_depth_path": str(high_image_path) if high_image_path else None,
                }
            )
    finally:
        if hasattr(cam, "cam"):
            cam.cam.release()
        if hasattr(cam, "serial_connection"):
            cam.serial_connection.close()
        if live_view:
            cv2.destroyAllWindows()


def realsensegrab(stop_event, q, save=False, save_path=None, live_view=False):
    # Create camera inside the worker process (required for Windows multiprocessing spawn).
    cam = realsense_cam.RealsenseCam()
    try:
        while not stop_event.is_set():
            depth_image, color_image = cam.grab_frames()
            if depth_image is None or color_image is None:
                continue

            timestamp = time.time()
            color_image_path = None
            depth_image_path = None

            if save and save_path is not None:
                frame_id = int(timestamp * 1000)
                color_image_path = save_path / f"realsense_color_{frame_id}.png"
                depth_image_path = save_path / f"realsense_depth_{frame_id}.png"
                cv2.imwrite(str(color_image_path), color_image)
                cv2.imwrite(str(depth_image_path), depth_image)

            if live_view:
                cv2.imshow("RealSense Color Image", color_image)
                cv2.imshow("RealSense Depth Image", depth_image)
                cv2.waitKey(1)

            q.put(
                {
                    "timestamp": timestamp,
                    "realsense_color_path": str(color_image_path) if color_image_path else None,
                    "realsense_depth_path": str(depth_image_path) if depth_image_path else None,
                }
            )
    finally:
        if hasattr(cam, "pipeline"):
            cam.pipeline.stop()
        if live_view:
            cv2.destroyAllWindows()


def _drain_latest(q):
    latest = None
    while not q.empty():
        latest = q.get()
    return latest


def grab_dataset(pose_path, target_position, target_force, force_threshold, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    q_sensor = multiprocessing.Queue()
    q_realsense = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    # Connect to arm and end effector.
    robot_mot = robot_motion.RobotMotion(end_effector_use=True)
    robot = robot_mot.robot
    robot.set_speed_percent(30)

    with open(pose_path, "r", encoding="utf-8") as pose_file:
        poses = json.load(pose_file)

    # Move to pose and wait for settling.
    robot_mot.move_and_wait(poses[0][0])
    time.sleep(5.0)

    sensor_process = multiprocessing.Process(
        target=sensorgrab,
        args=(stop_event, (50, 200, 1.0, 1.0), q_sensor, True, save_dir, False),
        daemon=True,
    )
    realsense_process = multiprocessing.Process(
        target=realsensegrab,
        args=(stop_event, q_realsense, True, save_dir, False),
        daemon=True,
    )

    sensor_process.start()
    realsense_process.start()

    records = []
    latest_sensor = None
    latest_realsense = None
    start_time = time.time()
    timeout_s = 60.0

    try:
        while True:
            status, current_position, current_force = robot_mot.move_gripper_slowly(
                target_position,
                target_force,
                force_threshold,
                speed=0.1,
                min_move=0.01,
            )

            latest_sensor = _drain_latest(q_sensor) or latest_sensor
            latest_realsense = _drain_latest(q_realsense) or latest_realsense

            records.append(
                {
                    "timestamp": time.time(),
                    "position": current_position,
                    "force": current_force,
                    "sensor": latest_sensor,
                    "realsense": latest_realsense,
                }
            )

            position_reached = abs(current_position - target_position) <= 0.01
            force_reached = current_force >= force_threshold
            if position_reached and force_reached:
                print("Gripper target position and force reached.")
                break

            if status:
                print("Gripper reported completion before both thresholds were satisfied.")
                break

            if time.time() - start_time > timeout_s:
                print(f"Timeout reached ({timeout_s}s). Stopping capture.")
                break

            time.sleep(0.02)
    finally:
        stop_event.set()
        sensor_process.join(timeout=5)
        realsense_process.join(timeout=5)
        if sensor_process.is_alive():
            sensor_process.terminate()
        if realsense_process.is_alive():
            realsense_process.terminate()

    metadata_path = save_dir / "dataset_records.json"
    with open(metadata_path, "w", encoding="utf-8") as records_file:
        json.dump(records, records_file, indent=2)
    print(f"Saved {len(records)} records to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Grab synchronized dataset while closing gripper")
    parser.add_argument(
        "--safe-poses-file",
        type=str,
        required=True,
        help="Path to JSON file containing safe robot poses",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to save dataset files",
    )
    parser.add_argument("--target-position", type=float, required=True, help="Gripper target position")
    parser.add_argument("--target-force", type=float, required=True, help="Gripper commanded force")
    parser.add_argument("--force-threshold", type=float, required=True, help="Stop threshold for force")
    args = parser.parse_args()

    grab_dataset(
        pose_path=args.safe_poses_file,
        target_position=args.target_position,
        target_force=args.target_force,
        force_threshold=args.force_threshold,
        save_dir=args.output_dir,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
