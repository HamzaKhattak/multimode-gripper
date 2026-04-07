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


def _load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _drain_latest(q):
    latest = None
    while not q.empty():
        latest = q.get()
    return latest


def grab_dataset(
    poses,
    target_position,
    target_force,
    force_threshold,
    sensor_params,
    save_dir,
    camera_serial,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    q_sensor = multiprocessing.Queue()
    q_realsense = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    # Connect to arm and end effector.
    robot_mot = robot_motion.RobotMotion(end_effector_use=True)
    robot = robot_mot.robot
    robot.set_speed_percent(30)

    sensor_save_fps = None
    sensor_unused_frame_mode = "discard"
    realsense_save_fps = None
    realsense_unused_frame_mode = "discard"
    if isinstance(sensor_params, dict):
        sensor_save_fps = sensor_params.get("save_fps", None)
        sensor_unused_frame_mode = sensor_params.get("unused_frame_mode", "discard")
        realsense_save_fps = sensor_params.get("realsense_save_fps", None)
        realsense_unused_frame_mode = sensor_params.get("realsense_unused_frame_mode", "discard")
 
    
    
    robot_mot.move_and_wait(poses[0][0])  # Move to the initial pose before starting capture to avoid including the movement to the initial pose in the dataset. 
    robot_mot.open_gripper()  # Open the gripper before starting capture so the closing motion is fully captured in the dataset.  
    sensor_process = multiprocessing.Process(
        target=sensor_camera.sensorgrab,
        args=(
            stop_event,
            sensor_params,
            q_sensor,
            True,
            save_dir,
            False,
            camera_serial,
            sensor_save_fps,
            sensor_unused_frame_mode,
        ),
        daemon=True,
    )
    realsense_process = multiprocessing.Process(
        target=realsense_cam.realsensegrab,
        args=(
            stop_event,
            q_realsense,
            True,
            save_dir,
            False,
            realsense_save_fps,
            realsense_unused_frame_mode,
        ),
        daemon=True,
    )

    sensor_process.start()
    realsense_process.start()

    records = []
    latest_sensor = None
    latest_realsense = None
    release_timeout_s = 60.0
    startup_timeout_s = 20.0
    move_to_pose_timeout_s = 30.0
    timeout_s = 60.0

    try:
        # Warm up both capture workers before any robot movement starts.
        startup_t0 = time.time()
        while latest_sensor is None or latest_realsense is None:
            latest_sensor = _drain_latest(q_sensor) or latest_sensor
            latest_realsense = _drain_latest(q_realsense) or latest_realsense

            if latest_sensor is not None and latest_realsense is not None:
                break

            # Detect silent child-process crashes immediately rather than waiting
            # for the full startup timeout. A dead process will never put frames
            # into its queue, so there is no value in continuing to wait.
            if not sensor_process.is_alive() and latest_sensor is None:
                raise RuntimeError(
                    f"Sensor camera process exited unexpectedly (exitcode={sensor_process.exitcode}) "
                    "before producing any frames."
                )
            if not realsense_process.is_alive() and latest_realsense is None:
                raise RuntimeError(
                    f"RealSense process exited unexpectedly (exitcode={realsense_process.exitcode}) "
                    "before producing any frames."
                )

            if time.time() - startup_t0 > startup_timeout_s:
                sensor_alive = sensor_process.is_alive()
                realsense_alive = realsense_process.is_alive()
                raise TimeoutError(
                    f"Camera worker startup timeout ({startup_timeout_s}s). "
                    f"sensor_got_frame={latest_sensor is not None} (alive={sensor_alive}), "
                    f"realsense_got_frame={latest_realsense is not None} (alive={realsense_alive})."
                )

            time.sleep(0.05)

        # Start moving to the initial arm pose only after capture is active.
        move_start_t = time.time()
        while True:
            pose_reached = robot_mot.move_non_blocking(poses[1][0])

            latest_sensor = _drain_latest(q_sensor) or latest_sensor
            latest_realsense = _drain_latest(q_realsense) or latest_realsense

            records.append(
                {
                    "timestamp": time.time(),
                    "phase": "move_to_start_pose",
                    "position": None,
                    "force": None,
                    "sensor": latest_sensor,
                    "realsense": latest_realsense,
                }
            )

            if pose_reached:
                break

            if time.time() - move_start_t > move_to_pose_timeout_s:
                print(f"Timeout reached ({move_to_pose_timeout_s}s) while moving to initial pose.")
                break

            time.sleep(0.02)

        # Remember the current opening so the release trajectory can be captured too.
        release_target_position, _ = robot_mot.get_gripper_wf()

        start_time = time.time()
        while True:
            status, current_position, current_force = robot_mot.grab_slowly(
                target_position,
                target_force,
                force_threshold,
                speed=0.001,
                min_move=0.0005,
            )

            latest_sensor = _drain_latest(q_sensor) or latest_sensor
            latest_realsense = _drain_latest(q_realsense) or latest_realsense

            records.append(
                {
                    "timestamp": time.time(),
                    "phase": "close_gripper",
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

        # Re-open the gripper to the original opening so release is included in the dataset.
        release_start_time = time.time()
        while True:
            status, current_position, current_force = robot_mot.release_slowly(
                release_force_threshold=.1,
                slow_speed=0.001,
                fast_speed=0.005,
                min_move=0.0005,
                target_position=release_target_position,
            )

            latest_sensor = _drain_latest(q_sensor) or latest_sensor
            latest_realsense = _drain_latest(q_realsense) or latest_realsense

            records.append(
                {
                    "timestamp": time.time(),
                    "phase": "open_gripper",
                    "position": current_position,
                    "force": current_force,
                    "sensor": latest_sensor,
                    "realsense": latest_realsense,
                }
            )

            release_position_reached = abs(current_position - release_target_position) <= 0.01
            if release_position_reached:
                print("Gripper returned to release position.")
                break

            if status:
                print("Gripper reported completion while opening.")
                break

            if time.time() - release_start_time > release_timeout_s:
                print(f"Timeout reached ({release_timeout_s}s) while opening gripper.")
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
    robot_mot.move_and_wait(poses[0][0])  # Move back to the initial pose at the end of the test.
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
    parser.add_argument("--camera-serial", type=str, required=True, help="Serial number of the non-RealSense sensor camera")
    parser.add_argument("--cap-paramfile", type=str, required=False, help="Path to JSON file containing camera capture parameters")
    args = parser.parse_args()
    cap_params = _load_json_file(args.cap_paramfile) if args.cap_paramfile else (100, 200, 1.0, 1.0, 'white')
    poses = _load_json_file(args.safe_poses_file)
    grab_dataset(
        poses=poses,
        target_position=args.target_position,
        target_force=args.target_force,
        force_threshold=args.force_threshold,
        sensor_params=cap_params,
        save_dir=args.output_dir,
        camera_serial=args.camera_serial,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Use spawn so child processes start with a clean interpreter and cannot
    # inherit open V4L2 / RealSense device handles from the parent process.
    multiprocessing.set_start_method("spawn")
    main()
