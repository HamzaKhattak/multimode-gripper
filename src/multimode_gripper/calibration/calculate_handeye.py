"""
calculate_handeye.py

Computes the hand-eye (camera-to-gripper) transformation from a hand-eye
capture dataset produced by capture_handeye.py and camera intrinsics produced
by camera_calibrate.py.

The result is a 4x4 homogeneous matrix  T_cam2gripper  that maps a 3-D point
expressed in the RealSense camera frame into the robot flange frame.

A helper function  camera_point_to_robot_pose  chains that transform with the
live gripper-to-base transform so its output can be fed directly to
robot.move_l (or RobotMotion.move_and_wait / move_non_blocking).

Typical usage
-------------
# --- offline: run once after capture ---
python calculate_handeye.py \
    --capture-data  calibrationdata/handeye_capture/handeye_capture_data.json \
    --intrinsics    calibrationdata/calibration_results/camera_intrinsics.yaml \
    --board-params-json calibrationdata/boardparams.json \
    --output        calibrationdata/handeye_result/T_cam2gripper.json

# --- online: convert a detected object point to a robot target ---
from calculate_handeye import load_handeye_result, camera_point_to_robot_pose
T = load_handeye_result("calibrationdata/handeye_result/T_cam2gripper.json")
target = camera_point_to_robot_pose(point_in_camera_frame, T, robot.get_flange_pose().msg)
robot.move_l(target)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_camera_intrinsics(intrinsics_yaml: str):
    """Load camera_matrix and dist_coeffs from an OpenCV YAML file."""
    fs = cv2.FileStorage(intrinsics_yaml, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()
    if camera_matrix is None or dist_coeffs is None:
        raise ValueError(f"Could not read intrinsics from {intrinsics_yaml}")
    return camera_matrix, dist_coeffs


def _load_charuco_board(board_config_path: str) -> cv2.aruco.CharucoBoard:
    """Reconstruct a CharucoBoard from the JSON parameters file."""
    with open(board_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    dictionary = cv2.aruco.getPredefinedDictionary(cfg["aruco_dictionary_id"])
    board = cv2.aruco.CharucoBoard(
        (cfg["squares_horizontally"], cfg["squares_vertically"]),
        float(cfg["square_length"]),
        float(cfg["marker_length"]),
        dictionary,
    )
    return board


def _pose6d_to_matrix(pose6d) -> np.ndarray:
    """
    Convert a [x, y, z, rx, ry, rz] flange pose to a 4x4 homogeneous matrix.

    The rotation part [rx, ry, rz] is treated as a Rodrigues (axis-angle)
    vector, which is the convention used by pyAgxArm for the Piper arm.
    """
    pose6d = np.asarray(pose6d, dtype=np.float64).flatten()
    rvec = pose6d[3:6]
    tvec = pose6d[0:3]
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T


def _matrix_to_pose6d(T: np.ndarray) -> list:
    """
    Convert a 4x4 homogeneous matrix back to [x, y, z, rx, ry, rz].

    The rotation is encoded as a Rodrigues vector to match pyAgxArm's
    move_l pose format.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    rvec = rvec.flatten()
    return [float(t[0]), float(t[1]), float(t[2]),
            float(rvec[0]), float(rvec[1]), float(rvec[2])]


# ---------------------------------------------------------------------------
# Core calibration
# ---------------------------------------------------------------------------

def calculate_handeye(
    capture_data_json: str,
    intrinsics_yaml: str,
    board_config_json: str,
    method: int = cv2.CALIB_HAND_EYE_TSAI,
) -> np.ndarray:
    """
    Calculate the hand-eye (camera-to-gripper) 4x4 homogeneous transform.

    Parameters
    ----------
    capture_data_json : str
        Path to ``handeye_capture_data.json`` produced by capture_handeye.py.
    intrinsics_yaml : str
        Path to ``camera_intrinsics.yaml`` produced by camera_calibrate.py.
    board_config_json : str
        Path to the ChArUco board parameters JSON used during capture.
    method : int
        OpenCV hand-eye calibration method (default: TSAI).

    Returns
    -------
    T_cam2gripper : np.ndarray, shape (4, 4)
        Homogeneous matrix mapping a point in the RealSense camera frame to
        the robot flange (gripper) frame.
    """
    camera_matrix, dist_coeffs = _load_camera_intrinsics(intrinsics_yaml)
    charuco_board = _load_charuco_board(board_config_json)

    with open(capture_data_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    read_poses = data["read_poses"]               # list of [flange_pose_6d, joint_angles]
    raw_charuco_corners = data["all_charuco_corners"]
    raw_charuco_ids = data["all_charuco_ids"]

    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []
    skipped = 0

    for i, (raw_corners, raw_ids, read_pose) in enumerate(
        zip(raw_charuco_corners, raw_charuco_ids, read_poses)
    ):
        if raw_corners is None or raw_ids is None:
            print(f"  Pose {i}: no charuco detections, skipping.")
            skipped += 1
            continue

        charuco_corners = np.array(raw_corners, dtype=np.float32)
        charuco_ids = np.array(raw_ids, dtype=np.int32)

        # estimatePoseCharucoBoard expects ids shaped (N, 1)
        if charuco_ids.ndim == 1:
            charuco_ids = charuco_ids.reshape(-1, 1)

        if len(charuco_ids) < 4:
            print(f"  Pose {i}: only {len(charuco_ids)} charuco corners, need ≥4, skipping.")
            skipped += 1
            continue

        # Estimate the board (target) pose in the camera frame
        rvec_init = np.zeros((3, 1), dtype=np.float64)
        tvec_init = np.zeros((3, 1), dtype=np.float64)
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, charuco_board,
            camera_matrix, dist_coeffs, rvec_init, tvec_init,
        )
        if not ok:
            print(f"  Pose {i}: estimatePoseCharucoBoard failed, skipping.")
            skipped += 1
            continue

        R_board, _ = cv2.Rodrigues(rvec)
        R_target2cam_list.append(R_board)
        t_target2cam_list.append(tvec.flatten())

        # Gripper (flange) pose in robot base frame
        flange_pose_6d = read_pose[0]             # [x, y, z, rx, ry, rz]
        T_gripper2base = _pose6d_to_matrix(flange_pose_6d)
        R_gripper2base_list.append(T_gripper2base[:3, :3])
        t_gripper2base_list.append(T_gripper2base[:3, 3])

    n_valid = len(R_gripper2base_list)
    print(f"Hand-eye calibration: {n_valid} valid pose pairs ({skipped} skipped).")

    if n_valid < 3:
        raise ValueError(
            f"Need at least 3 valid pose pairs for hand-eye calibration, got {n_valid}."
        )

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list,
        R_target2cam_list, t_target2cam_list,
        method=method,
    )

    T_cam2gripper = np.eye(4, dtype=np.float64)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
    return T_cam2gripper


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_handeye_result(output_path: str, T_cam2gripper: np.ndarray) -> None:
    """Save T_cam2gripper to a JSON file."""
    result = {"T_cam2gripper": T_cam2gripper.tolist()}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved hand-eye transform to {output_path}")


def load_handeye_result(result_json: str) -> np.ndarray:
    """Load a previously saved T_cam2gripper from JSON."""
    with open(result_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["T_cam2gripper"], dtype=np.float64)


# ---------------------------------------------------------------------------
# Runtime utility: camera-frame point → robot move_l target
# ---------------------------------------------------------------------------

def camera_point_to_robot_pose(
    point_camera,
    T_cam2gripper: np.ndarray,
    flange_pose_6d,
) -> list:
    """
    Convert a 3-D point in the camera frame to a robot flange pose suitable
    for ``robot.move_l``.

    The chain applied is:
        p_base = T_gripper2base @ T_cam2gripper @ p_cam

    The orientation of the resulting target pose is inherited from the current
    flange pose so the wrist attitude is preserved while only the position
    changes.  If you need a different end-effector orientation, replace
    ``T_target[:3, :3]`` with the desired rotation matrix before returning.

    Parameters
    ----------
    point_camera : array-like, shape (3,)
        Target point in the RealSense camera frame (metres).
    T_cam2gripper : np.ndarray, shape (4, 4)
        The hand-eye transform returned by ``calculate_handeye`` or
        ``load_handeye_result``.
    flange_pose_6d : array-like, shape (6,)
        Current robot flange pose ``[x, y, z, rx, ry, rz]`` from
        ``robot.get_flange_pose().msg``.

    Returns
    -------
    list[float], length 6
        ``[x, y, z, rx, ry, rz]`` in the robot base frame ready for
        ``robot.move_l``.
    """
    p_cam = np.array(point_camera, dtype=np.float64).flatten()
    p_cam_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])

    # Camera frame → gripper (flange) frame
    p_gripper_h = T_cam2gripper @ p_cam_h

    # Gripper frame → robot base frame
    T_gripper2base = _pose6d_to_matrix(flange_pose_6d)
    p_base_h = T_gripper2base @ p_gripper_h

    # Build output pose: keep current wrist orientation, update position only
    T_target = T_gripper2base.copy()
    T_target[:3, 3] = p_base_h[:3]
    return _matrix_to_pose6d(T_target)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_METHOD_MAP = {
    "tsai":       cv2.CALIB_HAND_EYE_TSAI,
    "park":       cv2.CALIB_HAND_EYE_PARK,
    "horaud":     cv2.CALIB_HAND_EYE_HORAUD,
    "andreff":    cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def main():
    parser = argparse.ArgumentParser(
        description="Calculate hand-eye calibration from a capture dataset"
    )
    parser.add_argument(
        "--capture-data",
        required=True,
        help="Path to handeye_capture_data.json from capture_handeye.py",
    )
    parser.add_argument(
        "--intrinsics",
        required=True,
        help="Path to camera_intrinsics.yaml from camera_calibrate.py",
    )
    parser.add_argument(
        "--board-params-json",
        required=True,
        help="Path to board parameters JSON used during capture",
    )
    parser.add_argument(
        "--output",
        default="calibrationdata/handeye_result/T_cam2gripper.json",
        help="Output path for the resulting transform JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--method",
        default="tsai",
        choices=list(_METHOD_MAP.keys()),
        help="Hand-eye calibration method (default: tsai)",
    )
    args = parser.parse_args()

    T_cam2gripper = calculate_handeye(
        args.capture_data,
        args.intrinsics,
        args.board_params_json,
        method=_METHOD_MAP[args.method],
    )

    print("T_cam2gripper =\n", T_cam2gripper)
    save_handeye_result(args.output, T_cam2gripper)


if __name__ == "__main__":
    main()
