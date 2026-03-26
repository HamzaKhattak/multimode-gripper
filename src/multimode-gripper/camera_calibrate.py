import cv2
import numpy as np
import json
import realsense_cam


def load_charuco_board_from_json(board_config_path: string) -> cv2.aruco.CharucoBoard:
    with open(board_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dictionary = cv2.aruco.getPredefinedDictionary(cfg["aruco_dictionary_id"])
    board = cv2.aruco.CharucoBoard(
        (cfg["squares_vertically"], cfg["squares_horizontally"]),
        cfg["square_length"],
        cfg["marker_length"],
        dictionary,
    )
    return board

def display_image_with_charuco_overlay(color_image, charuco_board):
    display = color_image.copy()
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, charuco_board.dictionary)

    if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
        cv2.aruco.drawDetectedMarkers(display, corners, ids)
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, charuco_board
        )
    cv2.putText(display, f"Captures: {capture_count} | c=capture  q=calibrate",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Charuco Calibration", display)
    return display, corners, ids, charuco_corners, charuco_ids

def calibrate_camera_charuco(board_config_path="charuco_board.json"):
    charuco_board = load_charuco_board_from_json(board_config_path)
    all_corners = []
    all_ids = []
    cam = realsense_cam.RealsenseCam()
    capture_count = 0

    print("Press 'c' to capture a frame, 'q' to finish and calibrate.")

    while True:
        depth_image, color_image = cam.grab_frames()
        if depth_image is None or color_image is None:
            continue
        display, corners, ids, charuco_corners, charuco_ids = display_image_with_charuco_overlay(color_image, charuco_board)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if ids is not None and len(ids) > 0:
                all_corners.append(corners)
                all_ids.append(ids)
                capture_count += 1
                print(f"Captured frame {capture_count}")
            else:
                print("No markers detected, frame not captured.")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break

    if len(all_corners) < 1:
        raise ValueError("Not enough valid frames for calibration")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, charuco_board, gray.shape[::-1], None, None
    )
    return camera_matrix, dist_coeffs

def create_and_save_new_board(
    squares_vertically,
    squares_horizontally,
    square_length,
    marker_length,
    save_name,
    board_config_path="charuco_board.json",
    aruco_dictionary_id=cv2.aruco.DICT_5X5_100,
):
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dictionary_id) #DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, DICT_7X7_1000
    board = cv2.aruco.CharucoBoard((squares_vertically, squares_horizontally), square_length, marker_length, dictionary)

    board_cfg = {
        "squares_vertically": squares_vertically,
        "squares_horizontally": squares_horizontally,
        "square_length": square_length,
        "marker_length": marker_length,
        "aruco_dictionary_id": int(aruco_dictionary_id),
    }
    with open(board_config_path, "w", encoding="utf-8") as f:
        json.dump(board_cfg, f, indent=2)

    size_ratio = squares_horizontally / squares_vertically
    img = cv2.aruco.CharucoBoard.generateImage(board, (1000, int(1000*size_ratio)), marginSize=10)
    return board, img

