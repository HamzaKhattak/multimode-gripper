import cv2
import numpy as np
import json
import argparse
from PIL import Image
from .. import realsense_cam

def load_charuco_board_from_json(board_config_path: str) -> cv2.aruco.CharucoBoard:
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

def display_image_with_charuco_overlay(color_image, charuco_board, capture_count=0):
    display = color_image.copy()
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(charuco_board.getDictionary(), params)
    corners, ids, _ = detector.detectMarkers(gray)
    charuco_corners, charuco_ids = None, None
    if corners is not None and ids is not None and len(ids) > 0:
        corners = [np.ascontiguousarray(c).copy() for c in corners]
        ids = np.ascontiguousarray(ids).copy()
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, charuco_board
        )

    if corners is not None and ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)

    if charuco_corners is not None and charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)

    charuco_count = 0 if charuco_ids is None else len(charuco_ids)
    
    cv2.putText(display, f"Captures: {capture_count} | c=capture  q=calibrate",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, f"Charuco corners: {charuco_count}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.imshow("Charuco Calibration", display)
    return display, charuco_corners, charuco_ids
    

def calibrate_camera_charuco(board_config_path="charuco_board.json"):
    charuco_board = load_charuco_board_from_json(board_config_path)
    all_charuco_corners = []
    all_charuco_ids = []
    all_images = []
    cam = realsense_cam.RealsenseCam()
    capture_count = 0
    image_size = None

    print("Press 'c' to capture a frame, 'q' to finish and calibrate.")

    while True:
        depth_image, color_image = cam.grab_frames()
        if depth_image is None or color_image is None:
            continue
        
        # Small delay to prevent overwhelming the camera hardware
        #time.sleep(0.01)  # 10ms delay between frame grabs
        
        # Capture image dimensions from first frame
        if image_size is None:
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]
        
        display, charuco_corners, charuco_ids = display_image_with_charuco_overlay(color_image, charuco_board, capture_count)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) >= 4:
                # Make explicit deep copies of image data to completely decouple from camera buffer
                color_copy = color_image.copy()
                depth_copy = depth_image.copy()
                
                # Store the copied data
                all_images.append([color_copy, depth_copy])
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

                capture_count += 1
                print(f"Captured frame {capture_count}")
            else:
                charuco_count = 0 if charuco_ids is None else len(charuco_ids)
                print(f"Insufficient ChArUco detections (charuco={charuco_count}), frame not captured.")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break

    if len(all_charuco_corners) < 1:
        raise ValueError("Not enough valid frames for calibration")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, charuco_board, image_size, None, None
    )
    return camera_matrix, dist_coeffs, all_images, [all_charuco_corners, all_charuco_ids]

def create_and_save_new_board(
    squares_vertically,
    squares_horizontally,
    square_length,
    marker_length,
    im_save_path="charuco_board.png",
    board_config_path="charuco_board.json",
    aruco_dictionary_id=cv2.aruco.DICT_6X6_250,
):
    '''
    Creates a new charuco board with the specified parameters, saves the board image and configuration for later use in calibration
    Note that this prints the board as a raster so generally better to use the online calib.io tool
    '''
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dictionary_id) #DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, DICT_7X7_1000
    board = cv2.aruco.CharucoBoard((squares_horizontally, squares_vertically), square_length, marker_length, dictionary)

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
    cv2.imwrite(im_save_path, img)
    return board, img


def save_calibration_data(save_dir, all_images, calibration_data, camera_matrix, dist_coeffs):
    """Save calibration images and data to disk."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    all_charuco_corners, all_charuco_ids = calibration_data
    
    # Save images as TIFF stacks
    color_images = [Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)) for color, _ in all_images]
    depth_images = [Image.fromarray(depth.astype(np.uint16)) for _, depth in all_images]
    
    color_images[0].save(
        os.path.join(save_dir, "color_images.tiff"),
        save_all=True,
        append_images=color_images[1:]
    )
    
    depth_images[0].save(
        os.path.join(save_dir, "depth_images.tiff"),
        save_all=True,
        append_images=depth_images[1:]
    )
    
    # Save calibration data as JSON
    calibration_dict = {
        "num_frames": len(all_images),
        "charuco_corners": [cc.tolist() if hasattr(cc, 'tolist') else cc for cc in all_charuco_corners],
        "charuco_ids": [ci.tolist() if hasattr(ci, 'tolist') else ci for ci in all_charuco_ids]
    }
    
    with open(os.path.join(save_dir, "calibration_data.json"), "w", encoding="utf-8") as f:
        json.dump(calibration_dict, f, indent=2)

    # Save camera intrinsics in OpenCV YAML format.
    intrinsics_path = os.path.join(save_dir, "camera_intrinsics.yaml")
    fs = cv2.FileStorage(intrinsics_path, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.release()
    
    print(f"Color images saved to: {os.path.join(save_dir, 'color_images.tiff')}")
    print(f"Depth images saved to: {os.path.join(save_dir, 'depth_images.tiff')}")
    print(f"Calibration data saved to: {os.path.join(save_dir, 'calibration_data.json')}")
    print(f"Camera intrinsics saved to: {intrinsics_path}")


def main():
    # Load default board parameters from config file
    import os
    
    config_file = "calibrationdata/boardparams.json"
    board_params = {}
    
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            board_params = json.load(f)
    else:
        raise FileNotFoundError(f"Config file '{config_file}' not found. Please create it with board parameters.")
    
    parser = argparse.ArgumentParser(
        description="Camera calibration tools for charuco board detection"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create board subcommand
    create_parser = subparsers.add_parser(
        "create_board",
        help="Create and save a new charuco board"
    )
    create_parser.add_argument(
        "--image-path",
        type=str,
        default="charuco_board.png",
        help="Path to save the board image (default: charuco_board.png)"
    )
    create_parser.add_argument(
        "--board-config-path",
        type=str,
        default="calibrationdata/charuco_board.json",
        help="Path to save the board configuration (default: charuco_board.json)"
    )

    # Calibrate camera subcommand
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate camera using charuco board"
    )
    calibrate_parser.add_argument(
        "--config-path",
        type=str,
        default="calibrationdata/charuco_board.json",
        help="Path to board configuration (default: charuco_board.json)"
    )
    calibrate_parser.add_argument(
        "--save-dir",
        type=str,
        default="calibrationdata/calibration_results",
        help="Directory to save calibration images and data (default: calibrationdata/calibration_results)"
    )

    args = parser.parse_args()

    if args.command == "create_board":
        dictionary_id = getattr(cv2.aruco, board_params["aruco_dictionary"])
        print(f"Creating charuco board with {board_params['squares_vertically']}x{board_params['squares_horizontally']} squares...")
        print(f"Square length: {board_params['square_length']}mm, Marker length: {board_params['marker_length']}mm")
        print(f"ArUco dictionary: {board_params['aruco_dictionary']}")
        
        board, img = create_and_save_new_board(
            squares_vertically=board_params["squares_vertically"],
            squares_horizontally=board_params["squares_horizontally"],
            square_length=board_params["square_length"],
            marker_length=board_params["marker_length"],
            im_save_path=args.image_path,
            board_config_path=args.board_config_path,
            aruco_dictionary_id=dictionary_id
        )
        
        print(f"✓ Board image saved to: {args.image_path}")
        print(f"✓ Board config saved to: {args.board_config_path}")
    
    elif args.command == "calibrate":
        print(f"Starting camera calibration using board config: {args.config_path}")
        print("Tips:")
        print("  - Press 'c' to capture a frame")
        print("  - Press 'q' to finish capturing and start calibration")
        print("  - Try to capture frames from different angles for better calibration")
        
        camera_matrix, dist_coeffs, all_images, calibration_data = calibrate_camera_charuco(
            board_config_path=args.config_path
        )
        
        save_calibration_data(args.save_dir, all_images, calibration_data, camera_matrix, dist_coeffs)
        
        print(f"✓ Calibration completed with {len(all_images)} frames")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

