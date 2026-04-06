import pyrealsense2 as rs
import cv2
import numpy as np
import time


class RealsenseCam:
    _RESET_WAIT_S = 5.0       # Time to wait after a hardware reset for USB re-enumeration.
    _FIRST_FRAME_TIMEOUT_S = 8.0   # How long to wait for the first valid frame after pipeline.start().
    _FIRST_FRAME_TIMEOUT_S_AFTER_RESET = 12.0  # Longer budget after a hardware reset.

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self._start_pipeline()

    def _enable_streams(self) -> None:
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    def _wait_for_first_frame(self, timeout_s: float) -> bool:
        '''
        Block until a coherent depth+colour frameset is received or the timeout
        expires.  Returns True if a valid frame arrived, False otherwise.
        '''
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            remaining_ms = max(100, int((deadline - time.time()) * 1000))
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=remaining_ms)
                if frames.get_depth_frame() and frames.get_color_frame():
                    return True
            except Exception:
                pass
        return False

    def _start_pipeline(self) -> None:
        '''
        Start the pipeline and verify that frames are actually flowing.
        pipeline.start() succeeding is not sufficient — the camera may be in a
        state where it opens without error but never delivers frames.
        If that happens (or if start() itself raises), perform a hardware reset
        and try once more with a longer frame-wait budget.
        '''
        try:
            self.pipeline.start(self.config)
            if self._wait_for_first_frame(self._FIRST_FRAME_TIMEOUT_S):
                return  # Camera is healthy and streaming.
            print("RealSense pipeline started but no frames received — resetting device and retrying...")
        except Exception as exc:
            print(f"RealSense pipeline.start() failed ({type(exc).__name__}: {exc}) — resetting device and retrying...")

        # Stop current pipeline before resetting so the device is fully released.
        try:
            self.pipeline.stop()
        except Exception:
            pass

        self._hardware_reset()

        # Re-create pipeline and config — stale handles cause a second failure.
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self._enable_streams()
        self.pipeline.start(self.config)
        if not self._wait_for_first_frame(self._FIRST_FRAME_TIMEOUT_S_AFTER_RESET):
            raise RuntimeError(
                "RealSense camera failed to produce frames after hardware reset. "
                "Check USB connection and that no other process holds the device."
            )

    def _hardware_reset(self) -> None:
        '''
        Reset all connected RealSense devices and wait long enough for USB
        re-enumeration to complete before the caller opens the pipeline again.
        '''
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                dev.hardware_reset()
            time.sleep(self._RESET_WAIT_S)
        except Exception:
            # Reset is best-effort only.
            pass

    def grab_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        '''
        Grabs a pair of depth and color frames from the RealSense camera. Returns None, None if frames cannot be grabbed.
        '''
        try:
            # Use wait_for_frames with short timeout to avoid blocking
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None, None

            # Make copies of the data to ensure frames are immediately freed from buffer
            depth_image = np.asanyarray(depth_frame.get_data()).copy()
            color_image = np.asanyarray(color_frame.get_data()).copy()
            
            # Frames are now released automatically as frames object goes out of scope
            return depth_image, color_image
        except RuntimeError:
            # Timeout or error - return None
            return None, None

    def _normalizeImg(self, img, low, high):
        '''
        Normalizes the input image to the range [0, 255] based on the provided low and high values.'''
        imgClip = np.clip(img, low, high)
        maxVal = np.max(imgClip)
        minVal = np.min(imgClip)
        return np.uint8((255.)/(maxVal-minVal)*(imgClip-maxVal)+255.)


    def display_realsense(self):
        ''' Continuously grabs frames from the RealSense camera and displays the color and depth streams. Press 'q' to quit. 
        '''
        try:
            while True:
                # Wait for a coherent pair of frames
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Display the images
                cv2.imshow('Color Stream', color_image)
                cv2.imshow('Depth Stream', self._normalizeImg(depth_image, 500, 850))

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def realsensegrab(stop_event, q, save=False, save_path=None, live_view=False):
    # Create camera inside the worker process (required for Windows multiprocessing spawn).
    cam = RealsenseCam()
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