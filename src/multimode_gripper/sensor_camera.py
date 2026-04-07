import cv2
import numpy as np
import serial
from serial.tools import list_ports
import time

from multimode_gripper.utils.camera_check import open_camera_by_serial

class SensorCamera:
    DEVICE_CODE = "RGBBL-001"

    def __init__(
        self,
        camera_serial: str,
        serial_port="/dev/ttyUSB0",
        baud_rate=115200,
        capwidth=1280,
        capheight=720,
        exposuretime=100,
        fps: float = 30,
    ):
        '''
        Initializes the camera and the serial connection to the microcontroller controlling the LED backlight.
         The camera is configured with the specified resolution and exposure time (check camera docs for allowed values)
        '''
        self.cam = open_camera_by_serial(camera_serial, width=capwidth, height=capheight, fps=fps)
        self.cam.set(cv2.CAP_PROP_SETTINGS, 0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,capwidth)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,capheight)
        self.cam.set(cv2.CAP_PROP_FPS, fps)
        ret_val , cap_for_exposure = self.cam.read() #Need to call cam.read() first before fiddling with the exposure
        self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, .75) #For some reason auto exposure off is 0.25 and 0.75 is on/off
        self.cam.set(cv2.CAP_PROP_GAIN,1)
        self.cam.set(cv2.CAP_PROP_EXPOSURE,exposuretime) #exposure is in ms or something for the ubuntu api
        #White Balance not working currently
        self.cam.set(cv2.CAP_PROP_AUTO_WB,3) #Off is 1 for some reason, 3 is on
        self.cam.set(cv2.CAP_PROP_AUTO_WB,1) #Off is 1 for some reason, 3 is on
        self.cam.set(cv2.CAP_PROP_WB_TEMPERATURE,2500)
        # Best effort: keep internal capture queue short to reduce stale frames.
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.serial_connection = self._connect_rgb_backlight(serial_port, baud_rate)

    def _frame_interval_s(self) -> float:
        fps = self.cam.get(cv2.CAP_PROP_FPS)
        fps_interval_s = 1.0 / fps if fps > 0 else 1.0 / 30.0

        # In this backend exposure is configured in milliseconds.
        exposure_ms = self.cam.get(cv2.CAP_PROP_EXPOSURE)
        exposure_interval_s = exposure_ms / 1000.0 if exposure_ms > 0 else 0.0

        return max(fps_interval_s, exposure_interval_s)

    def _capture_fresh_frame(self, settle_frames: int = 2) -> np.ndarray:
        '''
        Capture a frame that is very likely to reflect the latest camera/light state
        by draining queued frames for at least N frame intervals.
        '''
        frame_interval = self._frame_interval_s()
        deadline = time.perf_counter() + (max(1, settle_frames) * frame_interval)

        # Drain queued frames while new ones arrive.
        while time.perf_counter() < deadline:
            if not self.cam.grab():
                break

        ret, frame = self.cam.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from sensor camera.")
        return frame


    def _try_handshake(self, port: str, baud_rate: int, handshake_timeout: float = 5.0):
        '''
        Tries to establish a serial connection and perform a handshake by waiting for the expected DEVICE_CODE.
        Returns the serial connection if successful, or None if the handshake fails.'''
        try:
            connection = serial.Serial(port, baud_rate, timeout=0.25, write_timeout=1)
        except serial.SerialException:
            return None

        # Many MCU boards reset when the serial port opens.
        time.sleep(2.0)
        connection.reset_input_buffer()

        deadline = time.time() + handshake_timeout
        while time.time() < deadline:
            line = connection.readline().decode("utf-8", errors="ignore").strip()
            if line == self.DEVICE_CODE:
                # Acknowledge and set a white default LED state.
                connection.write(b"1\n")
                return connection

        connection.close()
        return None


    def _connect_rgb_backlight(self, preferred_port: str, baud_rate: int):
        '''
        Attempts to connect to the RGB backlight controller by trying the preferred port first, then scanning all available ports.'''
        candidate_ports = []

        if preferred_port:
            candidate_ports.append(preferred_port)

        for port_info in list_ports.comports():
            if port_info.device not in candidate_ports:
                candidate_ports.append(port_info.device)

        for port in candidate_ports:
            connection = self._try_handshake(port, baud_rate)
            if connection is not None:
                return connection

        raise RuntimeError(
            f"Could not find RGB backlight controller. Expected handshake '{self.DEVICE_CODE}' on ports: {candidate_ports}"
        )


    def lowhighframecap(self, lowtime: int, hightime: int, lowgain: float = 1.0, highgain: float = 1.0, light_type: str = 'white') -> tuple[np.ndarray, np.ndarray]:
        '''
        Captures two frames from the camera, one with low exposure and one with high exposure.
          The microcontroller is used to switch the whether the LED backlight is on or off.
        '''
        c0 = b'0'
        if light_type == 'white':
            c1 = b'1'  # White LEDs on
        elif light_type == 'rainbow':
            c1 = b'r'  # Rainbow LEDs on
        elif light_type == 'split':
            c1 = b's'  # Split LEDs on
        else:
            raise ValueError(f"Invalid light_type '{light_type}'. Expected 'white', 'rainbow', or 'split'.")
        
        self.serial_connection.write(c0)  # Turn LEDs off for the low-exposure frame
        self.cam.set(cv2.CAP_PROP_EXPOSURE,lowtime) #exposure is in ms or something for the ubuntu api
        self.cam.set(cv2.CAP_PROP_GAIN,lowgain)
        frame1 = self._capture_fresh_frame()

        self.serial_connection.write(c1)  # Turn LEDs on for the high-exposure frame
        self.cam.set(cv2.CAP_PROP_EXPOSURE,hightime) #exposure is in ms or something for the ubuntu api
        self.cam.set(cv2.CAP_PROP_GAIN,highgain)
        frame2 = self._capture_fresh_frame()
        return frame1, frame2

    def normalframecap(self) -> np.ndarray:
        '''
        Captures a single frame from the camera
        '''
        return self._capture_fresh_frame(settle_frames=1)

def sensorgrab(
    stop_event,
    cap_params,
    q,
    save=False,
    save_path=None,
    live_view=False,
    camera_serial: str | None = None,
    save_fps: float | None = None,
    unused_frame_mode: str = "discard",
):
    # Create camera inside the worker process (required for Windows multiprocessing spawn).
    if camera_serial is None:
        raise ValueError("sensorgrab requires a camera_serial value.")

    cam = SensorCamera(camera_serial=camera_serial)
    try:
        if isinstance(cap_params, dict):
            cap_args = (
                cap_params.get("lowtime", 100),
                cap_params.get("hightime", 200),
                cap_params.get("lowgain", 1.0),
                cap_params.get("highgain", 1.0),
                cap_params.get("light_type", "white"),
            )
            # Allow cap params to override defaults when provided.
            save_fps = cap_params.get("save_fps", save_fps)
            unused_frame_mode = cap_params.get("unused_frame_mode", unused_frame_mode)
        else:
            cap_args = tuple(cap_params)

        unused_frame_mode = str(unused_frame_mode).lower()

        if save_fps is not None:
            save_fps = float(save_fps)
            if save_fps <= 0:
                raise ValueError("save_fps must be > 0 when provided.")

        if unused_frame_mode not in {"discard", "average"}:
            raise ValueError("unused_frame_mode must be either 'discard' or 'average'.")

        save_interval_s = (1.0 / save_fps) if (save and save_fps is not None) else None
        next_output_t = time.perf_counter() if save_interval_s is not None else None

        pending_low_latest = None
        pending_high_latest = None
        pending_low_acc = None
        pending_high_acc = None
        pending_count = 0

        while not stop_event.is_set():
            low_image, high_image = cam.lowhighframecap(*cap_args)
            now_perf = time.perf_counter()
            out_low = low_image
            out_high = high_image

            if save_interval_s is not None:
                assert next_output_t is not None

                if unused_frame_mode == "discard":
                    pending_low_latest = low_image
                    pending_high_latest = high_image
                else:
                    if pending_low_acc is None:
                        pending_low_acc = low_image.astype(np.float32)
                        pending_high_acc = high_image.astype(np.float32)
                    else:
                        pending_low_acc += low_image
                        pending_high_acc += high_image
                pending_count += 1

                if now_perf < next_output_t:
                    if live_view:
                        cv2.imshow("Sensor Low Exposure Image", low_image)
                        cv2.waitKey(1)
                    continue

                if unused_frame_mode == "discard":
                    if pending_low_latest is None or pending_high_latest is None:
                        continue
                    out_low = pending_low_latest
                    out_high = pending_high_latest
                else:
                    if pending_low_acc is None or pending_high_acc is None or pending_count == 0:
                        continue
                    out_low = np.clip(pending_low_acc / pending_count, 0, 255).astype(np.uint8)
                    out_high = np.clip(pending_high_acc / pending_count, 0, 255).astype(np.uint8)

                pending_low_latest = None
                pending_high_latest = None
                pending_low_acc = None
                pending_high_acc = None
                pending_count = 0

                while next_output_t <= now_perf:
                    next_output_t += save_interval_s
            else:
                out_low = low_image
                out_high = high_image

            timestamp = time.time()
            low_image_path = None
            high_image_path = None

            if save and save_path is not None:
                frame_id = int(timestamp * 1000)
                low_image_path = save_path / f"sensor_low_{frame_id}.png"
                high_image_path = save_path / f"sensor_high_{frame_id}.png"
                cv2.imwrite(str(low_image_path), out_low)
                cv2.imwrite(str(high_image_path), out_high)

            if live_view:
                cv2.imshow("Sensor Low Exposure Image", out_low)
                cv2.waitKey(1)

            q.put(
                {
                    "timestamp": timestamp,
                    "sensor_low_path": str(low_image_path) if low_image_path else None,
                    "sensor_high_path": str(high_image_path) if high_image_path else None,
                }
            )
    finally:
        if hasattr(cam, "cam"):
            cam.cam.release()
        if hasattr(cam, "serial_connection"):
            cam.serial_connection.close()
        if live_view:
            cv2.destroyAllWindows()