import cv2
import numpy as np
import serial
from serial.tools import list_ports
import time

class SensorCamera:
    DEVICE_CODE = "RGBBL-001"

    def __init__(self,serial_port="/dev/ttyUSB0", baud_rate=115200,capwidth=1280,capheight=720,exposuretime=100):
        '''
        Initializes the camera and the serial connection to the microcontroller controlling the LED backlight.
         The camera is configured with the specified resolution and exposure time (check camera docs for allowed values)
        '''
        self.cam = cv2.VideoCapture(-1)
        self.cam.set(cv2.CAP_PROP_SETTINGS, 0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,capwidth)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,capheight)
        self.cam.set(cv2.CAP_PROP_FPS, 30)
        ret_val , cap_for_exposure = self.cam.read() #Need to call cam.read() first before fiddling with the exposure
        self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, .75) #For some reason auto exposure off is 0.25 and 0.75 is on/off
        self.cam.set(cv2.CAP_PROP_GAIN,1)
        self.cam.set(cv2.CAP_PROP_EXPOSURE,exposuretime) #exposure is in ms or something for the ubuntu api
        #White Balance not working currently
        self.cam.set(cv2.CAP_PROP_AUTO_WB,3) #Off is 1 for some reason, 3 is on
        self.cam.set(cv2.CAP_PROP_AUTO_WB,1) #Off is 1 for some reason, 3 is on
        self.cam.set(cv2.CAP_PROP_WB_TEMPERATURE,2500)
        self.serial_connection = self._connect_rgb_backlight(serial_port, baud_rate)


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
        ret, frame1 = self.cam.read()

        self.serial_connection.write(c1)  # Turn LEDs on for the high-exposure frame
        self.cam.set(cv2.CAP_PROP_EXPOSURE,hightime) #exposure is in ms or something for the ubuntu api
        self.cam.set(cv2.CAP_PROP_GAIN,highgain)
        ret, frame2 = self.cam.read()
        return frame1, frame2

    def normalframecap(self) -> np.ndarray:
        '''
        Captures a single frame from the camera
        '''
        ret, frame = self.cam.read()
        return frame

def sensorgrab(stop_event, cap_params, q, save=False, save_path=None, live_view=False):
    # Create camera inside the worker process (required for Windows multiprocessing spawn).
    cam = SensorCamera()
    try:
        if isinstance(cap_params, dict):
            cap_args = (
                cap_params.get("lowtime", 100),
                cap_params.get("hightime", 200),
                cap_params.get("lowgain", 1.0),
                cap_params.get("highgain", 1.0),
                cap_params.get("light_type", "white"),
            )
        else:
            cap_args = tuple(cap_params)

        while not stop_event.is_set():
            low_image, high_image = cam.lowhighframecap(*cap_args)
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