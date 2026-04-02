import cv2
import numpy as np
import serial
import time
capwidth = 1280
capheight = 720
finalframesize = 512

exposuretime = 50

class SensorCamera:
    def __init__(self,serial_port="/dev/ttyUSB0", baud_rate=115200):
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
        self.serial_connection = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)  # Wait for the serial connection to initialize


    def lowhighframecap(self, lowtime: int, hightime: int, lowgain: float = 1.0, highgain: float = 1.0) -> (np.ndarray, np.ndarray):
        '''
        Captures two frames from the camera, one with low exposure and one with high exposure.
          The microcontroller is used to switch the whether the LED backlight is on or off.
        '''
        self.serial_connection.write(b'1')  # Send a byte to trigger the microcontroller to set the camera to low exposure
        self.cam.set(cv2.CAP_PROP_EXPOSURE,lowtime) #exposure is in ms or something for the ubuntu api
        self.cam.set(cv2.CAP_PROP_GAIN,lowgain)
        ret, frame1 = self.cam.read()

        self.serial_connection.write(b'2')  # Send a byte to trigger the microcontroller to set the camera to high exposure
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