import cv2
import numpy as np

capwidth = 1280
capheight = 720
finalframesize = 512

exposuretime = 50


cam = cv2.VideoCapture(-1)

#cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*['M','J','P','G']))
#cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*['H','2','6','4']))


cam.set(cv2.CAP_PROP_SETTINGS, 0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,capwidth)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,capheight)
cam.set(cv2.CAP_PROP_FPS, 30)
ret_val , cap_for_exposure = cam.read() #Need to call cam.read() first before fiddling with the exposure
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, .75) #For some reason auto exposure off is 0.25 and 0.75 is on
cam.set(cv2.CAP_PROP_GAIN,1)
cam.set(cv2.CAP_PROP_EXPOSURE,100) #exposure is in ms or something for the ubuntu api
#White Balance not working currently
cam.set(cv2.CAP_PROP_AUTO_WB,3) #Off is 1 for some reason, 3 is on
cam.set(cv2.CAP_PROP_AUTO_WB,1) #Off is 1 for some reason, 3 is on
cam.set(cv2.CAP_PROP_WB_TEMPERATURE,2500)
#cam.set(cv2.CAP_EXPOSURE_AUTO_PRIORITY,0)

cframe = np.array(cap_for_exposure.shape)/2
cframe = cframe.astype(int)
imhalf = int(finalframesize/2)
croprange = [cframe[0]-imhalf,cframe[0]+imhalf,cframe[1]-imhalf,cframe[1]+imhalf]

newframe = cap_for_exposure[croprange[0]:croprange[1],croprange[2]:croprange[3]]


def lowhighframecap(lowtime: int, hightime: int) -> np.ndarray:
    cam.set(cv2.CAP_PROP_EXPOSURE,lowtime) #exposure is in ms or something for the ubuntu api
    ret, frame1 = cam.read()
    cam.set(cv2.CAP_PROP_EXPOSURE,hightime) #exposure is in ms or something for the ubuntu api
    ret, frame2 = cam.read()
    return frame1, frame2

i = 0
while True:
    frame1, frame2 = lowhighframecap(exposuretime, exposuretime*5)
    if i%2 == 0:
        cv2.imshow('Camera', frame1)
    else:
        cv2.imshow('Camera', frame2)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break
    i= i+1

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()