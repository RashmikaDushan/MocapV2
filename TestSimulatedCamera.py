from CameraSimulation import Camera
import cv2 as cv
import time

gain = 1.5
exposure = 0

def update_gain(value):
    global gain
    gain = value / 10
    cameras.set_gain(gain)

def update_exposure(value):
    global exposure
    exposure = value - 50
    cameras.set_exposure(exposure)

cameras = Camera(fps=90, resolution=Camera.RES_LARGE, gain=10, exposure=100)
num_cameras = len(cameras.exposure)
print(f"Number of cameras: {num_cameras}")

cv.namedWindow('Filtered Image')
cv.createTrackbar('Gain', 'Filtered Image', 10, 20, update_gain)
cv.createTrackbar('Exposure', 'Filtered Image', 50, 100, update_exposure)
time.sleep(1)

while True:
    frames, _ = cameras.read()
    for idx, frame in enumerate(frames):
        cv.imshow(f"Camera {idx}", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break