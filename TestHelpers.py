from Helpers import Cameras
import cv2 as cv
import time

cameras = Cameras.instance()
frames= cameras.get_frames()
print(frames)
cv.imshow("Camera 0", frames[0])
cv.waitKey(0)

# while True:
#     frames= cameras.get_frames()
#     for idx, frame in enumerate(frames):
#         cv.imshow(f"Camera {idx}", frame)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         cv.destroyAllWindows()
#         break