import cv2 as cv
import threading
import queue
from Helpers import get_extrinsics

camera_poses, camera_count = get_extrinsics()

frame_queues = {i: queue.Queue(maxsize=10) for i in range(camera_count)}  # One queue per camera
vid_paths = ["./videos/cam1.mp4", "./videos/cam2.mp4","./videos/cam3.mp4", "./videos/cam4.mp4", "./videos/cam5.mp4","./videos/cam6.mp4"]  # Video paths

lock = threading.Lock()

def capture_frames(camera_id):
    cap = cv.VideoCapture(vid_paths[0])
    while True:
        ret, frame = cap.read()
        if ret:
            cv.imshow(f"Camera {camera_id}", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_queues[camera_id].full():
                frame_queues[camera_id].get()  # Discard the oldest frame if the queue is full
            frame_queues[camera_id].put(frame)

    cap.release()


def process_frames():
    while True:
        if not frame_queues[0].empty():
            frame = frame_queues[0].get()  # Get the frame from the queue
            #image filter
            #get points 
            # calculate 3D points
            #send 3d points

capture_threads = [] # threads for each camera
for camera_id in range(camera_count):
    thread = threading.Thread(target=capture_frames, args=(camera_id,lock,))
    thread.start()
    capture_threads.append(thread)

# Start the processing thread
process_thread = threading.Thread(target=process_frames, args=(lock,))
process_thread.start()

# Wait for all threads to finish (optional)
for thread in capture_threads:
    thread.join()
process_thread.join()
