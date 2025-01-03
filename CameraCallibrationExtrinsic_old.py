from Helpers import Cameras
import cv2 as cv
import time
import numpy as np
from Helpers import triangulate_points, calculate_reprojection_errors, bundle_adjustment

image_points = [] # format [[timestamp1], [timestamp2], ...] -> timestamp1 = [camera1_points, camera2_points, ...]

# Initialize cameras
cameras = Cameras.instance()  # The camera array
cameras.over_websockets = False  # Set to True to stream the cameras over websockets

image_count = 13

def capture_points():
    global image_points
    global cameras
    global image_count

    # Target FPS
    fps_limit = 60
    frame_duration = 1 / fps_limit  # Duration of each frame in seconds

    print("Press 'space' to take a picture and 'q' to quit.")

    captured_frames = 0

    while True:
        print("Captured frames:", captured_frames)
        if captured_frames >= image_count:
            break
        start_time = time.time()  # Record the start time of the loop

        # Retrieve frames from all cameras
        cameras.start_capturing_points()
        frames = cameras.get_frames()

        # Display each frame in its corresponding window
        cv.imshow("cameras", frames)

        # Wait for a key press and limit the frame rate
        key = cv.waitKey(0) & 0xFF

        # If space is pressed, handle image capture
        if key == ord(' '):  # Spacebar key
            print("Taking picture...")
            image_points.append(cameras.image_points)
            captured_frames += 1
            # print(image_points)


        # If 'q' is pressed, exit the program
        elif key == ord('q'):  # 'q' key
            print("Exiting...")
            break

        # Sleep to maintain the target frame rate
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_duration:
            time.sleep(frame_duration - elapsed_time)

    image_points = np.array(image_points)
    image_points = np.transpose(image_points, axes=[2, 1, 0, 3])
    image_points = image_points[0]
    print("Image points shape:", image_points.shape)
    print("Image points:", image_points)
    # Release resources and close all OpenCV windows
    cv.destroyAllWindows()


def calculate_extrinsics():
    global image_points
    global cameras
    camera_poses = [{
        "R": np.eye(3),
        "t": np.array([[0],[0],[0]], dtype=np.float32)
    }]
    for camera_i in range(0, cameras.num_cameras-1): # for each camera pair
        camera1_image_points = np.array(image_points[camera_i])
        camera2_image_points = np.array(image_points[camera_i+1])
        print("Camera 1 image points:", camera1_image_points)
        print("Camera 2 image points:", camera2_image_points)

        F, _ = cv.findFundamentalMat(points1=camera1_image_points, points2=camera2_image_points, method=cv.FM_RANSAC, ransacReprojThreshold=1,confidence= 0.99999)
        # E = cv.sfm.essentialFromFundamental(F, cameras.get_camera_params(0)["intrinsic_matrix"], cameras.get_camera_params(1)["intrinsic_matrix"])
        # possible_Rs, possible_ts = cv.sfm.motionFromEssential(E)

        K1 = cameras.get_camera_params(0)["intrinsic_matrix"]  # Intrinsic matrix for camera 1
        K2 = cameras.get_camera_params(1)["intrinsic_matrix"]  # Intrinsic matrix for camera 2

        print("K1 shape", np.shape(K1))
        print("K2 shape", np.shape(K2))
        print("F shape", np.shape(F))

        E = np.transpose(K2) @ F @ K1

        # Decompose the Essential Matrix into Rotation and Translation
        R1, R2, t = cv.decomposeEssentialMat(E)

        # Output possible rotation matrices and translation vector
        possible_Rs = [R1, R1, R2, R2]  # 4 possible rotation matrices
        possible_ts = [t, -t, t, -t]   # 4 possible translation directions

        # Example of using R and t for further computations
        print("Possible Rotations:", possible_Rs)
        print("Possible Translations:", possible_ts)

        R = None
        t = None
        max_points_infront_of_camera = 0
        for i in range(0, 4):
            object_points = triangulate_points(np.hstack([np.expand_dims(camera1_image_points, axis=1), np.expand_dims(camera2_image_points, axis=1)]), np.concatenate([[camera_poses[-1]], [{"R": possible_Rs[i], "t": possible_ts[i]}]]))
            object_points_camera_coordinate_frame = np.array([possible_Rs[i].T @ object_point for object_point in object_points])

            points_infront_of_camera = np.sum(object_points[:,2] > 0) + np.sum(object_points_camera_coordinate_frame[:,2] > 0)

            if points_infront_of_camera > max_points_infront_of_camera:
                max_points_infront_of_camera = points_infront_of_camera
                R = possible_Rs[i]
                t = possible_ts[i]

        R = R @ camera_poses[-1]["R"]
        t = camera_poses[-1]["t"] + (camera_poses[-1]["R"] @ t)

        camera_poses.append({
            "R": R,
            "t": t
        })

    camera_poses = bundle_adjustment(image_points, camera_poses)

    object_points = triangulate_points(image_points, camera_poses)
    error = np.mean(calculate_reprojection_errors(image_points, object_points, camera_poses))
    print("Reprojection error:", error)

if __name__ == "__main__":
    capture_points()
    calculate_extrinsics()
