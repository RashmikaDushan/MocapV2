import cv2 as cv
import numpy as np
from Helpers import Cameras

cameras = Cameras.instance()

camera1_image_points = np.array([[183, 246], [115, 225], [130, 205], [80, 194], [102, 175], [163, 182], [135, 176], [201, 165], [96, 151], [142, 137], [121, 124], [178, 104], [140, 90], [99, 76], [125, 42]]) 
camera2_image_points = np.array([[189, 237], [121, 228], [134, 205], [84, 203], [105, 179], [165, 176], [138, 174], [201, 153], [97, 156], [142, 135], [121, 126], [176, 97], [138, 90], [97, 83], [121, 46]])

not_none_indicies = np.where(np.all(camera1_image_points != None, axis=1) & np.all(camera2_image_points != None, axis=1))[0]
camera1_image_points = np.take(camera1_image_points, not_none_indicies, axis=0).astype(np.float32)
camera2_image_points = np.take(camera2_image_points, not_none_indicies, axis=0).astype(np.float32)

F, _ = cv.findFundamentalMat(camera1_image_points, camera2_image_points, cv.FM_RANSAC, 1, 0.99999)

print(F)

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

camera_poses = bundle_adjustment(image_points, camera_poses, socketio)

object_points = triangulate_points(image_points, camera_poses)
error = np.mean(calculate_reprojection_errors(image_points, object_points, camera_poses))

socketio.emit("camera-pose", {"camera_poses": camera_pose_to_serializable(camera_poses)})