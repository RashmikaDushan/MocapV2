import cv2 as cv
import numpy as np
from Helpers import triangulate_points, calculate_reprojection_errors, bundle_adjustment
import json

image_points = [] # format [[camera1_points], [camera1_points], ...] -> timestamp1 = [timestamp1, timestamp2, ...]
images = []
camera_count = 0
global_camera_poses = [{
        "R": np.eye(3),
        "t": np.array([[0],[0],[0]], dtype=np.float32)
    }]

intrinsics_json = "./jsons/camera-params-in.json"
points_json = "./jsons/image_points.json"

camera_params_file = open(intrinsics_json)
camera_params = json.load(camera_params_file)

def get_points():
    global image_points
    global camera_count

    camera_count = len(camera_params)

    with open(points_json) as file:
        image_points = json.load(file)
    image_points = np.array(image_points)
    image_points = np.transpose(image_points, (1, 0, 2))



def calculate_extrinsics():
    global image_points
    global camera_params
    global camera_count
    global global_camera_poses

    camera_poses = [{
        "R": np.eye(3),
        "t": np.array([[0],[0],[0]], dtype=np.float32)
    }]
    for camera_i in range(0, camera_count-1):
        camera1_image_points = image_points[camera_i]
        camera2_image_points = image_points[camera_i+1]
        camera1_image_points = np.array(camera1_image_points, dtype=np.float32)
        camera2_image_points = np.array(camera2_image_points, dtype=np.float32)

        F, _ = cv.findFundamentalMat(camera1_image_points, camera2_image_points, cv.FM_RANSAC, 10, 0.99999)

        K1 = camera_params[0]["intrinsic_matrix"]
        K2 = camera_params[1]["intrinsic_matrix"]

        E = np.transpose(K2) @ F @ K1

        R1, R2, t = cv.decomposeEssentialMat(E)

        possible_Rs = [R1, R1, R2, R2]
        possible_ts = [t, -t, t, -t]

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
        print("R:", R)
        print("t:", t)
        camera_poses.append({
            "R": R,
            "t": t
        })

    global_camera_poses = camera_poses
    save_extrinsics("before_ba_")
    image_points = np.transpose(image_points, (1, 0, 2))
    camera_poses = bundle_adjustment(image_points, camera_poses)
    object_points = triangulate_points(image_points, camera_poses)
    save_objects("after_ba_",object_points)
    error = np.mean(calculate_reprojection_errors(image_points, object_points, camera_poses))
    global_camera_poses = camera_poses
    print("Reprojection error:", error)
    save_extrinsics("after_ba_")
    save_objects(prefix="after_ba_",object_points=object_points)

def save_extrinsics(prefix=""):
    global global_camera_poses
    global camera_count

    extrinsics = []
    for i in range(0, camera_count):
        extrinsics.append({
            "R": global_camera_poses[i]["R"].tolist(),
            "t": global_camera_poses[i]["t"].flatten().tolist()
        })

    extrinsics_filename = f"./jsons/{prefix}extrinsics.json"
    with open(extrinsics_filename, "w") as outfile:
        json.dump(extrinsics, outfile)

    print("Extrinsics saved to", extrinsics_filename)

def save_objects(prefix="",object_points=None):
    objects_filename = f"./jsons/{prefix}objects.json"
    with open(objects_filename, "w") as outfile:
        json.dump(object_points.tolist(), outfile)

    print("Object points saved to", objects_filename)


if __name__ == "__main__":  
    get_points()
    calculate_extrinsics()
