import cv2 as cv
import numpy as np
from lib.Helpers import get_extrinsics, triangulate_points, calculate_reprojection_errors, bundle_adjustment, find_point_correspondance_and_object_points
from CapturePoints import get_floor_images, capture_floor_points
import json
from itertools import combinations
import time
from itertools import combinations
import time

image_points = [] # format [[camera1_points], [camera1_points], ...] -> timestamp1 = [timestamp1, timestamp2, ...]
images = []
objs = []
R = None
origin = None
objs = []
R = None
origin = None
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
    '''Read image points from json'''
    global image_points
    global camera_count

    camera_count = len(camera_params)

    with open(points_json) as file:
        image_points = json.load(file)
    image_points = np.array(image_points)
    image_points = np.transpose(image_points, (1, 0, 2))

def calculate_extrinsics():
    '''Calculate extrinsics from image points'''
    global image_points
    global camera_params
    global camera_count
    global global_camera_poses
    Fs = []
    Fs = []

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
        Fs.append(F.tolist())
        Fs.append(F.tolist())

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
    
    with open("./jsons/fundamentals.json", "w") as outfile:
        json.dump(Fs, outfile)
    
    with open("./jsons/fundamentals.json", "w") as outfile:
        json.dump(Fs, outfile)

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
    '''Save camera poses to a json'''
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

def get_extrinsics():
    global global_camera_poses
    global camera_count
    with open("./jsons/after_ba_extrinsics.json") as file:
        global_camera_poses = json.load(file)
        for i in range(0, len(global_camera_poses)):
            global_camera_poses[i]["R"] = np.array(global_camera_poses[i]["R"])
            global_camera_poses[i]["t"] = np.array(global_camera_poses[i]["t"])
    camera_count = len(global_camera_poses)

def save_objects(prefix="",object_points=None):
    '''Save object (which were used to calculate the camera poses) to a json'''
    objects_filename = f"./jsons/{prefix}objects.json"
    with open(objects_filename, "w") as outfile:
        json.dump(object_points.tolist(), outfile)

    print("Object points saved to", objects_filename)

def get_origin(points_3d):
    global global_camera_poses
    global origin
    if len(points_3d) == 4:
        origin = np.mean(points_3d, axis=0)
        print("Origin:", origin)
        for pose in global_camera_poses:
            pose['t'] = pose['t'] - origin
    else:
        print("Invalid number of points to calculate origin")

def get_floor(points_3d):
    global global_camera_poses
    positions = [0,1,2,3]
    coms = list(combinations(positions, 3))
    if len(points_3d) == 4:
        normals = []
        for combination in coms:
            normal = calculate_normal([points_3d[combination[0]], points_3d[combination[1]], points_3d[combination[2]]])
            normals.append(normal)
        normal = np.mean(normals, axis=0)
        global R
        R = rotation_matrix_from_vectors(np.array([0, 0, -1]), normal) # [0,0,-1] can be changed to [0,0,1] if the normal is pointing in the wrong direction
        R = np.pad(R, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        R[3, 3] = 1
        print("poses: ", global_camera_poses[1])
        for i in range(len(global_camera_poses)):
            RT = np.eye(4)
            RT[:3,:3] = global_camera_poses[i]['R']
            RT[:3,3] = global_camera_poses[i]['t'].flatten()
            RT = R.T @ RT # idk why this works but it does should it be RT = RT @ R.T?
            global_camera_poses[i]['R'] = RT[:3,:3]
            global_camera_poses[i]['t'] = RT[:3,3].reshape(3,1)
        print("poses: ", global_camera_poses[1])
    else:
        print("Invalid number of points to calculate floor")

def calculate_normal(points_3d):
    if len(points_3d) == 3:
        normal = np.cross(points_3d[1]-points_3d[0], points_3d[2]-points_3d[0])
        normal = normal / normal[0]
        return normal
    else:
        print("Invalid number of points to calculate normal")
        return np.array([0,0,1])

def rotation_matrix_from_vectors(vec_orig, vec_rot):

    vec_orig = vec_orig / np.linalg.norm(vec_orig)
    vec_rot = vec_rot / np.linalg.norm(vec_rot)

    cross_prod = np.cross(vec_orig, vec_rot)
    cross_norm = np.linalg.norm(cross_prod)
    
    dot_prod = np.dot(vec_orig, vec_rot)
    
    if cross_norm == 0:
        if dot_prod > 0:
            return np.eye(3)
        else:
            axis = np.array([1, 0, 0]) if abs(vec_orig[0]) < 0.99 else np.array([0, 1, 0])
            cross_prod = np.cross(vec_orig, axis)
            cross_prod /= np.linalg.norm(cross_prod)
            cross_norm = 1

    K = np.array([
        [0, -cross_prod[2], cross_prod[1]],
        [cross_prod[2], 0, -cross_prod[0]],
        [-cross_prod[1], cross_prod[0], 0]
    ])

    I = np.eye(3)
    R = I + K + K @ K * ((1 - dot_prod) / (cross_norm ** 2))
    
    return np.array(R)

def correct_objs(): #  just for testing
    global R
    global origin
    with open("./jsons/after_ba_objects.json") as file:
        objs = json.load(file)
        objs = np.array(objs)
    
    for i in range(len(objs)):
        RT = np.eye(4)
        RT[:3,3] = objs[i]-origin
        RT = R.T @ RT
        objs[i] = RT[:3,3]
    save_objects("after_floor_",objs)
def get_origin(points_3d):
    global global_camera_poses
    global origin
    if len(points_3d) == 4:
        origin = np.mean(points_3d, axis=0)
        print("Origin:", origin)
        for pose in global_camera_poses:
            pose['t'] = pose['t'] - origin
    else:
        print("Invalid number of points to calculate origin")

def get_floor(points_3d):
    global global_camera_poses
    positions = [0,1,2,3]
    coms = list(combinations(positions, 3))
    if len(points_3d) == 4:
        normals = []
        for combination in coms:
            normal = calculate_normal([points_3d[combination[0]], points_3d[combination[1]], points_3d[combination[2]]])
            normals.append(normal)
        normal = np.mean(normals, axis=0)
        global R
        R = rotation_matrix_from_vectors(np.array([0, 0, -1]), normal) # [0,0,-1] can be changed to [0,0,1] if the normal is pointing in the wrong direction
        R = np.pad(R, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        R[3, 3] = 1
        print("poses: ", global_camera_poses[1])
        for i in range(len(global_camera_poses)):
            RT = np.eye(4)
            RT[:3,:3] = global_camera_poses[i]['R']
            RT[:3,3] = global_camera_poses[i]['t'].flatten()
            RT = R.T @ RT # idk why this works but it does should it be RT = RT @ R.T?
            global_camera_poses[i]['R'] = RT[:3,:3]
            global_camera_poses[i]['t'] = RT[:3,3].reshape(3,1)
        print("poses: ", global_camera_poses[1])
    else:
        print("Invalid number of points to calculate floor")

def calculate_normal(points_3d):
    if len(points_3d) == 3:
        normal = np.cross(points_3d[1]-points_3d[0], points_3d[2]-points_3d[0])
        normal = normal / normal[0]
        return normal
    else:
        print("Invalid number of points to calculate normal")
        return np.array([0,0,1])

def rotation_matrix_from_vectors(vec_orig, vec_rot):

    vec_orig = vec_orig / np.linalg.norm(vec_orig)
    vec_rot = vec_rot / np.linalg.norm(vec_rot)

    cross_prod = np.cross(vec_orig, vec_rot)
    cross_norm = np.linalg.norm(cross_prod)
    
    dot_prod = np.dot(vec_orig, vec_rot)
    
    if cross_norm == 0:
        if dot_prod > 0:
            return np.eye(3)
        else:
            axis = np.array([1, 0, 0]) if abs(vec_orig[0]) < 0.99 else np.array([0, 1, 0])
            cross_prod = np.cross(vec_orig, axis)
            cross_prod /= np.linalg.norm(cross_prod)
            cross_norm = 1

    K = np.array([
        [0, -cross_prod[2], cross_prod[1]],
        [cross_prod[2], 0, -cross_prod[0]],
        [-cross_prod[1], cross_prod[0], 0]
    ])

    I = np.eye(3)
    R = I + K + K @ K * ((1 - dot_prod) / (cross_norm ** 2))
    
    return np.array(R)

def correct_objs(): #  just for testing
    global R
    global origin
    with open("./jsons/after_ba_objects.json") as file:
        objs = json.load(file)
        objs = np.array(objs)
    
    for i in range(len(objs)):
        RT = np.eye(4)
        RT[:3,3] = objs[i]-origin
        RT = R.T @ RT
        objs[i] = RT[:3,3]
    save_objects("after_floor_",objs)

if __name__ == "__main__":
    get_points()
    calculate_extrinsics()
    # global_camera_poses, camera_count = get_extrinsics()
    # get_floor_images()
    # points = capture_floor_points()
    # # points = points.transpose(1,0,2)
    # start_time = time.time()
    # print(points.shape)
    # objs = find_point_correspondance_and_object_points(points, global_camera_poses)

    # end_time = time.time()
    # print(f"FPS for find_point_correspondance_and_object_points: {1/(end_time - start_time)} FPS")
    # save_objects("after_origin_",objs)
    # get_origin(objs)
    # get_floor(objs)
