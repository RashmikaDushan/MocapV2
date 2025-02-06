import numpy as np
from scipy import linalg, optimize
import json
import cv2 as cv
from scipy.spatial.transform import Rotation
import copy

camera_params = None
camera_params_path = "./jsons/camera-params-in.json"
Fs = []

def get_extrinsics():
    with open("./jsons/after_ba_extrinsics.json") as file:
        camera_poses = json.load(file)
        for i in range(0, len(camera_poses)):
            camera_poses[i]["R"] = np.array(camera_poses[i]["R"])
            camera_poses[i]["t"] = np.array(camera_poses[i]["t"])
    camera_count = len(camera_poses)
    return camera_poses, camera_count

def read_fundamental_matrix():
    global Fs

    if len(Fs) == 0:
        with open("./jsons/fundamentals.json") as file:
            Fs = json.load(file)
            print("Fundamental matrix loaded")

def read_camera_params():
    global camera_params
    global camera_params_path

    if camera_params is None:
        with open(camera_params_path, "r") as file:
            camera_params = json.load(file)
            camera_params = np.array(camera_params)
            print("Camera params loaded")


def triangulate_point(image_points, camera_poses):
    """image_points shape = [camera_count,2]"""

    global camera_params

    read_camera_params()

    image_points = np.array(image_points)
    none_indicies = np.where(np.all(image_points == None, axis=1))[0]
    image_points = np.delete(image_points, none_indicies, axis=0)
    camera_poses = np.delete(camera_poses, none_indicies, axis=0)

    if len(image_points) <= 1:
        return [None, None, None]

    Ps = [] # projection matricies
    for i, camera_pose in enumerate(camera_poses):
        RT = np.c_[camera_pose["R"], camera_pose["t"]]
        P = camera_params[i]["intrinsic_matrix"] @ RT
        Ps.append(P)

    # https://temugeb.github.io/computer_vision/2021/02/06/direct-linear-transorms.html
    def DLT(Ps, image_points):

        """image_points: [[x_cam1, y_cam1], [x_cam2, y_cam2], ... , [x_cam6, y_cam6]]"""

        A = []

        for P, image_point in zip(Ps, image_points):
            A.append(image_point[1]*P[2,:] - P[1,:])
            A.append(P[0,:] - image_point[0]*P[2,:])
            
        A = np.array(A).reshape((len(Ps)*2,4))
        B = A.transpose() @ A
        U, s, Vh = linalg.svd(B, full_matrices = False)
        object_point = Vh[3,0:3]/Vh[3,3]

        return object_point

    object_point = DLT(Ps, image_points)

    return object_point


def triangulate_points(image_points, camera_poses):
    '''image_points shape = [obj points,camera_count,2]'''
    object_points = []
    for image_points_i in image_points:
        object_point = triangulate_point(image_points_i, camera_poses)
        object_points.append(object_point)
    
    return np.array(object_points)


def calculate_reprojection_errors(image_points, object_points, camera_poses):
    errors = np.array([])
    for image_points_i, object_point in zip(image_points, object_points):
        error = calculate_reprojection_error(image_points_i, object_point, camera_poses)
        if error is None:
            continue
        errors = np.concatenate([errors, [error]])

    return errors


def calculate_reprojection_error(image_points, object_point, camera_poses):
    '''image points shape (cam_count,2)
    object_point shape (3)
    '''
    global camera_params

    read_camera_params()

    image_points = np.array(image_points)
    none_indicies = np.where(np.all(image_points == None, axis=1))[0]
    image_points = np.delete(image_points, none_indicies, axis=0)
    camera_poses = np.delete(camera_poses, none_indicies, axis=0)

    if len(image_points) <= 1:
        return None

    errors = np.array([])
    for i, camera_pose in enumerate(camera_poses):
        if np.all(image_points[i] == None, axis=0):
            continue
        projected_img_point, _ = cv.projectPoints(
            np.expand_dims(object_point, axis=0).astype(np.float32), 
            np.array(camera_pose["R"], dtype=np.float64), 
            np.array(camera_pose["t"], dtype=np.float64), 
            np.array(camera_params[i]["intrinsic_matrix"]), 
            np.array(camera_params[i]["distortion_coef"])
        )
        projected_img_point = projected_img_point[0][0]
        errors = np.concatenate([errors, (np.array(image_points[i])-np.array(projected_img_point)).flatten() ** 2])
    
    return errors.mean()


def bundle_adjustment(image_points, camera_poses):
    num_cameras = len(camera_poses)

    def params_to_camera_poses(params):
        camera_poses = [{
            "R": np.eye(3),
            "t": np.array([0,0,0], dtype=np.float32)
        }]
        for i in range(0, num_cameras-1):
            camera_poses.append({
                "R": Rotation.as_matrix(Rotation.from_rotvec(params[i*6 : i*6 + 3])),
                "t": params[i*6 + 3 : i*6 + 6]
            })

        return camera_poses

    def residual_function(params):
        camera_poses = params_to_camera_poses(params)
        object_points = triangulate_points(image_points, camera_poses)
        errors = calculate_reprojection_errors(image_points, object_points, camera_poses)
        errors = errors.astype(np.float32)
        
        return errors
    
    init_params = np.array([])
    for i, camera_pose in enumerate(camera_poses[1:]):
        init_params = np.concatenate([init_params, Rotation.from_matrix(camera_pose["R"]).as_rotvec(), camera_pose["t"].flatten()])
    
    result = optimize.least_squares(residual_function, init_params, verbose=2,loss="linear", method='trf', ftol=1E-5, xtol=1E-15)
    camera_poses = params_to_camera_poses(result.x)
    
    return camera_poses

def find_point_correspondance_and_object_points(image_points, camera_poses):
    '''image_points shape = [camera_count, obj points, 2]'''
    global camera_params
    obj_count = len(image_points[0])
    read_camera_params()

    for image_points_i in image_points:
        try:
            image_points_i.remove([None, None])
        except:
            pass

    # [object_points, possible image_point groups, image_point from camera]
    correspondances = [[[i]] for i in image_points[0]]

    Ps = [] # projection matricies
    for i, camera_pose in enumerate(camera_poses):
        RT = np.c_[camera_pose["R"], camera_pose["t"]]
        P = camera_params[i]["intrinsic_matrix"] @ RT
        Ps.append(P)

    root_image_points = [{"camera": 0, "point": point} for point in image_points[0]]

    read_fundamental_matrix()

    for i in range(1, len(camera_poses)):
        epipolar_lines = []
        for root_image_point in root_image_points:
            F = np.array(Fs[i-1])
            line = cv.computeCorrespondEpilines(np.array([root_image_point["point"]], dtype=np.float32), 1, F)
            epipolar_lines.append(line[0,0].tolist())
            # frames[i] = drawlines(frames[i], line[0])

        not_closest_match_image_points = np.array(image_points[i])
        points = np.array(image_points[i])

        for j, [a, b, c] in enumerate(epipolar_lines):
            distances_to_line = np.array([])
            if len(points) != 0:
                distances_to_line = np.abs(a*points[:,0] + b*points[:,1] + c) / np.sqrt(a**2 + b**2)

            possible_matches = points[distances_to_line < 5].copy()

            # Commenting out this code produces more points, but more garbage points too
            # delete closest match from future consideration
            # if len(points) != 0:
            #     points = np.delete(points, np.argmin(distances_to_line), axis=0)

            # sort possible matches from smallest to largest
            distances_to_line = distances_to_line[distances_to_line < 5]
            possible_matches_sorter = distances_to_line.argsort()
            possible_matches = possible_matches[possible_matches_sorter]
    
            if len(possible_matches) == 0:
                for possible_group in correspondances[j]:
                    possible_group.append([None, None])
            else:
                not_closest_match_image_points = [row for row in not_closest_match_image_points.tolist() if row != possible_matches.tolist()[0]]
                not_closest_match_image_points = np.array(not_closest_match_image_points)
                
                new_correspondances_j = []
                for possible_match in possible_matches:
                    temp = copy.deepcopy(correspondances[j])
                    for possible_group in temp:
                        possible_group.append(possible_match.tolist())
                    new_correspondances_j += temp
                correspondances[j] = new_correspondances_j

        for not_closest_match_image_point in not_closest_match_image_points:
            root_image_points.append({"camera": i, "point": not_closest_match_image_point})
            temp = [[[None, None]] * i]
            temp[0].append(not_closest_match_image_point.tolist())
            correspondances.append(temp)

    object_points = []
    errors = []
    for image_points in correspondances:
        object_points_i = triangulate_points(image_points, camera_poses)

        if np.all(object_points_i == None):
            continue

        errors_i = calculate_reprojection_errors(image_points, object_points_i, camera_poses)

        object_points.append(object_points_i[np.argmin(errors_i)])
        errors.append(np.min(errors_i))
    sorted_indices = np.argsort(errors)
    sorted_errors = sorted_indices[:obj_count]
    selected_object_points = [object_points[i] for i in sorted_errors]
    return np.array(object_points)