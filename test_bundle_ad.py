import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from Helpers import read_camera_params, triangulate_points, calculate_reprojection_errors

def bundle_adjustment(image_points, camera_poses, camera_params):
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
    
    result = least_squares(residual_function, init_params, verbose=2,loss="cauchy", method='trf', ftol=1E-3)
    camera_poses = params_to_camera_poses(result.x)
    
    return camera_poses