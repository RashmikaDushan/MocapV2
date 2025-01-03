from Helpers import Cameras
import cv2 as cv
import time
import numpy as np
from Helpers import triangulate_points, calculate_reprojection_errors, bundle_adjustment
import os
import glob
import json
from dotenv import load_dotenv

image_points = [] # format [[timestamp1], [timestamp2], ...] -> timestamp1 = [camera1_points, camera2_points, ...]
images = []
image_count = 0
camera_count = 0

load_dotenv()
filename = os.getenv("CAMERA_PARAMS_IN")
cam_params = open(filename)
camera_params = json.load(cam_params)

def get_images(preview=False):
    global images
    global image_count
    global camera_count
    images = []

    img_folder_paths = sorted([f"extrinsics/{i}" for i in os.listdir("extrinsics") if i.startswith('cam')])
    camera_count = len(img_folder_paths)
    print(img_folder_paths)

    for img_folder_path in img_folder_paths:
        image_names = sorted(glob.glob(f'./{img_folder_path}/*.jpg'))
        image_count = len(image_names)
        print(image_names)
        imgs = []
        for fname in image_names:
            img = cv.imread(fname)
            img = image_filter(img)
            if preview:
                cv.imshow(f'{fname}',img)
                key = cv.waitKey(0) & 0xFF
                if key == ord('q'):
                    print("Exiting...")
                    cv.destroyAllWindows()
                    quit()
            imgs.append(img)
        images.append(imgs)

    images = np.array(images)
    print("image_count:", image_count)
    print("camera_count:", camera_count)
    print(images.shape)

def image_filter(image):
    global camera_params
    image = np.rot90(image, k=camera_params[0]["rotation"])
    # image = make_square(image) # make the image square with padding
    image = cv.undistort(image, np.array(camera_params[0]["intrinsic_matrix"]), np.array(camera_params[0]["distortion_coef"])) ## 0 because all the cameras have the same intrinsic matrix
    image = cv.medianBlur(image, 5)
    # image = cv.GaussianBlur(image,(9,9),0)
    # kernel = np.array([[-2,-1,-1,-1,-2],
    #                     [-1,1,3,1,-1],
    #                     [-1,3,4,3,-1],
    #                     [-1,1,3,1,-1],
    #                     [-2,-1,-1,-1,-2]])
    # image = cv.filter2D(image, -1, kernel)
    # image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image



def _find_dot(img):
    """Finds the points in a frame and prints on the image then return both image and the points"""
    grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    grey = cv.threshold(grey, 255*0.2, 255, cv.THRESH_BINARY)[1]
    contours,_ = cv.findContours(grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img = cv.drawContours(img, contours, -1, (0,255,0), 1)

    image_points = []
    for contour in contours:
        moments = cv.moments(contour)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"]) #finding the center of a point
            cv.putText(img, f'({center_x}, {center_y})', (center_x,center_y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.3, (100,255,100), 1)
            cv.circle(img, (center_x,center_y), 1, (100,255,100), -1)
            image_points.append([center_x, center_y])

    if len(image_points) == 0:
        image_points = [[None, None]]

    return img, image_points


def capture_points(preview=False):
    global image_points
    global image_count
    global images

    if preview:
        print("Press 'space' to take a picture and 'q' to quit.")

    for i in range(0, camera_count):
        image_points_single_camera = []
        for j in range(0, image_count):
            image, image_points_single_frame = _find_dot(images[i][j])
            if len(image_points_single_frame) != 1:
                print("Found more than one point in the image. Please make sure there is only one point in the image.")
                quit()
            else:
                image_points_single_frame = image_points_single_frame[0]
            if preview:
                cv.imshow("Image", image)
                key = cv.waitKey(0) & 0xFF
                # # If space is pressed, handle image capture
                if key == ord(' '):  # Spacebar key
                    print("Saving points...")
                    image_points_single_camera.append(image_points_single_frame)

                # If 'q' is pressed, exit the program
                if key == ord('q'):  # 'q' key
                    print("Exiting...")
                    cv.destroyAllWindows()
                    quit()
            else:
                image_points_single_camera.append(image_points_single_frame)
        print("Image points for camera", i, ":", image_points_single_camera)
        image_points.append(image_points_single_camera)

    image_points = np.array(image_points)
    print("Image points shape:", image_points.shape)
    print("Image points:", image_points)
    # Release resources and close all OpenCV windows
    cv.destroyAllWindows()


def calculate_extrinsics():
    global image_points
    global camera_params
    global camera_count

    camera_poses = [{
        "R": np.eye(3),
        "t": np.array([[0],[0],[0]], dtype=np.float32)
    }]
    for camera_i in range(0, camera_count-1): # for each camera pair
        camera1_image_points = image_points[camera_i]
        camera2_image_points = image_points[camera_i+1]
        camera1_image_points = np.array(camera1_image_points, dtype=np.float32)
        camera2_image_points = np.array(camera2_image_points, dtype=np.float32)
        print("Camera 1 image points:", camera1_image_points)
        print("Camera 2 image points:", camera2_image_points)

        F, _ = cv.findFundamentalMat(camera1_image_points, camera2_image_points, cv.FM_RANSAC, 1, 0.99999)
        # E = cv.sfm.essentialFromFundamental(F, cameras.get_camera_params(0)["intrinsic_matrix"], cameras.get_camera_params(1)["intrinsic_matrix"])
        # possible_Rs, possible_ts = cv.sfm.motionFromEssential(E)

        K1 = camera_params[0]["intrinsic_matrix"]  # Intrinsic matrix for camera 1
        K2 = camera_params[1]["intrinsic_matrix"]  # Intrinsic matrix for camera 2

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
    get_images()
    capture_points()
    calculate_extrinsics()