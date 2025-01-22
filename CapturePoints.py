import cv2 as cv
import numpy as np
from Helpers import triangulate_points, calculate_reprojection_errors, bundle_adjustment
import os
import glob
import json
from dotenv import load_dotenv

image_points = [] # format [[camera1_points], [camera1_points], ...] -> timestamp1 = [timestamp1, timestamp2, ...]
images = []
image_count = 0
camera_count = 0

intrinsics_json = "./jsons/camera-params-in.json"
points_json = "./jsons/image_points.json"

camera_params_file = open(intrinsics_json)
camera_params = json.load(camera_params_file)

def get_pose_images(preview=False,debug=False):
    '''output images shape: (camera_count, image_count, height, width, channels)
    no prerequisites needed'''
    global images
    global image_count
    global camera_count
    images = []

    img_folder_paths = sorted([f"captured_images/{i}" for i in os.listdir("captured_images") if i.startswith('cam')])
    camera_count = len(img_folder_paths)
    if debug:
        print(img_folder_paths)

    for img_folder_path in img_folder_paths:
        image_names = sorted(glob.glob(f'./{img_folder_path}/*.jpg'))
        image_count = len(image_names)
        if debug:
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
    print("Point count:", image_count)
    print("Camera count:", camera_count)
    print("Images shape",images.shape)

def get_floor_images(preview=False,debug=False):
    '''output images shape: (camera_count, 1, height, width, channels)
    no prerequisites needed'''
    global images
    global image_count
    global camera_count
    images = []

    image_names = sorted(glob.glob(f'./get_floor_images/*.png'))
    image_count = 1
    camera_count = len(image_names)
    if debug:
        print(image_names)
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
        images.append([img]) 

    images = np.array(images)
    print("Point count:", image_count)
    print("Camera count:", camera_count)
    print("Images shape",images.shape)

def image_filter(image,camera_number=0):
    '''output: filtered image
    prerequisites: camera_params distortion'''
    global camera_params
    if camera_params is None:
        print("Camera parameters not found.")
        quit()
    image = cv.undistort(image, np.array(camera_params[camera_number]["intrinsic_matrix"]), np.array(camera_params[camera_number]["distortion_coef"]))
    image = cv.medianBlur(image, 5)
    return image

def _find_dot(img):
    '''output: image with dot and dot coordinates
    prerequisites needed: image'''
    grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    grey = cv.threshold(grey, 255*0.2, 255, cv.THRESH_BINARY)[1]
    contours,_ = cv.findContours(grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img = cv.drawContours(img, contours, -1, (0,255,0), 1)

    image_points = []
    for contour in contours:
        moments = cv.moments(contour)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            cv.putText(img, f'({center_x}, {center_y})', (center_x,center_y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.3, (100,255,100), 1)
            cv.circle(img, (center_x,center_y), 1, (100,255,100), -1)
            image_points.append([center_x, center_y])

    if len(image_points) == 0:
        image_points = [[None, None]]

    return img, image_points

def capture_pose_points(preview=False,debug=False):
    '''output: saves points to points_json
    prerequisites needed: images shape: (camera_count, image_count, height, width, channels)'''
    global image_points
    global image_count
    global images

    if preview:
        print("Press 'space' to take a picture and 'q' to quit.")

    image_points = []

    for j in range(0, image_count):
        proccessed_images = []
        calculated_points = np.zeros((camera_count, 2))
        for i in range(0, camera_count):
            image, image_point = _find_dot(images[i][j])
            calculated_points[i] = np.array(image_point).flatten()
            proccessed_images.append(image)
            if len(image_point) != 1:
                print("Found more than one point or no point in the image . Please make sure there is only one point in the image.")
                quit()
            else:
                image_point = image_point[0]
        if preview:
            top = np.hstack([proccessed_images[0], np.hstack([proccessed_images[1], proccessed_images[2]])])
            bottom = np.hstack([proccessed_images[3], np.hstack([proccessed_images[4], proccessed_images[5]])])
            image = np.vstack([top, bottom])
            cv.imshow("Image", image)
            key = cv.waitKey(0) & 0xFF
            if key == ord(' '):
                print("Saving points...")
                image_points.append(calculated_points.tolist())

            if key == ord('q'):
                print("Exiting...")
                cv.destroyAllWindows()
                quit()
        else:
            image_points.append(calculated_points.tolist())
        
        if debug:
            print("Image points for camera", i, ":", image_points)

    with open(points_json, "w") as file:
        json.dump(image_points, file)

    cv.destroyAllWindows()

def capture_floor_points(preview=False,debug=False):
    '''output: calculated points
    prerequisites needed: images shape: (camera_count, 1, height, width, channels)'''
    global image_count
    global images

    if preview:
        print("Press 'space' to take a picture and 'q' to quit.")

    proccessed_images = []
    calculated_points = []
    for i in range(0, camera_count):
        image, image_point = _find_dot(images[i][0])
        calculated_points.append(image_point)
        proccessed_images.append(image)

    if preview:
        top = np.hstack([proccessed_images[0], np.hstack([proccessed_images[1], proccessed_images[2]])])
        bottom = np.hstack([proccessed_images[3], np.hstack([proccessed_images[4], proccessed_images[5]])])
        image = np.vstack([top, bottom])
        cv.imshow("Image", image)
        key = cv.waitKey(0) & 0xFF
        if key == ord(' '):
            print("Getting points...")

        if key == ord('q'):
            print("Exiting...")
            cv.destroyAllWindows()
            quit()
    
    if debug:
        print("Image points for camera", i, ":", image_points)

    cv.destroyAllWindows()
    return np.array(calculated_points)

if __name__ == "__main__":
    get_pose_images()
    capture_pose_points(preview=True)