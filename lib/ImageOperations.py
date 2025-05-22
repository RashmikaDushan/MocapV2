import cv2 as cv
from lib.CudaOperations import fast_cuda_blur
import threading
import numpy as np
import json



cuda_lock = threading.Lock()

intrinsics_json = "./jsons/camera-params-in.json"
camera_params_file = open(intrinsics_json)
camera_params = json.load(camera_params_file)

def image_filter_cpu(image,camera_number=0):
    '''output: filtered image
    prerequisites: camera_params distortion'''
    global camera_params
    image = cv.medianBlur(image, 5)
    image = cv.threshold(image, 255*0.85, 255, cv.THRESH_BINARY)[1]
    return image

def image_filter_gpu(image,camera_number=0):
    '''output: filtered image
    prerequisites: camera_params distortion'''
    global camera_params
    with cuda_lock:
        image =  fast_cuda_blur(image, kernel_size=5)
    image = cv.threshold(image, 255*0.85, 255, cv.THRESH_BINARY)[1]
    image = cv.medianBlur(image, 5)
    return image

def _find_dot(img,print_location=False,return_filtered=False):
    '''output: image with dot and dot coordinates
    prerequisites needed: image'''
    # grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    camera_number = 0
    img = cv.undistort(img, np.array(camera_params[camera_number]["intrinsic_matrix"]), np.array(camera_params[camera_number]["distortion_coef"]))
    # img = cv.undistort(img, np.array(camera_params[camera_number]["intrinsic_matrix"]), np.array(camera_params[camera_number]["distortion_coef"]))
    grey = image_filter_gpu(img)
    all_contours,_ = cv.findContours(grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = []
    for cnt in all_contours:
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        if perimeter:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            # print("Area:", area)
            # print("Perimeter:", circularity)
            if circularity > 0.5 and area > 500:
                contours.append(cnt)
    if return_filtered:
        img = cv.drawContours(grey, contours, -1, (0,0,255), 4)
    else:
        img = cv.drawContours(img, contours, -1, (0,0,255), 4)

    image_points = []
    for contour in contours:
        moments = cv.moments(contour)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            # cv.putText(img, f'({center_x}, {center_y})', (center_x-240,center_y - 15), cv.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 4)
            # cv.circle(img, (center_x,center_y), 2, (0,255,0), 8)
            image_points.append([center_x, center_y])

    if len(image_points) > 0:
        for i in range(len(image_points)):
            if print_location:
                cv.putText(img, f'({image_points[i][0]}, {image_points[i][1]})', (image_points[i][0]-240,image_points[i][1] - 15), cv.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 4)
            else:
                cv.putText(img, str(i), (image_points[i][0]-20,image_points[i][1] - 15), cv.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 4)
            cv.circle(img, (image_points[i][0],image_points[i][1]), 2, (0,255,0), 8)

    if len(image_points) == 0:
        image_points = [[None, None]]

    return img, image_points