import cv2 as cv
from dotenv import load_dotenv
import os
import json
import numpy as np
from Helpers import triangulate_points

load_dotenv()
filename = os.getenv("CAMERA_PARAMS_IN")
cam_params = open(filename)
camera_params = json.load(cam_params)

Image_points_camera_0 = np.array([[382, 489], [691, 411], [528, 312], [281, 356], [269, 181], [530, 157], [707, 213], [379, 260], [267, 182], [253, 85], [530, 49], [705, 108], [374, 190], [470, 382], [470, 219], [469, 96], [750, 353], [364, 167], [60, 263], [753, 179]])
Image_points_camera_1 = np.array([[244, 420], [552, 496], [658, 362], [411, 320], [407, 164], [662, 189], [558, 265], [224, 223], [403, 164], [396, 79], [672, 67], [561, 140], [201, 161], [465, 389], [465, 227], [464, 104], [257, 380], [614, 177], [175, 187], [863, 267]])

F_RANSAC,_ = cv.findFundamentalMat(Image_points_camera_0, Image_points_camera_1, cv.FM_RANSAC, 10, 0.99999)
F_LMEDS,_ = cv.findFundamentalMat(Image_points_camera_0, Image_points_camera_1, cv.FM_LMEDS)
F_8POINT,_ = cv.findFundamentalMat(Image_points_camera_0, Image_points_camera_1, cv.FM_8POINT)
F_7POINT,_ = cv.findFundamentalMat(Image_points_camera_0, Image_points_camera_1, cv.FM_7POINT)

for i in range(len(Image_points_camera_0)):
    x1 = np.array([Image_points_camera_0[i][0], Image_points_camera_0[i][1], 1])
    x2 = np.array([Image_points_camera_1[i][0], Image_points_camera_1[i][1], 1])
    l_ransac = x1.T @ F_RANSAC @ x2
    print("x1.T @ F @ x2 RANSAC: ",l_ransac)
    l_8point = x1.T @ F_8POINT @ x2
    print("x1.T @ F @ x2 8_POINT: ",l_8point)
    l_7point = x1.T @ F_7POINT @ x2
    print("x1.T @ F @ x2 7_POINT: ",l_7point)
    l_lmeds = x1.T @ F_LMEDS @ x2
    print("x1.T @ F @ x2 LMEDS: ",l_lmeds)
    print("Best Fundamental matrix: ",min(np.absolute(np.array([l_ransac,l_8point,l_7point,l_lmeds]))))
    print("")