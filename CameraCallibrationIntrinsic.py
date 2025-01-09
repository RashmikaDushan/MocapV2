#!/usr/bin/env python
 
import cv2
import numpy as np
import os
import glob
from dotenv import load_dotenv
 
load_dotenv()

cam_images_folder_name = os.getenv("CHECKERBOARD_FOLDERNAME") # Folder name containing images
cam_images_folder_name_calibrated = f'{cam_images_folder_name}_c'

# Defining the dimensions of checkerboard
# CHECKERBOARD = (6,9)
CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []
 
 
# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
 
# Extracting path of individual image stored in a given directory
images = glob.glob(f'./{cam_images_folder_name}/*.png')
print(images)
for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    # ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,None)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH)
    print("Found the checker pattern? ",ret)
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
         
        imgpoints.append(corners2)
 
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
     
    cv2.imshow('img',img)

    # Wait for a key press
    key = cv2.waitKey(0) & 0xFF

    # If 'q' is pressed, exit
    if key == ord('q'):  # 'q' key
        print("Exiting...")
        cv2.destroyAllWindows()
        quit()
    
    # new_frame_name = cam_images_folder_name_calibrated + '/' + os.path.basename(fname)
    # print(new_frame_name)
    # cv2.imwrite(new_frame_name, img)

 
cv2.destroyAllWindows()
 
"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
 
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)