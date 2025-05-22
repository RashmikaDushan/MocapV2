import cv2
import numpy as np
import glob
import json

def calculate_camera_intrinsics(wait_time=1):
    """
    Calculate camera intrinsics using checkerboard images.

    Parameters:
        wait_time (int) : Waiting time until a key press

    Returns:
        tuple: Camera matrix and distortion coefficients.
    """
    cam_images_folder_name = "checkerboard" # Folder name containing images

    CHECKERBOARD = (6,9) # Use only 6x9 checkerboard for calibration
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objpoints = []
    imgpoints = []
    
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    
    image_names = glob.glob(f'./{cam_images_folder_name}/*.png')

    print(image_names)
    window_name = "Checkerboad"
    window = cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for fname in image_names:
        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH)
        print("Found the checker pattern? ",ret)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
        cv2.imshow(window_name,img)

        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'):
            print("Exiting...")
            cv2.destroyAllWindows()
            quit()
    
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    return mtx,dist

def save_intrinsics(mtx,dist):
    """
    Save the camera intrinsics to a JSON file.

    Parameters:
        mtx (list[list[float]]): Camera matrix (2D list).
        dist (list[list[float]]): Distortion coefficients (1xN list).

    Returns:
        bool: True if saved successfully, False otherwise.
    """
    mtx = mtx.tolist()
    dist = dist.tolist()
    intrinsics = {"intrinsic_matrix":mtx,"distortion_coef":dist[0]}
    intrinsic_filename = f"./jsons/camera-intrinsics.json"

    try:
        with open(intrinsic_filename, "w") as outfile:
            json.dump(intrinsics, outfile)
        return True
    except Exception as e:
        print(f"Error saving intrinsics: {e}")
        return False
    

if __name__ == "__main__":
    mtx,dist = calculate_camera_intrinsics(1)
    save_intrinsics(mtx,dist)