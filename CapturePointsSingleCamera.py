import cv2 as cv
from blur import fast_cuda_blur
import threading

cuda_lock = threading.Lock()

cuda_lock = threading.Lock()

def image_filter(image,camera_number=0):
    '''output: filtered image
    prerequisites: camera_params distortion'''
    # global camera_params
    # if camera_params is None:
    #     print("Camera parameters not found.")
    #     quit()
    # image = cv.undistort(image, np.array(camera_params[camera_number]["intrinsic_matrix"]), np.array(camera_params[camera_number]["distortion_coef"]))
    # image = cv.medianBlur(image, 5)
    with cuda_lock:
        image =  fast_cuda_blur(image, kernel_size=5)
    image = cv.threshold(image, 255*0.85, 255, cv.THRESH_BINARY)[1]
    image = cv.medianBlur(image, 5)
    return image

def find_points(img):
    '''output: image with dot and dot coordinates
    prerequisites needed: image'''
    # grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # img = cv.threshold(img, 255*0.2, 255, cv.THRESH_BINARY)[1]
    contours,_ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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

    return img#, image_points