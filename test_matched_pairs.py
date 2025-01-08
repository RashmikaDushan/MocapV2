import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the left and right images in grayscale
imgLeft = cv2.imread('./test_images/cam1_full.jpg', 0)
imgRight = cv2.imread('./test_images/cam2_full.jpg', 0)

# Convert points to integer
ptsLeft = np.array([[382, 489], [691, 411], [528, 312], [281, 356], [269, 181], [530, 157], [707, 213], [379, 260], [267, 182], [253, 85], [530, 49], [705, 108], [374, 190], [470, 382], [470, 219], [469, 96], [750, 353], [364, 167], [60, 263], [753, 179]])
ptsRight = np.array([[244, 420], [552, 496], [658, 362], [411, 320], [407, 164], [662, 189], [558, 265], [224, 223], [403, 164], [396, 79], [672, 67], [561, 140], [201, 161], [465, 389], [465, 227], [464, 104], [257, 380], [614, 177], [175, 187], [863, 267]])

# ptsLeft = np.int32([[382, 489], [691, 411], [528, 312], [281, 356], [269, 181], [530, 157], [707, 213], [379, 260]])
# ptsRight = np.int32([[244, 420], [552, 496], [658, 362], [411, 320], [407, 164], [662, 189], [558, 265], [224, 223]])

print("Points in left image: ", ptsLeft)
print("Points in right image: ", ptsRight)

# Compute the fundamental matrix
F, mask = cv2.findFundamentalMat(ptsLeft, ptsRight, cv2.FM_RANSAC,10,0.99999)   

print("Fundamental matrix: ", F)

# Select only inlier points
# ptsLeft = ptsLeft[mask.ravel() == 1]
# ptsRight = ptsRight[mask.ravel() == 1]

# Function to draw epilines
def drawlines(img1, img2, pts1, pts2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for i in range(len(pts1)): 
        print(pts1[i]," | ",pts2[i])
        color = tuple(np.random.randint(0, 255, 3).tolist())
        if i == 5:
            color = (255, 0, 0)
        img1 = cv2.circle(img1, pts1[i], 5, color, -1)
        img2 = cv2.circle(img2, pts2[i], 5, color, -1)
    return img1, img2

img7,img8 = drawlines(imgLeft, imgRight, ptsLeft, ptsRight)

# Display the images
plt.subplot(121), plt.imshow(cv2.cvtColor(img7, cv2.COLOR_BGR2RGB)), plt.title("Matches on Left Image")
plt.subplot(122), plt.imshow(cv2.cvtColor(img8, cv2.COLOR_BGR2RGB)), plt.title("Matches on Right Image")
plt.show()


for i in range(len(ptsLeft)):
    x1 = np.array([ptsLeft[i][0], ptsLeft[i][1], 1])
    x2 = np.array([ptsRight[i][0], ptsRight[i][1], 1])
    print("x1: ",x1)
    print("x2: ",x2)
    print("x1.T @ F @ x2: ",x1.T @ F @ x2)