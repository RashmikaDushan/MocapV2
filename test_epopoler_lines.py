import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the left and right images in grayscale
imgLeft = cv2.imread('./test_images/cam1_full.jpg', 0)
imgRight = cv2.imread('./test_images/cam2_full.jpg', 0)

# # Detect SIFT keypoints and compute descriptors
# sift = cv2.SIFT_create()
# keyPointsLeft, descriptorsLeft = sift.detectAndCompute(imgLeft, None)
# keyPointsRight, descriptorsRight = sift.detectAndCompute(imgRight, None)

# # Create FLANN matcher object
# FLANN_INDEX_KDTREE = 0
# indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# searchParams = dict(checks=50)
# flann = cv2.FlannBasedMatcher(indexParams, searchParams)

# # Match descriptors using KNN
# matches = flann.knnMatch(descriptorsLeft, descriptorsRight, k=2)

# # Apply ratio test
# goodMatches = []
# ptsLeft = []
# ptsRight = []

# for m, n in matches:
#     if m.distance < 0.8 * n.distance:
#         goodMatches.append([m])
#         ptsLeft.append(keyPointsLeft[m.queryIdx].pt)
#         ptsRight.append(keyPointsRight[m.trainIdx].pt)

# Convert points to integer
ptsLeft = np.int32([[382, 489], [691, 411], [528, 312], [281, 356], [269, 181], [530, 157], [707, 213], [379, 260], [267, 182], [253, 85], [530, 49], [705, 108], [374, 190], [470, 382], [470, 219], [469, 96], [750, 353], [364, 167], [60, 263], [753, 179]])
ptsRight = np.int32([[244, 420], [552, 496], [658, 362], [411, 320], [407, 164], [662, 189], [558, 265], [224, 223], [403, 164], [396, 79], [672, 67], [561, 140], [201, 161], [465, 389], [465, 227], [464, 104], [257, 380], [614, 177], [175, 187], [863, 267]])

# ptsLeft = np.int32([[382, 489], [691, 411], [528, 312], [281, 356], [269, 181], [530, 157], [707, 213], [379, 260]])
# ptsRight = np.int32([[244, 420], [552, 496], [658, 362], [411, 320], [407, 164], [662, 189], [558, 265], [224, 223]])

print("Points in left image: ", ptsLeft)
print("Points in right image: ", ptsRight)

# Compute the fundamental matrix
# F, mask = cv2.findFundamentalMat(ptsLeft, ptsRight, cv2.FM_RANSAC,10,0.99999)
# F, mask = cv2.findFundamentalMat(ptsLeft, ptsRight, cv2.FM_7POINT)
F, mask = cv2.findFundamentalMat(ptsLeft, ptsRight, cv2.FM_LMEDS)

print("Fundamental matrix: ", F)

# Select only inlier points
# ptsLeft = ptsLeft[mask.ravel() == 1]
# ptsRight = ptsRight[mask.ravel() == 1]

# Function to draw epilines
def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = np.random.randint(0, 255, 3).tolist()
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(map(int, pt1)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(map(int, pt2)), 5, color, -1)
    return img1, img2


# Find epilines for points in the right image and draw them on the left image
linesLeft = cv2.computeCorrespondEpilines(ptsRight.reshape(-1, 1, 2), 2, F)
linesLeft = linesLeft.reshape(-1, 3)
img5, img6 = drawlines(imgLeft, imgRight, linesLeft, ptsLeft, ptsRight)

# Find epilines for points in the left image and draw them on the right image
linesRight = cv2.computeCorrespondEpilines(ptsLeft.reshape(-1, 1, 2), 1, F)
linesRight = linesRight.reshape(-1, 3)
img3, img4 = drawlines(imgRight, imgLeft, linesRight, ptsRight, ptsLeft)

# Display the images
plt.subplot(121), plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)), plt.title("Epilines on Left Image")
plt.subplot(122), plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)), plt.title("Epilines on Right Image")
plt.show()

for i in range(len(ptsLeft)):
    x1 = np.array([ptsLeft[i][0], ptsLeft[i][1], 1])
    x2 = np.array([ptsRight[i][0], ptsRight[i][1], 1])
    print("x1: ",x1)
    print("x2: ",x2)
    print("x1.T @ F @ x2: ",x1.T @ F @ x2)