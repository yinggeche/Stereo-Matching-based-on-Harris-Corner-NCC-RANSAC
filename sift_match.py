import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('cast-left.jpg',0)          # queryImage
img2 = cv2.imread('cast-right.jpg',0) # trainImage

# Initiate SIFT detector
# sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
cv2.imwrite("SIFT_left_features.jpg",cv2.drawKeypoints(img1,kp1,np.array([]),
            (0,0,255), flags = 2))
cv2.imwrite("SIFT_right_features.jpg",cv2.drawKeypoints(img2,kp2,np.array([]),
            (0,0,255), flags = 2))

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.55*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags = 2)

cv2.imwrite("SIFT_matching_result.jpg", img3)

plt.imshow(img3),plt.show()
