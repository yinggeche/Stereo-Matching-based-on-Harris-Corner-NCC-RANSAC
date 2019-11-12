import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('cast-left.jpg',0)  # queryImage
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

    # Sort them in order of distance and pick the first 8 matches
    # matches_2 = bf.match(des1,des2)
    # matches_2 = sorted(matches_2, key = lambda x:x.distance)
    # eight = matches_2[:8]
    # img4 = cv2.drawMatches(img1,kp1,img2,kp2,eight,None,flags = 2)
    # cv2.imwrite("SIFT_matching_8pts.jpg", img4)
    # plt.imshow(img4),plt.show()

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.55*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags = 2)
    cv2.imwrite("SIFT_matching_result.jpg", img3)
    # plt.imshow(img3),plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
def get_disparity(img1, img2):
    # stereo = cv2.StereoSGBM_create( numDisparities=32, blockSize=15, P1 = 8*3*3**2, P2 = 32*3*3**2, disp12MaxDiff = 1, uniquenessRatio = 10, speckleWindowSize = 100, speckleRange = 32)
    stereo = cv2.StereoSGBM_create(numDisparities=32, blockSize=15, P1 = 8*3*3**2, P2 = 32*3*3**2, disp12MaxDiff = 1, uniquenessRatio = 10, speckleWindowSize = 100, speckleRange = 32)
    disparity = stereo.compute(img1,img2).astype(np.float32)
    # / 16.0
    # Normalize the disparity map into 0-255
    min_val, max_val, min_idx, max_idx = cv2.minMaxLoc(disparity)
    cv2.imwrite("result/disparity.png", disparity*(255/max_val))
    # cv2.imwrite("result/disparity_1.png", disparity+60)
    # plt.imshow(disparity),plt.show()
    # cv2.imwrite("result/disparity_2.png",disparity)
