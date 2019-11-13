import numpy as np
import cv2
from matplotlib import pyplot as plt
def sift_match(img1,img2):
    ''' Given two images, get the sift features and descriptors
        Compute the correspondences between features by Knn
        Set a threshold and get the good correspondences
        Draw the lines representing good matches
    '''
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    cv2.imwrite("result/SIFT_left_features.png",cv2.drawKeypoints(img1,kp1,np.array([]),
                (0,0,255), flags = 2))
    cv2.imwrite("result/SIFT_right_features.png",cv2.drawKeypoints(img2,kp2,np.array([]),
                (0,0,255), flags = 2))

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    pts1 = []
    pts2 = []
    for m,n in matches:
        if m.distance < 0.55*n.distance:
            good.append([m])
            pts1.append(kp1[m.trainIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags = 2)
    cv2.imwrite("result/SIFT_matching_result.png", img3)
    return good, pts1, pts2

def draw_matches(img1, img2, pts1, pts2):
    # Get the size of Images
    color = tuple(np.random.randint(0,255,3).tolist())
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    # print(img1.shape)
    # Create a canvas to draw matches
    canvas = np.ones((max(h1, h2), w1+w2+100), np.uint8)
    canvas.fill(255)
    # Draw the two pictures in the canvas
    for i in range(len(pts1)):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        x2 = x2 + w1 + 100
        img1 = cv2.line(canvas, (x1, y1), (x2,y2), color, 1)
    cv2.imwrite("result/inliers.png", canvas)
    # print(img1.shape)
    # print(img2.shape)
    # print(pts1)


def find_fundamental(matches, pts1, pts2):
    ''' Given the good matches and two images
        Compute the Fundamental Matrix by RANSAC
        Draw the inliners correspondences
        Store the F matrix into a file
    '''
    minNumber = 8
    if len(matches) < minNumber:
        return ArgumentDefaultsHelpFormatter
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    # Get fundamental matrix by RANSAC
    F1, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
    # Find the inliners
    # If the points are inliners, the mask is 1
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    file=open('result/pts1_inliers.txt','w')
    file.write(str(pts1))
    file.close()
    file=open('result/pts2_inliers.txt','w')
    file.write(str(pts2))
    file.close()
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags = 2)
    # cv2.imwrite("result/RANSAC_inliners_matching.png", img3)
    F2, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
    file=open('result/FundamentalMat.txt','w')
    file.write(str(F2))
    file.close()
    return F2, pts1, pts2

def draw_lines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1

def get_disparity(img1, img2):
    """ Compute the disparity map by Semi-Global Block Matching
        First compute the Horizontal and Vertical graident
        And filter the gradient to get the disparity
    """
    stereo = cv2.StereoSGBM_create(numDisparities=32, blockSize=15, P1 = 8*3*3**2, P2 = 64*3*3**2, disp12MaxDiff = 1, uniquenessRatio = 10, speckleWindowSize = 100, speckleRange = 32)
    disparity = stereo.compute(img1,img2).astype(np.float32) / 10.0
    # Normalize the disparity map into 0-255
    min_val, max_val, min_idx, max_idx = cv2.minMaxLoc(disparity)
    cv2.imwrite("result/disparity_original.png", disparity)
    cv2.imwrite("result/disparity.png", disparity*(255/max_val))
    # Compute the horizontal and vertical disparity
    sobelx = cv2.Sobel(disparity*(255/max_val),cv2.CV_64F,1,0,ksize=5) # kernal_size=3
    cv2.imwrite("result/Horizontal_disp.png", sobelx+60)
    sobely = cv2.Sobel(disparity*(255/max_val),cv2.CV_64F,0,1,ksize=5)
    cv2.imwrite("result/Vertical_disp.png", sobely+60)

def get_hor_ver(img1, img2, pts1, pts2):
    ptsx = []
    ptsy = []
    pts = pts1-pts2
    for i in pts:
        ptsx.append(i[0])
        ptsy.append(i[1])
    x_max = np.max(np.abs(ptsx))
    y_max = np.max(np.abs(ptsy))
    print(x_max, y_max)
    print(len(pts))
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    canvas = np.ones((h1, w1), np.uint8)
    canvas.fill(255)


if __name__ == '__main__':
    # Load images
    # img1 = cv2.imread('Cones_im2.jpg',0)  # queryImage
    # img2 = cv2.imread('Cones_im6.jpg',0) # trainImage
    img1 = cv2.imread('cast-left.jpg',0)  # queryImage
    img2 = cv2.imread('cast-right.jpg',0) # trainImage
    # Get matches and points
    good, pts1, pts2 = sift_match(img1,img2)
    # Get Fundamental Matrix
    F, pts1, pts2 = find_fundamental(good, pts1, pts2)
    # Display the inlier correspondences
    draw_matches(img1, img2, pts1, pts2)
    # Draw the epilines
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2),1, F)
    lines2 = lines1.reshape(-1, 3)
    # Store the epilines images
    img1_epi = draw_lines(img1, img2, lines1, pts1, pts2)
    img2_epi = draw_lines(img2, img1, lines2, pts2, pts1)
    cv2.imwrite("result/left_with_epilines.png", img1_epi)
    cv2.imwrite("result/right_with_epilines.png", img2_epi)
    # Compute the disparity and store the results
    get_disparity(img1, img2)
    # COmpute the horizontal and vertical disparity
    get_hor_ver(img1, img2, pts1, pts2)
