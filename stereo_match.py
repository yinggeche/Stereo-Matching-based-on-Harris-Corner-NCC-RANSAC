import sys
import argparse
import math
import numpy as np
import cv2

def harris_corner(filename):
    """ Find interesting corner features based on Harris Corner Detection
    """
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Convert image into gray scale
    gray = np.float32(gray)
    # Convert image into float32
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    #print(dst)
    #   img[dst>0.00000001*dst.max()]=[0,0,255]
    img[dst>0.01*dst.max()]=[0,0,255]

    # cv2.imshow('dst',img)
    return img
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

def sift_feature(image):
    """ Find intersting corner features based on SIFT
    """
    pass

def ncc_match(image1, corner1, image2, corner2):
    """ Match the features in two images by NCC
    """
    pass

def ransac():
    pass

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description = 'stereo Match Exercise,  return the corner features, Fundamental Matrix, and dense disparity map images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('input1',default=None, help='Input the first image')
    # parser.add_argument('input2',default=None, help='Input the second image')
    # args = parser.parse_args()

    cv2.imwrite("left_corner.jpg",harris_corner("cast-left.jpg"));
    cv2.imwrite("right_corner.jpg",harris_corner("cast-right.jpg"));
