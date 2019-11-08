import sys
import argparse
import math
import numpy as np
import cv2

def harris_corner(image):
    """ Find interesting corner features based on Harris Corner Detection
    """
    pass

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

    parser = argparse.ArgumentParser(description = 'stereo Match Exercise,  return the corner features, Fundamental Matrix, and dense disparity map images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input_file1',default=None, help='Input the first image')
    parser.add_argument('-input_file2',default=None, help='Input the second image')
    args = parser.parse_args()
