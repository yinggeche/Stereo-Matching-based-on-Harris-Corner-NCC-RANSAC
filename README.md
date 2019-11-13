# Stereo-Matching-based-on-Harris-Corner-NCC-RANSAC
### Step 1: Find the interesting features and correspondences between the images
1. Get the corner features by using Harris Corner Detection and NCC algorithms
2. Can also use SIFT features and descriptors
3. Match the features using lines
### Step 2: Estimate the Fundamental Matrix
1. Use the correspondences above
2. Use RANSAC to eliminate outliners
3. Display the inliers correspondences
4. Draw the epi-lines in both images
5. Compute the Fundamental Matrix by 8 Points Algorithms
### Step 3: Compute the dense disparity map
1. Use the Fundamental matrix to reduce the search space
2. Compute the disparity map based on Semi-Global Block Matching
3. Display the results by three images
    1. Vertical disparity component
    2. Horizontal disparity component
    3. Disparity vector using color(0-255), direction coded by hue, length coded by saturation

# Language
Python

# Libaries
1. OpenCV
2. Numpy
3. Matplotlib
