# Stereo-Matching-based-on-Harris-Corner-NCC-RANSAC
### Step 1: Find the interesting features and correspondences between the images
1. Get the corner features by using Harris Corner Detection and NCC algorithms
2. Can also use SIFT features
### Step 2: Estimate the Fundamental Matrix
1. Use the correspndences above
2. Use RANSAC to elimate outliners
3. Remain the inliners' correspondences
### Step 3: Compute the dense disparity map
1. Use the Fundamental matrix
2. Display the results by three images

# Language
Python

# Libaries
OpenCV
Numpy
