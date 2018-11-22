#!/usr/bin/env python

'''
stereo2depth - generate a depth map from a stereoscopic image using OpenCV.

Usage:
    Run `python stereo2depth images/stereo.jpg`
    and the depth map will be saved to `images/stereo_depth.jpg`
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import sys, os

def stereo2depth(filename):

    # Parameters from all steps are defined here to make it easier to adjust values.
    resolution     = 1.0    # (0, 1.0]
    numDisparities = 16     # has to be dividable by 16
    blockSize      = 5      # (0, 25]
    windowSize     = 5      # Usually set equals to the block size
    filterCap      = 63     # [0, 100]
    lmbda          = 80000  # [80000, 100000]
    sigma          = 1.2
    brightness     = 0      # [-1.0, 1.0]
    contrast       = 1      # [0.0, 3.0]

    # Step 1 - Load the input stereoscopic image
    img = cv2.imread(filename)

    # Step 2 - Slice the input image into the left and right views.
    height, width = img.shape[:2]
    imgL = img[0:int((height / 2)), 0:width]
    imgR = img[int((height / 2)):height, 0:width]

    # Step 3 - Downsampling the images to the resolution level to speed up the matching at the cost of quality degradation.
    resL = cv2.resize(imgL, None, fx = resolution, fy = resolution, interpolation = cv2.INTER_AREA)
    resR = cv2.resize(imgR, None, fx = resolution, fy = resolution, interpolation = cv2.INTER_AREA)

    # Step 4 - Setup two stereo matchers to compute disparity maps both for left and right views.
    left_matcher = cv2.StereoSGBM_create(
        minDisparity = 0,
        numDisparities = numDisparities,
        blockSize = blockSize,
        P1 = 8 * 3 * windowSize ** 2,
        P2 = 32 * 3 * windowSize ** 2,
        disp12MaxDiff = 1,
        uniquenessRatio = 15,
        speckleWindowSize = 0,
        speckleRange = 2,
        preFilterCap = filterCap,
        mode = cv2.STEREO_SGBM_MODE_HH # Run on HH mode which is more accurate than the default mode but much slower so it might not suitable for real-time scenario.
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # Step 5 - Setup a disparity filter to deal with stereo-matching errors.
    #          It will detect inaccurate disparity values and invalidate them, therefore making the disparity map semi-sparse.
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left = left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # Step 6 - Perform stereo matching to compute disparity maps for both left and right views.
    displ = left_matcher.compute(resL, resR)
    dispr = right_matcher.compute(resR, resL)

    # Step 7 - Perform post-filtering
    imgLb = cv2.copyMakeBorder(imgL, top = 0, bottom = 0, left = np.uint16(numDisparities / resolution), right = 0, borderType= cv2.BORDER_CONSTANT, value = [155,155,155])
    filteredImg = wls_filter.filter(displ, imgLb, None, dispr)

    # Step 8 - Adjust image resolution, brightness, contrast, and perform disparity truncation hack
    filteredImg = filteredImg * resolution
    filteredImg = filteredImg + (brightness / 100.0)
    filteredImg = (filteredImg - 128) * contrast + 128
    filteredImg = np.clip(filteredImg, 0, 255)
    filteredImg = np.uint8(filteredImg)
    filteredImg = cv2.resize(filteredImg, (width, int(height / 2)), interpolation = cv2.INTER_CUBIC) # Disparity truncation hack
    filteredImg = filteredImg[0:height, np.uint16(numDisparities / resolution):width]
    filteredImg = cv2.resize(filteredImg, (width,int(height / 2)), interpolation = cv2.INTER_CUBIC)  # Disparity truncation hack

    return filteredImg

if __name__ == '__main__':

    try:
        arg = sys.argv[1]
    except IndexError:
        print(__doc__)
        sys.exit()

    depthMap = stereo2depth(arg)

    pathname = os.path.dirname(arg)
    basename = os.path.basename(arg)
    filename = os.path.splitext(basename)[0] + '_depth.jpg'
    cv2.imwrite(os.path.join(pathname, filename), depthMap, [cv2.IMWRITE_JPEG_QUALITY, 100])
