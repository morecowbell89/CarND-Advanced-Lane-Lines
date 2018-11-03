# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 22:21:18 2018

@author: steven

image_gen performs color/derivative thresholding and applies perspective
transform
"""

import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in saved objpoints and imagepoints
dist_pickle = pickle.load(open('./calibration_pickle.p','rb'))
mtx = dist_pickle["mtx"]
dist = dist_pickle['dist']

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
      
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    grad_dir = np.arctan2(np.absolute(sobely),np.absolute(sobelx))
    
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & \
                  (grad_dir <= thresh[1])] =1 
    
    return binary_output

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3,thresh=(0,255)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        pass
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) &
                  (scaled_sobel <= thresh[1])] = 1
    return binary_output

def img_masked_trapz(img,bottom_left=(0,1280),top_left=(0,0),top_right=(720,0),bottom_right=(720,1280)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #grayscale conversion
    
    mask = np.zeros_like(gray)
    ignore_mask_color = 1
    vertices = np.array([[bottom_left,top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    return mask
    

def img_thresh_pipeline(img,s_thresh=(75,255),l_s_filter=45,l_above=250,sx_thresh=(50,100)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ls_binary = np.zeros_like(gray)
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    L = hls[:,:,1]
    S = hls[:,:,2]
    
    ls_binary[((S > s_thresh[0]) &
              (S <= s_thresh[1]) &
              (L >= l_s_filter)) |
              (L >= l_above)] = 1
    
    # take derivative of x in L
    # Sobel x
    sobelx = cv2.Sobel(L, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    return ls_binary | sxbinary

img = mpimg.imread('../test_images/test2.jpg')

# Unwarp camera image
undistorted = cv2.undistort(img,mtx,dist,None,mtx)

color_thresh = img_thresh_pipeline(undistorted,s_thresh=(120,255),sx_thresh=(50,255))

# find mask
masked_color = img_masked_trapz(undistorted)
# Apply perspective transform
src = np.float32([[263, 670], [577, 460], [705, 460], [1044, 670]])
dst = np.float32([[320, 720], [320, 0], [950, 0], [950, 720]])
#src = np.float32([[585,460],[203,720],[1127,720],[695,460]])
#dst = np.float32([[320,0],[320,720],[960,720],[960,0]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)  # Use for later
warped = cv2.warpPerspective(color_thresh, M,
                             color_thresh.shape[::-1],
                             flags=cv2.INTER_LINEAR)

plt.imshow(warped)