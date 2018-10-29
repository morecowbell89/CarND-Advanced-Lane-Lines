# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 21:09:05 2018

@author: steven

cam_cal: Takes glob of images located in ../camera_cal
finds points from chessboard image to calculate the distortion correction
coefficients for the camera

meant for a 9,6 chessboard pattern
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

# Read in and make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# Arrays to store object points and image points from all the images

objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image space

# prepare object points
nx = 9# the number of inside corners in x
ny = 6# the number of inside corners in y

# Prepare object points, like (0,0,0), (1,0,0),(2,0,0) ...,(8,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinates


for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray,(nx, ny), None)
    
    # If found, add object points, image points
    if ret == True:
        print('working on ', fname)
        imgpoints.append(corners)
        objpoints.append(objp)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        write_name = '../camera_cal/corners_found/corners_found'+str(idx)+'.jpg'
        cv2.imwrite(write_name,img)

img = cv2.imread('../camera_cal/calibration1.jpg')
img_size = (img.shape[1],img.shape[0])

# find camera cal from cal images objpoint and imgpoints
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

# save the camera calibration results for later
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open('./calibration_pickle.p','wb'))