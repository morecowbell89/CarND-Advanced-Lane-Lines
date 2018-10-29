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

# Read in saved objpoints and imagepoints
dist_pickle = pickle.load(open('./calibration_pickle.p','rb'))
mtx = dist_pickle["mtx"]
dist = dist_pickle['dist']
