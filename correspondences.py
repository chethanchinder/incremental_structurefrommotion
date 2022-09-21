import numpy as np
import cv2
import matplotlib.pyplot as plt
import motion
import os

def getKpOpticalFLow(im1, im2):
	## convert RGB to gray
	if(len(im1.shape)==3):
		cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
	if(len(im2.shape)==3):
		cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

	ffd = cv2.FastFeatureDetector_create()
	feat1 = ffd.detect(im1)
	p1 = cv2.KeyPoint_convert(feat1)
	p2, st, err = cv2.calcOpticalFlowPyrLK(im1, im2,p1,None)
	## test flow
	good_left =[]
	good_right= []
	for i,st_ in enumerate(st):
		if st_ ==1:
			good_left.append(p1[i])
			good_right.append(p2[i])
	return good_left, good_right

def getKpHighLevelFeatures(im1, im2):
	## iniitializing the SIFT detector
	sift = cv2.SIFT_create()
	## find the sift keypoints and descriptors for images using SIFT
	kp1, des1 = sift.detectAndCompute(im1, None)
	kp2, des2 = sift.detectAndCompute(im2, None)
	##FLANN based matcher
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	saerch_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params, saerch_params)
	matches = flann.knnMatch(des1, des2, k=2)
	goodMatches = []

	for m,n in matches:
		if m.distance < n.distance*0.75:
			goodMatches.append(m)

	# drawParms = dict(matchColor = (0,255,0),
	# 				 singlePointColor =(255,0,5))
	#
	# im3 = cv2.drawMatches(im1, kp1,im2, kp2,goodMatches,None,**drawParms)
	# plt.imshow(im3)
	# plt.show()
	return kp1, kp2, goodMatches


