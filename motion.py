import numpy as np
import cv2
from correspondences import getKpHighLevelFeatures
import os

def getEssentialMat(points1, points2 , cameraCalib):
	## get the kps in terms of point2f
	essentialMat, mask = cv2.findEssentialMat(np.array(points1), np.array(points2),cameraCalib,cv2.FM_RANSAC, 0.99,0.1)
	return essentialMat, mask

def getCameraFromEssential(E):
	## calculautes R, T from SVD
	U, sigma, Vt = np.linalg.svd(E)
	T1 = U[:,-1].reshape((3,1))
	T2 = -T1
	W = np.array([[0, -1, 0],[1, 0, 0],[0, 0 ,1]])
	R1 = U @ W @ Vt
	R2 = U @ np.transpose(W) @ Vt

	return  R1, R2, T1, T2

def getEssentialAndInliers(im1, im2):
	kp1, kp2, matches = getKpHighLevelFeatures(im1,im2)
	root_dir = "Benchmarking_Camera_Calibration_2008/"

	camera_calib_path = os.path.join(root_dir, 'castle-P19/images/K.txt')
	K = np.loadtxt(camera_calib_path)
	points1 = []
	points2 = []
	for i in range(len(matches)):
		points1.append(kp1[matches[i].queryIdx].pt)
		points2.append(kp2[matches[i].trainIdx].pt)
	essentialMat, mask = getEssentialMat(points1, points2,K)
	inliers1= []
	inliers2 =[]
	for indx,val in enumerate(mask.ravel()):
		if(val ==1):
			inliers1.append(points1[indx])
			inliers2.append(points2[indx])
	return inliers1, inliers2, essentialMat

def isRotationValid(R, thresh = 1e-9):
	if np.abs(np.linalg.det(R)) - 1  < thresh:
		return True
	else:
		return False
