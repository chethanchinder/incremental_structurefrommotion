import numpy as np
import cv2
from correspondences import getKpHighLevelFeatures
import os

def get_3D_points(P1,u1,P2,u2):
	A = np.array([[u1[0] * P1[2, 0] - P1[0, 0], u1[0] * P1[2, 1] - P1[0, 1], u1[0] * P1[2, 2] - P1[0, 2]],
				  [u1[1] * P1[2, 0] - P1[1, 0], u1[1] * P1[2, 1] - P1[1, 1], u1[1] * P1[2, 2] - P1[1, 2]],
				  [u2[0] * P2[2, 0] - P2[0, 0], u2[0] * P2[2, 1] - P2[0, 1], u2[0] * P2[2, 2] - P2[0, 2]],
				  [u2[1] * P2[2, 0] - P2[1, 0], u2[1] * P2[2, 1] - P2[1, 1], u2[1] * P2[2, 2] - P2[1, 2]]])

	B = np.array([-(u1[0] * P1[2, 3] - P1[0, 3]),
				  -(u1[1] * P1[2, 3] - P1[1, 3]),
				  -(u2[0] * P2[2, 3] - P2[0, 3]),
				  -(u2[1] * P2[2, 3] - P2[1, 3])])


	X = cv2.solve(A,B,flags=cv2.DECOMP_SVD)
	return X[1]

def getCameraFromEssential(E):
	## calculautes R, T from SVD
	U, sigma, Vt = np.linalg.svd(E)
	T1 = U[:,-1].reshape((3,1))
	T2 = -T1
	W = np.array([[0, -1, 0],[1, 0, 0],[0, 0 ,1]])
	R1 = U @ W @ Vt
	R2 = U @ np.transpose(W) @ Vt

	return  R1, R2, T1, T2

def calculateReprojectionError(rotation, translation, calib_matrix, point2D, point3D):
	reprojected_homg = np.matmul(calib_matrix, np.matmul(rotation, point3D) + translation)
	reprojected_2d = cv2.convertPointsFromHomogeneous(reprojected_homg.T)[:,0,:].T
	error = np.linalg.norm( np.reshape(point2D,(2,1)) - np.reshape(reprojected_2d, (2,1)))
	return error

def checkTriangulation(points_3D, camera_projection ):
	camera_projection_homg =np.vstack((camera_projection,np.array([0,0,0,1])))
	reprojected_points = cv2.perspectiveTransform(points_3D[np.newaxis], m=camera_projection_homg)
	depth_values = reprojected_points[0,:,-1]
	""" if the ratio of the positive depth values to total depth values is greater than threshold"""
	if(np.sum(depth_values > 0)/depth_values.shape[0] )< 0.75:
		return False
	else:
		return True

def getFundamentalAndInliers(view1, view2, match):
	kp1 = np.array([kp.pt for kp in view1.keypoints])[match.indices1]
	kp2 = np.array([kp.pt for kp in view2.keypoints])[match.indices2]
	essentialMatrix, mask = cv2.findFundamentalMat(kp1, kp2, method= cv2.FM_RANSAC,confidence= 0.99,ransacReprojThreshold= 0.9)
	mask = mask.astype(bool).flatten()
	match.inliers1 = np.array(match.indices1)[mask]
	match.inliers2 = np.array(match.indices2)[mask]
	return essentialMatrix

def isRotationValid(R, thresh = 1e-9):
	if np.abs(np.linalg.det(R)) + 1  < thresh:
		return False
	else:
		return True
