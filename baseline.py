import glob
import numpy as np
import cv2
import os
import pickle
from motion_utils import *
class Baseline:
	def __init__(self, view1, view2, match,calib_matrix):
		self.view1 = view1
		self.view1.rotation = np.eye(3,3)
		self.view2 = view2
		self.match = match
		self.calibration_matrix = calib_matrix
		self.thresh_err = 100

	def getPose(self):
		F = getFundamentalAndInliers(self.view1, self.view2,self.match )
		essentialMatrix = self.calibration_matrix.T @ F @ self.calibration_matrix
		return self.checkPose(essentialMatrix)

	def checkPose(self, E):
		R1, R2, T1, T2 = getCameraFromEssential(E)
		if not isRotationValid(R1):
			R1, R2, T1, T2 = getCameraFromEssential(-E)
		print(f" R1, R2, T1, T2 {R1, R2, T1, T2}")
		projection_error, points_3d = self.triangulate(R1,T1)
		print(f" R1 T1 projection error {projection_error} traingulation check  {checkTriangulation(points_3d, np.hstack((R1, T1)))}")

		if(projection_error> self.thresh_err or not checkTriangulation(points_3d,np.hstack((R1,T1)))):
			projection_error, points_3d = self.triangulate(R1, T2)
			print(f" R1 T2 projection error {projection_error} traingulation check  {checkTriangulation(points_3d, np.hstack((R1, T2)))}")
			if(projection_error> self.thresh_err or not checkTriangulation(points_3d,np.hstack((R1,T2)))):
				projection_error, points_3d = self.triangulate(R2, T1)
				print(f" R2 T1 projection error {projection_error} traingulation check  {checkTriangulation(points_3d, np.hstack((R2, T1)))}")

				if (projection_error > self.thresh_err or not checkTriangulation(points_3d, np.hstack((R2, T1)))):
					print(" R2, T2")
					projection_error, points_3d = self.triangulate(R2, T2)
					print(f" R2 T2 projection error {projection_error} traingulation check  {checkTriangulation(points_3d, np.hstack((R2, T2)))}")
					return R2, T2

				else:
					print(" R2, T1")
					return R2, T1
			else:
				print(" R1, T2")
				return R1, T2
		else:
			print(" R1, T1")
			return R1, T1

	def triangulate(self, rotation, translation):
		K_inv = np.linalg.inv(self.calibration_matrix)
		P1= np.hstack((self.view1.rotation, self.view1.translation))
		P2= np.hstack((rotation, translation))

		pixel_points1 = np.array([kp.pt for kp in self.view1.keypoints])[self.match.inliers1]
		pixel_points2 = np.array([kp.pt for kp in self.view2.keypoints])[self.match.inliers2]

		pixel_points1_homg =  cv2.convertPointsToHomogeneous(pixel_points1)[:,0,:]
		pixel_points2_homg =  cv2.convertPointsToHomogeneous(pixel_points2)[:,0,:]

		reprojection_errors = []
		points_3d = np.zeros((0, 3))

		for id, pixel_point1 in enumerate(pixel_points1_homg):
			u1 = pixel_point1
			u2 = pixel_points2_homg[id]
			u1_normalized = np.matmul(K_inv, u1)
			u2_normalized = np.matmul(K_inv, u2)
			point_3d = get_3D_points(P1, u1_normalized, P2, u2_normalized)
			points_3d = np.concatenate((points_3d,point_3d.T), axis=0)
			error = calculateReprojectionError(rotation, translation, self.calibration_matrix, u2[:2], point_3d)
			reprojection_errors.append(error)
		return np.mean(reprojection_errors),points_3d

