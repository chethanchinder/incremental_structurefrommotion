import bz2
import os
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
class BundleAdjustment:
	def __init__(self, points_3d, pixel_points, camera_matrices, K, images_names, camera_indices, point_indices):
		self.points_3d = np.array(points_3d)
		self.pixel_points = np.array(pixel_points)
		self.camera_matrices = np.array(camera_matrices)
		self.K = K
		self.n_cameras =  self.camera_matrices.shape[0]
		self.n_points = self.points_3d.shape[0]
		self.camera_indices  = np.array(camera_indices)
		self.point_indices = np.array(point_indices)

	def project(self, points, camera_params):
		points_proj_cart = []
		for camera_param, point in zip(camera_params, points):
			point_proj_hom = np.matmul(self.K, np.matmul(camera_param[:,:3], point))
			point_proj_cart = point_proj_hom[:2]/point_proj_hom[2]
			points_proj_cart.append(point_proj_cart)
		return np.reshape(points_proj_cart, (-1,2))

	def get_residual(self,params,points_3d,points_2d, camera_params ):
		point_2d_projected = self.project(points_3d[self.point_indices], camera_params[self.camera_indices])
		return (points_2d - point_2d_projected).ravel()

	def ba_sparsity_matrix(self,  n_cameras, n_points,camera_indices, point_indices):
		# m is number of 3d points
		m = len(camera_indices)*2
		# consider the 3D points from all the camer views
		print(f" m is {m}")
		n = n_cameras*12 + n_points*3
		print(f" n is {n}")
		A = lil_matrix((m,n), dtype= int)
		camear_idx = np.arange(len(camera_indices))
		for s in range(12):
			A[2*camear_idx, camera_indices*12 + s] = 1
			A[2*camear_idx + 1, camera_indices*12 + s] = 1
		for s in range(3):
			A[2 * camear_idx, n_cameras*12 + point_indices*3 + s] = 1
			A[2 * camear_idx + 1, n_cameras*12 + point_indices*3 + s] = 1
		return A

	def bundle_adjust(self):
		params = np.hstack((self.camera_matrices.ravel(), self.points_3d.ravel()))
		inital_err= self.get_residual(params,self.points_3d,self.pixel_points,self.camera_matrices)
		plt.plot(inital_err)
		sparse_matrix = self.ba_sparsity_matrix(self.n_cameras, self.n_points, self.camera_indices,self.point_indices)

		t0 = time.time()
		res = least_squares(self.get_residual, params, jac_sparsity=sparse_matrix, verbose=2, x_scale='jac', ftol=1e-5, method='trf',
							args=(self.points_3d, self.pixel_points, self.camera_matrices ))


		t1 = time.time()
		print(f" optimization took {t1-t0} seconds")
		plt.plot(res.fun)
		#plt.show()
		return  res.x[:self.n_cameras*12].reshape((self.n_cameras, 3,4)), res.x[self.n_cameras*12:].reshape((self.n_points, 3))
