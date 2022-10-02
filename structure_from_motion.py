from baseline import Baseline
from motion_utils import *
from structureFromMotion_python.bundle_adjust import BundleAdjustment

class Structure:

	def __init__(self, views, matches, K):
		self.views = views
		self.matches = matches
		self.names = []
		self.completed = []
		self.K = K
		self.point_map = {}
		self.errors = []
		self.reconstructed_count = 0
		self.points_3d = np.zeros((0,3))
		self.colors = np.zeros((0,3))
		self.point3d_indices = []
		self.camera_indices = []
		self.camera_matrices  = []
		self.completed_pixel_points =[]
		self.BA_rate = 3
		for view in views:
			self.names.append(view.image_name)
		self.results_path = os.path.join(self.views[0].root_dir, 'reconstructed')
		if not os.path.exists(self.results_path):
			os.makedirs(self.results_path)

	def compute_pose(self, view1,view2=None):
		if view2:
			match = self.matches[(view1.image_name, view2.image_name)]
			baseline = Baseline(view1, view2, match, self.K)
			view2.rotation , view2.translation =  baseline.getPose()
			self.camera_matrices.append(np.hstack((view1.rotation, view1.translation)))
			self.camera_matrices.append(np.hstack((view2.rotation, view2.translation)))
			rpe1, rpe2= self.triangulate(view1, view2)
			self.errors.append(np.mean(rpe1))
			self.errors.append(np.mean(rpe2))
			self.completed.append(view1)
			self.completed.append(view2)

		else:
			view1.rotation, view1.translation = self.pnp_projection(view1)
			self.camera_matrices.append(np.hstack((view1.rotation, view1.translation)))
			errors = []
			for id, old_view in enumerate(self.completed):
				match = self.matches[(old_view.image_name,view1.image_name)]
				kp1 = np.array([kp.pt for kp in old_view.keypoints])[match.indices1]
				kp2 = np.array([kp.pt for kp in view1.keypoints])[match.indices2]
				F , mask = cv2.findFundamentalMat(kp1, kp2, method= cv2.FM_RANSAC,confidence= 0.99,ransacReprojThreshold= 0.9)
				mask = mask.astype(bool).flatten()
				match.inliers1 = np.array(match.indices1)[mask]
				match.inliers2 = np.array(match.indices2)[mask]

				self.remove_mapped_points(match,id)
				_, rpe = self.triangulate(old_view, view1)
				errors+=rpe
			self.completed.append(view1)
			self.errors.append(np.mean(errors))


	def remove_mapped_points(self, match, img_idx):
		## Removes inliers which are already constructed in the completed views

		inliers1 = []
		inliers2 = []
		for i, (inlier1, inlier2) in enumerate(zip(match.inliers1,match.inliers2)):
			if (img_idx,inlier1) not in self.point_map:
				inliers1.append(inlier1)
				inliers2.append(inlier2)
		match.inliers1 = inliers1
		match.inliers2 = inliers2

	def reconstruct(self):
		## compute baselines
		## replace lated using graphs for the best baseline
		print(f"[structure from motion] reconstructing from view 0  and view 1")
		self.compute_pose(self.views[0] , self.views[1])
		self.plot_points()
		for i, view in enumerate(self.views[2:]):
			print(f"[structure from motion] reconstructing from view {i + 2} usng pnp")
			self.compute_pose(view)
			## bundle adjust every {BA_rate} frames
			if(i%self.BA_rate==0):
				print(f" Bundle Adjusting ...")
				ba = BundleAdjustment(self.points_3d, np.array(self.completed_pixel_points), self.camera_matrices, self.K, self.names, self.camera_indices, self.point3d_indices)
				camera_matrices, self.points_3d =ba.bundle_adjust()
				self.camera_matrices = []
				for j,camera_matrice in enumerate(camera_matrices):
					R=camera_matrice[:,:3].reshape(3,3)
					T = camera_matrice[:,3].reshape(3,1)
					self.completed[j].rotation = R
					self.completed[j].translation = T
					self.camera_matrices.append(camera_matrice)
			self.plot_points()

	def plot_points(self):
		file_path = os.path.join(self.results_path, str(len(self.completed))+"_images.ply")
		# point_cloud = open3d.geometry.PointCloud()
		# print(f" points 3d size {self.points_3d.shape}")
		# point_cloud.points = open3d.utility.Vector3dVector(self.points_3d)
		# open3d.io.write_point_cloud(file_path, point_cloud)

		with open(file_path, "w") as f:
			f.write('ply\n')
			f.write('format ascii 1.0\n')
			f.write('element vertex {}\n'.format(self.points_3d.shape[0]))

			f.write('property float x\n')
			f.write('property float y\n')
			f.write('property float z\n')

			f.write('property uchar red\n')
			f.write('property uchar green\n')
			f.write('property uchar blue\n')

			f.write('end_header\n')
			for point_3d, color in zip(self.points_3d, self.colors):
				f.write('{} {} {} {} {} {}\n'.format(point_3d[0],point_3d[1],point_3d[2],
													 int(color[0]),int(color[1]),int(color[2])))


	def triangulate(self, view1 , view2):
		match = self.matches[(view1.image_name,view2.image_name)]

		pixel_points1 = np.array([ kp.pt for kp in view1.keypoints])[match.inliers1]
		pixel_points2 = np.array([ kp.pt for kp in view2.keypoints])[match.inliers2]
		img_path = os.path.join(view1.root_dir, "images", view1.image_name+".jpg")
		if(pixel_points1.shape[0] == 0):
			return [],[]
		img = cv2.imread(img_path)[:, :, ::-1]
		print(f" pixel_points1 shape {pixel_points1.shape}")
		pts1_homg = cv2.convertPointsToHomogeneous(pixel_points1)[:,0,:]
		pts2_homg = cv2.convertPointsToHomogeneous(pixel_points2)[:,0,:]

		K_inv = np.linalg.inv(self.K)
		P1 = np.hstack((view1.rotation, view1.translation))
		P2 = np.hstack((view2.rotation, view2.translation))
		reprojection_error1 = []
		reprojection_error2 = []


		for id, pt1 in enumerate(pts1_homg):
			u1_norm = np.matmul( K_inv ,pt1)
			u2_norm = np.matmul(K_inv ,pts2_homg[id])
			point_3d = get_3D_points(P1,u1_norm,P2,u2_norm)
			color = np.array(img[pt1[1].astype(int) ,pt1[0].astype(int)]).reshape((1,3))

			self.colors = np.concatenate((self.colors, color),axis =0 )
			self.points_3d = np.concatenate((self.points_3d, point_3d.T), axis=0)

			reprojection_error1.append(calculateReprojectionError(view1.rotation, view1.translation, self.K, pt1[0:2], point_3d))
			reprojection_error2.append(calculateReprojectionError(view2.rotation, view2.translation, self.K, pts2_homg[id][0:2], point_3d))
			self.point_map[(self.names.index(view1.image_name), match.inliers1[id])] = self.reconstructed_count
			self.point_map[(self.names.index(view2.image_name), match.inliers2[id])] = self.reconstructed_count
			self.completed_pixel_points.append(pt1[0:2])
			self.completed_pixel_points.append(pts2_homg[id][0:2])
			self.camera_indices.append(self.names.index(view1.image_name))
			self.camera_indices.append(self.names.index(view2.image_name))
			self.point3d_indices.append(self.reconstructed_count)
			self.point3d_indices.append(self.reconstructed_count)
			self.reconstructed_count += 1
		return reprojection_error1, reprojection_error2

	def pnp_projection(self, new_view):

		matcher = cv2.BFMatcher(cv2.NORM_L2)
		old_descriptors = []
		for completed_view in self.completed:
			old_descriptors.append(completed_view.descriptors)
		matcher.add(old_descriptors)
		matcher.train()
		matches =matcher.match(queryDescriptors = new_view.descriptors)
		points_3d, points_2d = np.zeros((0,3)), np.zeros((0,2))
		for match in matches:
			old_image_id ,new_kp_idx, old_kp_idx =  match.imgIdx , match.queryIdx, match.trainIdx
			if (old_image_id, old_kp_idx) in self.point_map:
				point_3d_idx = self.point_map[(old_image_id, old_kp_idx)]
				point_3d = self.points_3d[point_3d_idx,:].T.reshape((1,3))
				points_3d = np.concatenate((points_3d,point_3d), axis=0)
				point_2d = np.array(new_view.keypoints[new_kp_idx].pt).T.reshape((1,2))
				points_2d = np.concatenate((points_2d, point_2d), axis=0)

		_,R,T, _ = cv2.solvePnPRansac(points_3d[:,np.newaxis],points_2d[:,np.newaxis],self.K,None,confidence=0.99, reprojectionError=8.0,flags=cv2.SOLVEPNP_DLS)
		R,_ = cv2.Rodrigues(R)
		return R,T