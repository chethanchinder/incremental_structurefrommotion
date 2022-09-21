import glob
import numpy as np
import cv2
import os
import pickle

class FeatureMatch:
	def __init__(self, feature1, feature2,match_path):
		self.inliers1 = []
		self.inliers2 = []
		self.indices1 = []
		self.indices2 = []
		self.distances = []
		self.image1 = feature1.image_name
		self.image2 = feature2.image_name
		self.descriptors1 =feature1.descriptors
		self.descriptors2 = feature2.descriptors
		self.keypoints1 = feature1.keypoints
		self.keypoints2 = feature2.keypoints

		self.mathes_path = match_path
		if(os.path.exists(self.mathes_path)):
			self.load_matches()
		else:
			self.match_features()

	def match_features(self):
		brute_force_matcher  = cv2.BFMatcher_create(cv2.NORM_L2, True)
		# index parameters
		FLANN_INDEX = 0
		index_params = dict(algorithm = FLANN_INDEX, trees = 5)
		# search parameters
		search_params = dict(checks = 100)
		flann_matcher = cv2.FlannBasedMatcher(index_params,search_params)
		potential_matches = flann_matcher.knnMatch(np.array(self.descriptors1),np.array(self.descriptors2), k=2)
		## lowes ratio test
		matches = [match1 for match1, match2 in potential_matches if match1.distance > 0.75*match2.distance ]
		## sort matches
		matches = sorted(matches, key= lambda x: x.distance)
		for match in matches:
			self.indices1.append(match.queryIdx)
			self.indices2.append(match.trainIdx)
			self.distances.append(match.distance)
		self.write_matches()

	def load_matches(self):
		print(f"[FeatureMatcher] loading the features matches for images ({self.image1},{self.image2}) ")
		matches =pickle.load(open(self.mathes_path,"rb"))
		for id, (indice1, indice2, distance) in enumerate(matches):
			self.indices1.append(indice1)
			self.indices2.append(indice2)
			self.distances.append(distance)

	def write_matches(self):
		print(f"[FeatureMatcher] writing the features matches for images ({self.image1},{self.image2}) ")
		matches_path_dir = self.mathes_path[:self.mathes_path.rfind("/")]
		if( not os.path.exists(matches_path_dir )):
			os.makedirs(matches_path_dir)
		match_file=open(self.mathes_path, "wb")
		temp_array = []
		for id, (indice1, indice2, distance) in enumerate(zip(self.indices1, self.indices2, self.distances)):
			temp_array.append((indice1, indice2, distance))
		pickle.dump(temp_array,match_file)
		match_file.close()

def match_features(views, root_dir):
	matches = {}
	print(f"[FeatureMatcher] matching view features ")
	for i in range(len(views)):
		for j in range(i+1, len(views)):
			match_path =  os.path.join(root_dir, "feature_matches",views[i].image_name+"+"+views[j].image_name+ ".pkl")
			matches[(views[i].image_name,views[j].image_name)] = FeatureMatch(views[i], views[j], match_path)
	return matches
