import glob
import numpy as np
import cv2
import os
import pickle
import numpy as np

class FeatureExtractor:
	def __init__(self, file_path, featurePath, root_dir):
		self.image_path = file_path
		self.image_name = file_path[file_path.rfind("/")+1:file_path.rfind(".")]
		self.keypoints = []
		self.descriptors = []
		self.feature_path = os.path.join(featurePath, self.image_name+".pkl")
		self.root_dir =root_dir
		self.rotation = np.zeros((3,3))
		self.translation = np.zeros((3,1))
		self.extractKpAndDescriptors()

	def extractKpAndDescriptors(self):
		if(os.path.exists(self.feature_path)):
			self.keypoints , self.descriptors = self.featureLoader()
		else:
			detector = cv2.SIFT_create()
			image = cv2.imread(self.image_path)
			self.keypoints, self.descriptors = detector.detectAndCompute(image, None)
			self.featureWriter()

	def featureWriter(self):
		print(f"[FeatureExtractor] writing the features for image {self.image_name} ")
		if not os.path.exists(os.path.join(self.root_dir,"features")):
			os.makedirs(os.path.join(self.root_dir,"features"))
		featureArray = []
		for id, kp in enumerate(self.keypoints):
			featureArray.append( (kp.pt, kp.size, kp.angle ,kp.response, kp.octave,kp.class_id, self.descriptors[id]))
		feature_file = open(self.feature_path, "wb")
		pickle.dump(featureArray, feature_file)
		feature_file.close()

	def featureLoader(self):
		print(f"[FeatureExtractor] loading the features for image {self.image_name} ")
		keypoints = []
		descriptors = []

		features  = pickle.load(open(self.feature_path, "rb"))
		for feature in features:
			keypoints.append(cv2.KeyPoint(x= feature[0][0],y= feature[0][1],size = feature[1], angle = feature[2],
										  response = feature[3], octave = feature[4], class_id = feature[5]))
			descriptors.append(feature[6])
		return keypoints, np.array(descriptors)

def extractViewFeatures(root_dir):
	file_paths = sorted(glob.glob(os.path.join(root_dir,"images", "*.jpg")))
	viewList = []
	print(f"[FeatureExtractor] extracting view features ")
	for id, file_path in  enumerate(file_paths):
		featuresPath = os.path.join(root_dir, "features")
		viewList.append(FeatureExtractor(file_path, featuresPath, root_dir))
	return viewList

#root_dir = "./Benchmarking_Camera_Calibration_2008/castle-P19"
# extractViewFeatures(root_dir)

