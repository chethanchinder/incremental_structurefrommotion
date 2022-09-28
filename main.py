import os
import cv2
import numpy as np
from feature_matches import match_features
from features_extraction import extractViewFeatures
from structure_from_motion import *
import argparse

if __name__ == '__main__':
    print("Welocome to c-SFM v.01")
    #root_dir = "./../Benchmarking_Camera_Calibration_2008/entry-P10"
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="specify the path of root dir of images")
    root_dir = parser.parse_args().root_dir
    view_features = extractViewFeatures(root_dir)
    view_mathces  = match_features(view_features, root_dir)
    K = np.loadtxt(os.path.join(root_dir, "images", "K.txt"))
    print(f" K type {type(K)} shape {K.shape} ")
    sfm = Structure(view_features, view_mathces, K)
    sfm.reconstruct()