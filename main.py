import os
import cv2
import numpy as np
from motion import getEssentialAndInliers, getCameraFromEssential, isRotationValid
from feature_matches import match_features
from features_extraction import extractViewFeatures

def deletelater():
    root_dir = "Benchmarking_Camera_Calibration_2008/"
    im1 = cv2.imread(root_dir + 'castle-P19/images/0005.jpg', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(root_dir + 'castle-P19/images/0006.jpg', cv2.IMREAD_GRAYSCALE)
    inliers1, inliers2, essentialMat = getEssentialAndInliers(im1, im2)
    R1, R2, T1, T2 = getCameraFromEssential(essentialMat)
    print("R1 ", R1,"R2 ", T1)
    print("T1 ", R2, "T2 ", T2)
    print(" R1 valid ", isRotationValid(R1, 1e-9), " R2 valid? ", isRotationValid(R2, 1e-9))
    P1 = np.append(R1, T1, 1)
    P2 = np.append(R2, T2, 1)
    print(" P1 ", P1)
    print(" P2  ", P2)

if __name__ == '__main__':
    print("Welocome to c-SFM v.01")
    root_dir = "./../Benchmarking_Camera_Calibration_2008/castle-P19"
    view_features = extractViewFeatures(root_dir)
    view_mathces  = match_features(view_features, root_dir)
    K = np.loadtxt(os.path.join(root_dir, "images", "K.txt"))
