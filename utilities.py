import cv2

def esssentialFromFundamental(F, K):
	return K.t()*F*K

