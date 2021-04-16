import cv2
import numpy as np
from matplotlib import pyplot as plt
import threading
import sys

#Usage: `python3 featuredetect.py <filename>`
img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = img
img3 = img
img4 = img

def sift():
	sift = cv2.SIFT_create()
	(kps, descs) = sift.detectAndCompute(gray, None)
	cv2.drawKeypoints(gray, kps, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imshow('SIFT Algorithm', img)


def surf():
	surf = cv2.xfeatures2d.SURF_create()
	(kps2, descs2) = surf.detectAndCompute(gray, None)
	cv2.drawKeypoints(gray, kps2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imshow('SURF Algorithm', img2)

def fast():
	fast = cv2.FastFeatureDetector_create()
	kps3 = fast.detect(gray, None)
	cv2.drawKeypoints(gray, kps3, img3, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imshow('FAST Algorithm', img3)

def orb():
	orb = cv2.ORB_create()
	kps4 = orb.detect(gray, None)
	(kps4, des4) = orb.compute(gray, kps4)
	cv2.drawKeypoints(gray, kps4, img4, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imshow('ORB Algorithm', img4)

sift()
surf()
fast()
orb()


cv2.waitKey(0)

cv2.destroyAllWindows()
