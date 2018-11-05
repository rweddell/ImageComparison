from ObjectDetection.Detector import detect_objects
from ObjectDetection.FeatureFinder import find_features
from ObjectDetection.Saliency import object_saliency
import cv2

# This is a script to test and run the various forms
# of object detection in this project
# as well as to compare the objects of different images
# This is a script to test and run the various forms
# of object detection in this project
# as well as to compare the objects of different images

image1 = cv2.imread('Images/droid.jpg')
image2 = cv2.imread('Images/dewcan.jpg')

cv2.imshow('droid', image1)
cv2.imshow('dewcan', image2)
cv2.waitKey()

img1_objects = detect_objects()

