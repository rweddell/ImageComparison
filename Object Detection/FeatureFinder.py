import cv2
import numpy as np
from pathlib import Path

def find_features(filename, num_features=32):
    image_path = Path('Images')/filename
    out_path = Path('Images')
    img = cv2.imread('Images/'+filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sifter = cv2.KAZE_create()
    key_points = sifter.detect(gray)
    key_points = sorted(key_points, key=lambda x: -x.response)[:num_features]
    kps, descriptors = sifter.compute(img, key_points)
    descriptors = descriptors.flatten()
    needed_size = num_features * 64
    img_copy = img.copy()
    cv2.drawKeypoints(image=gray, keypoints=key_points, outImage=img_copy)
    if descriptors.size < needed_size:
        descriptors = np.concatenate([descriptors, np.zeros(needed_size - descriptors.size)])
    cv2.imwrite('keypoints.jpg', img_copy)
    cv2.imshow('keypoints.jpg', img_copy)
    cv2.waitKey()
    return descriptors


lester = find_features('countertop.jpg')
print(lester)