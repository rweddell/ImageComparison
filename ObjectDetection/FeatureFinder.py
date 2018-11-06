import cv2
import os
import numpy as np


def find_features(filename, folder='1', num_features=32):
    current_path = os.getcwd()
    out_path = os.path.join(current_path, 'Features' + folder)
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sifter = cv2.KAZE_create()
    key_points = sifter.detect(gray)
    key_points = sorted(key_points, key=lambda x: -x.response)[:num_features]
    kps, descriptors = sifter.compute(img, key_points)
    print(type(kps))
    print(kps)
    descriptors = descriptors.flatten()
    needed_size = num_features * 64
    cv2.drawKeypoints(image=gray, keypoints=key_points, outImage=img)
    if descriptors.size < needed_size:
        descriptors = np.concatenate([descriptors, np.zeros(needed_size - descriptors.size)])
    cv2.imwrite(os.path.join(out_path, 'feats' + folder + '.jpg'), img)
    return descriptors, img
