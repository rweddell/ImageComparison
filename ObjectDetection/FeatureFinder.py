import cv2
import os
import numpy as np


def get_detected(folder):
    detected_path = os.path.join(os.getcwd(), 'Found' + folder)
    detected_path = os.path.join(detected_path, 'Detected' + folder + '-objects')
    return [f for f in os.listdir(detected_path) if os.path.isfile(os.path.join(detected_path, f))]


def orb_features(filename, unique, folder='1', num_features=200):
    current_path = os.path.join(os.getcwd(), 'Found' + folder)
    in_path = os.path.join(current_path, 'Detected' + folder + '-objects')
    out_path = os.path.join(os.getcwd(), 'Features' + folder)
    img = cv2.imread(os.path.join(in_path, filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create()
    kps, descriptors = detector.detectAndCompute(img, mask=None)
    kps = sorted(kps, key=lambda x: -x.response)[:num_features]
    cv2.drawKeypoints(image=gray,
                      keypoints=kps,
                      outImage=img)
                      #flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    needed_size = num_features * 64
    #if descriptors.size < needed_size:
    #    descriptors = np.concatenate([descriptors, np.zeros(needed_size - descriptors.size)])
    cv2.imwrite(os.path.join(out_path, 'feats' + folder + unique + '.jpg'), img)
    return [img, kps, descriptors]


def kaze_features(filename, unique, folder='1', num_features=200):
    current_path = os.path.join(os.getcwd(), 'Found' + folder)
    in_path = os.path.join(current_path, 'Detected' + folder + '-objects')
    out_path = os.path.join(os.getcwd(), 'Features' + folder)
    img = cv2.imread(os.path.join(in_path, filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.KAZE_create()
    kps, descriptors = detector.detectAndCompute(img, mask=None)
    kps = sorted(kps, key=lambda x: -x.response)[:num_features]
    cv2.drawKeypoints(image=gray,
                      keypoints=kps,
                      outImage=img,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    needed_size = num_features * 64
    #if descriptors.size < needed_size:
    #    descriptors = np.concatenate([descriptors, np.zeros(needed_size - descriptors.size)])
    cv2.imwrite(os.path.join(out_path, 'feats' + folder + unique + '.jpg'), img)
    return [img, kps, descriptors]
