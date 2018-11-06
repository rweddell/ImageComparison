
import cv2
import numpy as np
from pathlib import Path


def spectral_saliency(filename):
    img_path = Path('Images/' + filename)
    img = cv2.imread(str(img_path))
    salient = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, sal_map = salient.computeSaliency(img)
    cv2.imshow('Image', img)
    cv2.imshow('Output', sal_map)
    cv2.waitKey()


# haven't been able to get this function to work yet
def fine_grained_saliency(filename):
    img_path = Path('Images/' + filename)
    img = cv2.imread(str(img_path))
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, sal_map = saliency.computeSaliency(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    sal_map = np.array(sal_map * 255, dtype=np.uint8)
    thresh_map = cv2.threshold(sal_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow('Original Image', img)
    cv2.imshow('Saliency Detection', sal_map)
    cv2.imshow('Thresholded', thresh_map)
    cv2.waitKey()


def object_saliency(filename, detections):
    img_path = Path('Images/' + filename)
    model_path = Path('Models/')
    img = cv2.imread(str(img_path))
    saliency = cv2.saliency.ObjectnessBING_create()
    saliency.setTrainingPath(str(model_path))
    success, sal_map = saliency.computeSaliency(img)
    num_detections = sal_map.shape[0]
    print(num_detections)
    output = img.copy()
    for i in range(detections):
        xstart, ystart, xend, yend = sal_map[i].flatten()
        rand_color = np.random.randint(0,255, size=(3,))
        color = [int(c) for c in rand_color]
        cv2.rectangle(output, (xstart, ystart), (xend, yend), color, 2)
        cv2.imshow('Original image', output)
        cv2.waitKey()
