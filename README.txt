
This is a final project for CSC481 Image Processing.

This project is intended to locate the salient object of a given image
and compare that object to the salient object of another given image
to see if they are the same object.

The detect_objects method, which uses retinanet, will find whatever
objects are present in a given image. Those objects are saved as
separate images in a 'Foundx' folder. Those objects are then fed
into the feature extractor which finds different points of interest,
or 'keypoints', in each image. Then we compare histograms of color
images as well as the keypoints to determine the similarity of
the main objects in each image.


Object detection:
    I followed the guide found on this site:
    https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606

    This was a site for a custom AI library for image analysis:
    https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl

    For ObjectDetector to work, you will need to download: resnet50_coco_best_v2.0.1.h5
    Place it in the 'Models' directory.


Saliency:
    There is a tutorial found on this site:
    https://www.pyimagesearch.com/2018/07/16/opencv-saliency-detection/


Feature extraction:
    I found a tutorial on this site:
    https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774


Comparing keypoints and features:
    https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html
