from imageai.Detection import ObjectDetection
from pathlib import Path
import cv2
import os


def detect_objects(filename):
    current_path = os.getcwd()
    img_path = Path('Images/'+filename)
    out_path = Path('Images/FoundObjects')
    model_path = Path('Models/resnet50_coco_best_v2.0.1.h5')
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path)
    detector.loadModel()

    # returning a list of predicted objects and their traits
    detections, extracted = detector.detectObjectsFromImage(input_image=img_path,
                                                            output_image_path=os.path.join(current_path,'FoundObjects'),
                                                            minimum_percentage_probability=30,
                                                            extract_detected_objects=True)
    '''
    for thing, thingpath in zip(detections, extracted):
        print(thing['name'], ':', thing['percentage_probability'], ':', thing['box_points'])
        print('object image saved in ' + thingpath)
        print('---------------------------------')
    '''
    return zip(detections, extracted)

detect_objects('countertop.jpg')
