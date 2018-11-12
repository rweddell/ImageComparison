from imageai.Detection import ObjectDetection
from pathlib import Path
import os


def detect_objects(filename, folder='1'):
    current_path = os.getcwd()
    img_path = os.path.join(os.path.join(current_path, 'Images'),filename)
    out_path = os.path.join(current_path, 'Found' + folder)
    model_path = Path('Models/resnet50_coco_best_v2.0.1.h5')
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path)
    detector.loadModel()
    # returning a list of predicted objects and their traits
    # also fills 'Foundx' with sub-images of objects
    detections, extracted = detector.detectObjectsFromImage(input_image=img_path,
                                                            output_image_path=os.path.join(out_path,
                                                                                           'Detected'+folder),
                                                            minimum_percentage_probability=55,
                                                            extract_detected_objects=True)
    for thing, thingpath in zip(detections, extracted):
        print(thing['name'], ':', thing['percentage_probability'], ':', thing['box_points'])
        print('object image saved in ' + thingpath)
        print('------------------------------------------')

    return zip(detections, extracted)

business = detect_objects('bottle2.jpg', '2')

