from ObjectDetection.Detector import detect_objects
from Preprocessing.Preprocessing import clear_folder
from Comparison.ImageCompare import *
from ObjectDetection.FeatureFinder import find_features
import os


# This is a script to test and run the various forms
# of object detection in this project
# as well as to compare the objects of different images

# Folder indices and image names
folders = ['1', '2']
images = ['spritecan.jpg', 'soda3.jpg']

# Clears the folders so that old computations are erased
for x in folders:
    clear_folder('Found'+x)

# find objects in the images and send them to
# the folder in indices[i]
objects1 = detect_objects(images[0], folders[0])
objects2 = detect_objects(images[1], folders[1])

# Clear feature folders so that old computations are erased
for x in folders:
    clear_folder('Features'+x)

# identify features in the object detection results
# stores them in the folder at indices[i]
# how can I point the function to the file without hard-coding the name?
obj1 = 'Found1/Detected1-objects/bottle-1.jpg'
obj2 = 'Found2/Detected2-objects/bottle-1.jpg'
desc1 = find_features(obj1, folder=folders[0])
desc2 = find_features(obj2, folder=folders[1])



