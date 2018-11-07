from ObjectDetection.Detector import detect_objects
from Preprocessing.Preprocessing import clear_folder
from Comparison.ImageCompare import *
from ObjectDetection.FeatureFinder import *
import os


# This is a script to test and run the various forms
# of object detection in this project
# as well as to compare the objects of different images

# Folder indices and image names
folders = ['1', '2']
images = ['soda4.jpg', 'soda4.jpg']

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


print()
"""
sim_color_report = compare_norm_color(obj1, obj2)
print('Color similarity is: ' + str(sim_color_report))

sim_gray_report = compare_norm_gray(obj1, obj2)
print('Grayscale similarity is: ' + str(sim_gray_report))

print()
if sim_color_report > 0.65 and sim_gray_report > 0.65:
    print('The objects are similar')
else:
    print("The objects are not very similar")
"""

detected_directories = ['Found' + n for n in folders]
histo_results = directory_hist(detected_directories[0], detected_directories[1])

print(histo_results)

feature_directories = ['Features' + n for n in folders]
feature_results = directory_keypoints(feature_directories[0], feature_directories[1])

print(feature_results)

# identify features in the object detection results
# stores them in the folder at indices[i]
# how can I point the function to the file without hard-coding the name?
#obj1 = 'Found1/Detected1-objects/bottle-1.jpg'
#obj2 = 'Found2/Detected2-objects/bottle-1.jpg'
#desc1 = kaze_features(obj1, folder=folders[0])
#desc2 = kaze_features(obj2, folder=folders[1])
