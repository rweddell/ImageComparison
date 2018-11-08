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

# The two images to be compared at the start
images = ['bottle1.jpg', 'bottle2.jpg']

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

detected_directories = ['Found' + n for n in folders]
histo_results = directory_hist(detected_directories[0], detected_directories[1])


detected1 = get_detected('1')
detected2 = get_detected('2')

feature_results1 = []
feature_results2 = []
i=1
for j in detected1:
    feature_results1.append(kaze_features(j, unique=str(i), folder='1', num_features=500))
    i+=1
i=1
for k in detected2:
    feature_results2.append(kaze_features(k, unique=str(i), folder='2',num_features=500))
    i+=1
feature_directories = ['Features' + n for n in folders]
feature_results = []
i=1
for entry1 in feature_results1:
    for entry2 in feature_results2:
        feature_results.append(compare_keypoints(entry1, entry2, str(i)))
        i+=1
#feature_results = compare_keypoints(feature_results1[0], feature_results2[0])
#keypoint_results = directory_keypoints(feature_directories[0], feature_directories[1])

for entry in histo_results:
    print('Histogram similarities : ' + str(entry))

for entry in feature_results:
    print('Average feature distance in image = ' + str(entry))

#for entry in keypoint_results: print('Keypoint similarity : ' + entry)
