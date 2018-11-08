import cv2
import os
import shutil


def resize_image(filenames, reduceby=0.5):
    # can make the images a more manageable size
    for pic in filenames:
        img = cv2.imread('Images/' + pic)
        newimg = cv2.resize(img, None, fx=reduceby, fy=reduceby, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('s'+pic, newimg)


def clear_found():
    # deletes and remakes 'Found' folder
    current_path = os.getcwd()
    try:
        shutil.rmtree(os.path.join(current_path, 'Found'))
    except FileNotFoundError:
        print('"Found" might not exist yet')
    os.makedirs(os.path.join(current_path, 'Found'))


def clear_features():
    # deletes and remakes the 'Features' folder
    current_path = os.getcwd()
    try:
        shutil.rmtree(os.path.join(current_path, 'Features'))
    except FileNotFoundError:
        print('"Features" might not exist yet')
    os.makedirs(os.path.join(current_path, 'Features'))


def clear_folder(fname):
    # clears and remakes the desired directory
    # be careful with this...
    current_path = os.getcwd()
    try:
        shutil.rmtree(os.path.join(current_path, fname))
    except (FileNotFoundError):
        print(fname + ' might not exist yet')
    try:
        os.makedirs(os.path.join(current_path, fname))
    except (PermissionError):
        print('Something about permissions.')