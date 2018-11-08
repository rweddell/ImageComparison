import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def compare_norm_gray(pic1, pic2):
    # img1 and img2 should both be numpy arrays at this point
    gray1 = cv2.imread(pic1, cv2.IMREAD_GRAYSCALE)
    gray2 = cv2.imread(pic2, cv2.IMREAD_GRAYSCALE)

    grayhist1 = cv2.calcHist([gray1], [0], None, [256], [0,256])
    grayhist2 = cv2.calcHist([gray2], [0], None, [256], [0,256])

    gray_sim = cv2.compareHist(grayhist1, grayhist2, method=cv2.HISTCMP_CORREL)
    plt.hist(grayhist1, np.linspace(0, 10, max(grayhist1)), label='histogram1')
    plt.hist(grayhist2, np.linspace(0, 10, max(grayhist2)), label='histogram2')
    plt.xlabel('Intensity')
    plt.ylabel('Pixel amount')
    plt.xticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.title('Grayscale histogram comparison')
    plt.show()
    # Attempting to be illumination invariant
    cv2.equalizeHist(grayhist1.astype(np.uint8), grayhist1)
    cv2.equalizeHist(grayhist2.astype(np.uint8), grayhist2)
    cv2.normalize(grayhist1, grayhist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(grayhist2, grayhist2, 0, 1, cv2.NORM_MINMAX)

    # comparing the histograms

    return gray_sim


def compare_norm_color(pic1, pic2):
    # Compares normalized color histograms
    img1 = cv2.imread(pic1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(pic2, cv2.IMREAD_COLOR)
    clrhist1 = cv2.calcHist([img1], [0,1,2], None, [64,64,64], [0, 256, 0, 256, 0, 256])
    clrhist2 = cv2.calcHist([img2], [0,1,2], None, [64,64,64], [0, 256, 0, 256, 0, 256])
    cv2.normalize(clrhist1, clrhist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(clrhist2, clrhist2, 0, 1, cv2.NORM_MINMAX)
    clr_sim = cv2.compareHist(clrhist1, clrhist2, method=cv2.HISTCMP_INTERSECT)
    # should this be subtracted from 1?
    return clr_sim/10


def compare_keypoints(analysis1, analysis2, unique):
    #print('Analyses below:')
    #print(analysis1, analysis2)
    bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
    cv2.imshow('analysis2', analysis2[0])
    cv2.imshow('analysis1', analysis1[0])
    cv2.waitKey()
    matches = bf.match(analysis1[2], analysis2[2])
    print('Length of matches : ' + str(len(matches)))
    good=[]
    for x in matches:
        if x.distance < 2.0:
            good.append(x)
    print('Length of good : ' + str(len(good)))
    matches = sorted(matches, key=lambda x: x.distance)[:10]
    matched_img = cv2.drawMatches(analysis1[0],
                                  analysis1[1],
                                  analysis2[0],
                                  analysis2[1],
                                  matches,
                                  analysis1[0],
                                  flags=2)
    dists = []
    for x in matches:
        dists.append(x.distance)
    avg = np.mean(dists)
    print(dists)
    cv2.imwrite(os.path.join(os.getcwd(), 'matchedimage' + unique + '.jpg'), matched_img)
    # return percentage of good matches
    return ['matchedimage' + unique + '.jpg', avg]


def directory_hist(dir1, dir2):
    current_path = os.getcwd()
    path1 = os.path.join(os.path.join(current_path, dir1), 'Detected1-objects')
    path2 = os.path.join(os.path.join(current_path, dir2), 'Detected2-objects')
    files1 = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
    files2 = [f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]
    #print('Files in Found1 : ' + str(files1))
    #print('Files in Found2 : ' + str(files2))
    comparison_results = []
    for file1 in files1:
        pathfile1 = os.path.join(path1, file1)
        for file2 in files2:
            pathfile2 = os.path.join(path2, file2)
            comparison_results.append([compare_norm_color(pathfile1, pathfile2),
                                       compare_norm_gray(pathfile1, pathfile2),
                                       file1,
                                       file2])
    return comparison_results


def directory_keypoints(dir1, dir2):
    current_path = os.getcwd()
    path1 = os.path.join(current_path, dir1)
    path2 = os.path.join(current_path, dir2)
    print(path1)
    print(path2)
    files1 = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
    files2 = [f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]
    kpmatch_results = []
    i = 1
    for file1 in files1:
        for file2 in files2:
            kpmatch_results.append([compare_keypoints(file1, file2, str(i)), file1, file2])
            i+=1
    return kpmatch_results


def batch_keypoints(features1, features2):
    kpmatch_results = []
    i = 1
    for entry1, in features1:
        for entry2 in features2:
            print(type(entry1[0]), type(entry2[0]))
            kpmatch_results.append([compare_keypoints(entry1, entry2, str(i))])
            i+=1
    return kpmatch_results

